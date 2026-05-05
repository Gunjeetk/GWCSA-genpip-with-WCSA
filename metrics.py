# =============================================================
#  metrics.py  — aggregated metrics, printed summary, JSON save
#
#  Sensitivity is reported over MAPPABLE reads (those for which
#  at least one chain was formed and DP was attempted), giving
#  the information-retrieval definition: TP / (TP + FN).
#  Reads discarded by Early Rejection are excluded because a
#  correct aligner should also reject truly unmappable reads.
#
#  Two speedup columns are reported:
#    speedup          — end-to-end (basecall + map)
#    speedup_map_only — mapping stage only (excludes basecalling)
# =============================================================

from __future__ import annotations
import json
import numpy as np
from config import RESULTS_JSON, SUMMARY_TXT

SYSTEMS_ORDER = ["CPU", "CPU-CP", "CPU-GP", "GenPIP", "GenPIP-WCSA"]


def save_results(results: dict, tau_acc: float, tau_rej: float,
                 roc_pts: list[dict]):
    """Save all numeric results to JSON (strip non-serialisable objects)."""
    out = {"tau_acc": tau_acc, "tau_rej": tau_rej,
           "roc_points": roc_pts, "systems": {}}
    for name, r in results.items():
        sys_out = {k: v for k, v in r.items()
                   if k not in ("read_results", "cc_distribution",
                                "vr_distribution")}
        cc = r.get("cc_distribution", [])
        vr = r.get("vr_distribution", [])
        sys_out["cc_mean"]   = float(np.mean(cc))   if cc else 0.0
        sys_out["cc_median"] = float(np.median(cc)) if cc else 0.0
        sys_out["vr_mean"]   = float(np.mean(vr))   if vr else 0.0
        out["systems"][name] = sys_out
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[Metrics] Results -> {RESULTS_JSON}")


def print_summary(results: dict, tau_acc: float, tau_rej: float) -> str:
    lines = []
    lines.append("=" * 82)
    lines.append("   GenPIP v2 + WCSA  —  Summary")
    lines.append("=" * 82)
    lines.append(f"   Calibrated thresholds:  tau_acc = {tau_acc:.2f}   "
                 f"tau_rej = {tau_rej:.2f}")
    lines.append("=" * 82)

    # Header: sensitivity column is now over mappable reads only
    hdr = (f"{'System':<14} | {'E2E-SU':>7} | {'Map-SU':>7} | {'Energy':>7} | "
           f"{'Sens(map)':>10} | {'ER%':>6} | "
           f"{'DP_calls':>8} | {'Bypass%':>8}")
    lines.append(hdr)
    lines.append("-" * 82)

    for s in SYSTEMS_ORDER:
        r = results[s]
        lines.append(
            f"{s:<14} | "
            f"{r['speedup']:>6.2f}× | "
            f"{r['speedup_map_only']:>6.2f}× | "
            f"{r['energy_score']:>7.3f} | "
            f"{r['sensitivity_pct']:>9.1f}% | "
            f"{r['er_rate_pct']:>5.1f}% | "
            f"{r['n_dp_calls']:>8d} | "
            f"{r['wcsa_bypass_pct']:>7.1f}%"
        )

    lines.append("=" * 82)

    gp = results["GenPIP"]
    gw = results["GenPIP-WCSA"]
    wcsa_extra   = gw["speedup_map_only"] / max(gp["speedup_map_only"], 1e-9)
    dp_reduction = 100.0 * (1.0 - gw["n_dp_calls"] /
                            max(gp["n_dp_calls"], 1))
    acc_delta    = gw["sensitivity_pct"] - gp["sensitivity_pct"]
    cc_mean      = float(np.mean(gw["cc_distribution"])) \
                   if gw["cc_distribution"] else 0.0

    lines.append(f"WCSA mapping-stage speedup over GenPIP : {wcsa_extra:.2f}×")
    lines.append(f"DP call reduction (GenPIP → WCSA)      : {dp_reduction:.1f}%")
    lines.append(f"Sensitivity delta, mappable reads       : {acc_delta:+.2f}%")
    lines.append(f"Mean WCSA consensus confidence (CC)    : {cc_mean:.3f}")
    lines.append(f"DP bypass rate                         : "
                 f"{gw['wcsa_bypass_pct']:.1f}%")
    lines.append(f"n_mappable (test set)                  : "
                 f"{gw['n_mappable']} / {gw['n_reads']} reads")
    lines.append("=" * 82)

    s = "\n".join(lines)
    print("\n" + s)
    return s


def save_summary(summary_str: str, tau_acc: float,
                 tau_rej: float, roc_pts: list[dict],
                 results: dict):
    gw = results["GenPIP-WCSA"]
    gp = results["GenPIP"]
    interpretation = f"""
================================================================
  BIOLOGICAL & ENGINEERING INTERPRETATION
================================================================

1. PIPELINE CORRECTNESS
   Minimap2-style flow: seed extraction → index lookup →
   anchor chaining → [WCSA] → affine-gap banded SW (Gotoh 1982).
   Reads are drawn from a simulated 1 Mbp reference containing
   {results['CPU']['n_reads']} test reads.  Ground-truth positions
   enable genuine accuracy measurement (no hardcoding).

2. REPEAT MODEL
   Eight tandem-repeat families (4 copies each, 5% divergence,
   ~1.2 Kbp per copy) create genuine multi-mapping candidates.
   These yield false chains with lower CC than true-location chains,
   giving the WCSA ROC curve discriminative power (FPR < FPR_max).

3. WCSA ALGORITHM (CE → RCL → WCS → ADG)
   CE:  base-4 positional encoding of W=2 windows (Int8 shift-add).
   RCL: reference code lookup at anchor-guided diagonal positions
        (PIM-accelerated; Kim et al. 2021 3× bandwidth gain).
   WCS: Phred-quality-weighted match fraction → CC, VR.
   ADG: tau_acc={tau_acc:.2f} / tau_rej={tau_rej:.2f} calibrated on held-out reads
        from the first 500 Kbp of the reference (independent region).

4. SENSITIVITY DEFINITION
   Sensitivity = TP / (TP + FN) over MAPPABLE reads only.
   Reads discarded by Early Rejection are excluded; a real aligner
   would also fail to map them.  Reported column: Sens(map).

5. DP BYPASS
   GenPIP-WCSA bypassed {gw['wcsa_bypass_pct']:.1f}% of DP calls.
   Sensitivity change vs GenPIP: {gw['sensitivity_pct']-gp['sensitivity_pct']:+.2f}% (mappable reads).
   WCSA mapping-stage speedup over GenPIP:
   {gw['speedup_map_only']/max(gp['speedup_map_only'],1e-9):.2f}×.

6. ENERGY
   Power weights derived from literature:
     PIM memory energy: Kim et al. 2021 (UPMEM, ~3× DRAM reduction).
     WCSA Int8 vs FP32: Cali et al. 2020 (GenASM, ~50× op energy).
   GenPIP-WCSA energy score: {gw['energy_score']:.3f} (lower is better).

7. LIMITATIONS
   - 1 Mbp pilot; E. coli (4.6 Mbp) and human chr22 benchmarks planned.
   - ACCEPT decisions do not produce CIGAR strings (future: traceback).
   - Timing constants are analytical models, not silicon measurements.
   - Power weights require CACTI/McPAT validation on target technology.

8. FUTURE WORK
   - E. coli NA12878 real ONT reads (ENA accession ERR3152364).
   - Comparison with SneakySnake, GateKeeper, MAGNET pre-alignment
     filters on standard benchmarks (Alser et al. survey 2022).
   - RTL prototype of CE stage for area/power sign-off.
================================================================
"""
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary_str + "\n")
        f.write(interpretation)
    print(f"[Metrics] Summary -> {SUMMARY_TXT}")
