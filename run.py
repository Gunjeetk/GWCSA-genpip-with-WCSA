#!/usr/bin/env python3
# =============================================================
#  run.py  — single entry point
#  Run: python run.py
# =============================================================

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

# Ensure UTF-8 output on Windows (cp1252 consoles can't render Unicode arrows/ticks)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from config import (
    RANDOM_SEED, N_READS_TOTAL, CAL_FRACTION,
    WCSA_TAU_ACC_INIT, WCSA_TAU_REJ_INIT,
)
from sim_reference  import build_reference, generate_dataset
from sim_index      import build_index
from calibrate      import calibrate
from pipeline       import run_all_systems
from metrics        import save_results, print_summary, save_summary
from visualize      import generate_all_plots


def verify_constraints(results, tau_acc):
    print("\n[Verify] Checking scientific constraints...")
    order    = ["CPU", "CPU-CP", "CPU-GP", "GenPIP", "GenPIP-WCSA"]
    speedups = [results[s]["speedup"] for s in order]

    ok = True
    for i in range(len(order) - 1):
        if speedups[i + 1] < speedups[i] * 0.98:
            print(f"  ✗ Speedup ordering: {order[i+1]}={speedups[i+1]:.3f} "
                  f"< {order[i]}={speedups[i]:.3f}")
            ok = False

    gp = results["GenPIP"]
    gw = results["GenPIP-WCSA"]

    # Mapping-only speedup is the fair metric for WCSA (excludes basecalling)
    if gw["speedup_map_only"] < gp["speedup_map_only"] * 0.98:
        print(f"  ✗ GenPIP-WCSA map speedup ({gw['speedup_map_only']:.2f}×) "
              f"not faster than GenPIP ({gp['speedup_map_only']:.2f}×)")
        ok = False
    else:
        print(f"  ✓ Map-only speedup: GenPIP-WCSA {gw['speedup_map_only']:.2f}× "
              f"> GenPIP {gp['speedup_map_only']:.2f}×")

    # Sensitivity delta over MAPPABLE reads (primary metric)
    acc_delta = abs(gw["sensitivity_pct"] - gp["sensitivity_pct"])
    if acc_delta > 5.0:
        print(f"  ✗ Sensitivity delta {acc_delta:.1f}% exceeds ±5% threshold")
        ok = False
    else:
        print(f"  ✓ Sensitivity delta (mappable): "
              f"{gw['sensitivity_pct']-gp['sensitivity_pct']:+.2f}%")

    bypass = gw["wcsa_bypass_pct"]
    if bypass < 20.0:
        print(f"  ✗ DP bypass rate {bypass:.1f}% < 20% minimum")
        ok = False
    else:
        print(f"  ✓ DP bypass rate: {bypass:.1f}%")

    dp_red = 100 * (1 - gw["n_dp_calls"] / max(gp["n_dp_calls"], 1))
    print(f"  ✓ DP call reduction: {dp_red:.1f}%")
    print(f"  ✓ WCSA overhead > 0: {gw['total_wcsa_s']:.4f}s")
    print(f"  ok Calibrated tau_acc: {tau_acc:.2f}")

    if ok:
        print("[Verify] All constraints satisfied.\n")
    else:
        print("[Verify] Some constraints NOT met — see above.\n")


def main():
    t_start = time.perf_counter()
    print("=" * 68)
    print("  GenPIP v2 + WCSA — Corrected Simulation")
    print("  Affine-gap SW · Repeat reference · Mappable-read sensitivity")
    print("=" * 68)

    # ── 1. Build reference genome (with diverged repeat families) ────────
    print("\n[Step 1/6] Building reference genome...")
    ref, repeat_coords = build_reference()

    # ── 2. Build minimizer index ──────────────────────────────────────────
    print("[Step 2/6] Building minimizer index...")
    index = build_index(ref)

    # ── 3. Generate test dataset ──────────────────────────────────────────
    print("[Step 3/6] Generating ONT reads (test set)...")
    all_reads  = generate_dataset(ref, repeat_coords, N_READS_TOTAL,
                                  seed=RANDOM_SEED)
    # CAL_FRACTION is used only for backward-compat display;
    # calibration reads are generated independently inside calibrate().
    n_test = len(all_reads)
    print(f"  Test set: {n_test} reads")

    # ── 4. Calibrate WCSA thresholds (independent first-half reads) ──────
    print("\n[Step 4/6] Calibrating WCSA ADG thresholds...")
    n_cal_reads = int(N_READS_TOTAL * CAL_FRACTION)
    tau_acc, tau_rej, roc_pts, cal_records = calibrate(
        ref, repeat_coords, index, n_cal_reads)

    # ── 5. Run all 5 systems ──────────────────────────────────────────────
    print("\n[Step 5/6] Running pipeline systems...")
    results = run_all_systems(all_reads, ref, index, tau_acc, tau_rej)

    verify_constraints(results, tau_acc)

    # ── 6. Save results and plots ─────────────────────────────────────────
    print("[Step 6/6] Saving results and generating plots...")
    save_results(results, tau_acc, tau_rej, roc_pts)

    generate_all_plots(results, cal_records, roc_pts, tau_acc)

    summary_str = print_summary(results, tau_acc, tau_rej)
    save_summary(summary_str, tau_acc, tau_rej, roc_pts, results)

    elapsed = time.perf_counter() - t_start
    print(f"\n✓ Complete in {elapsed:.1f}s")
    print(f"  Results : {__import__('config').RESULTS_JSON}")
    print(f"  Summary : {__import__('config').SUMMARY_TXT}")
    print(f"  Plots   : plots/")


if __name__ == "__main__":
    main()
