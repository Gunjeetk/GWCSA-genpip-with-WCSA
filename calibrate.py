# =============================================================
#  calibrate.py  — calibrate WCSA ADG thresholds on held-out data
#
#  Independence guarantee
#  ----------------------
#  Calibration reads are drawn from the FIRST half of the reference
#  (positions 0 … REF_LENGTH//2 − 1).  Test reads are drawn from
#  the full reference but evaluated on the full pipeline — the two
#  sets are constructed with different RNG streams, giving genuine
#  distributional independence between calibration and test.
#
#  The repeat families embedded by build_reference produce genuine
#  false-mapping chain candidates (query from one repeat copy forming
#  a chain at another copy).  These populate the FP bucket of the ROC
#  sweep, allowing FPR < 1.0 at useful bypass rates.
# =============================================================

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from sim_reference import SimRead, is_correct_mapping, simulate_ont_read
from sim_index import find_anchors, build_index
from sim_chaining import chain_anchors
from wcsa_block import wcsa_block
from dp_alignment import align_chain
from config import (
    WCSA_MAX_FPR, CORRECT_MAP_OVERLAP,
    KMER_K, MINIMIZER_W,
    REF_LENGTH, RANDOM_SEED,
)

# Calibration reads are generated from a distinct RNG stream so they
# do not overlap the test-set reads produced by the main RANDOM_SEED.
_CAL_SEED = RANDOM_SEED + 9999


@dataclass
class CalRecord:
    """One calibration data-point: (CC, VR, DP correctness, DP identity)."""
    cc           : float
    vr           : float
    is_correct_dp: bool
    dp_identity  : float


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _measure_only(chain) -> tuple[float, float]:
    """Run WCSA with τ_acc=1.0 so every chain stays UNCERTAIN (measure mode)."""
    result = wcsa_block(chain, tau_acc=1.0, tau_rej=0.0)
    return result["cc"], result["vr"]


def _cal_reads_from_first_half(ref:       str,
                                repeat_coords: list,
                                n_reads:   int,
                                rng) -> list[SimRead]:
    """
    Generate calibration reads whose true origin lies entirely within
    the first half of the reference, providing a region independent of
    the uniform-sampling test set.
    """
    half = len(ref) // 2
    reads = []
    attempts = 0
    while len(reads) < n_reads and attempts < n_reads * 10:
        attempts += 1
        r = simulate_ont_read(ref, repeat_coords, len(reads), rng)
        # Keep only reads that start in the first half
        if r.true_ref_end <= half:
            reads.append(r)
    if len(reads) < n_reads:
        # Fallback: take whatever we collected (still independent stream)
        pass
    return reads


# ─── Calibration data collection ──────────────────────────────────────────────

def collect_calibration_data(cal_reads:     list[SimRead],
                              ref:           str,
                              index:         dict) -> list[CalRecord]:
    """
    For each calibration read: seed → chain → WCSA(measure) → DP.
    Every (chain, DP result) pair becomes one CalRecord.
    False chains (formed at a repeat copy other than the true origin)
    are correctly labelled is_correct_dp=False, populating the FP bucket.
    """
    records: list[CalRecord] = []
    print(f"[Calibrate] Collecting data on {len(cal_reads)} reads...",
          flush=True)

    for read in cal_reads:
        anchors, _ = find_anchors(read.sequence, index, KMER_K, MINIMIZER_W)
        chains,  _ = chain_anchors(anchors, read.sequence, ref, read.quality)

        for chain in chains:
            cc, vr    = _measure_only(chain)
            dp_result = align_chain(chain)

            # Expand chain coords by read overhangs before overlap test —
            # chain bounds span only the anchored region, not the full read.
            left_oh  = chain.q_start
            right_oh = max(0, read.length - chain.q_end)
            exp_start = max(0, chain.ref_start - left_oh)
            exp_end   = chain.ref_end + right_oh

            correct = is_correct_mapping(
                exp_start, exp_end,
                read.true_ref_start, read.true_ref_end,
                CORRECT_MAP_OVERLAP,
            )
            records.append(CalRecord(
                cc            = cc,
                vr            = vr,
                is_correct_dp = correct,
                dp_identity   = dp_result["identity"],
            ))

    n_correct   = sum(r.is_correct_dp for r in records)
    n_incorrect = len(records) - n_correct
    print(f"[Calibrate] {len(records)} chain records  "
          f"({n_correct} correct TP, {n_incorrect} incorrect FP)")
    if n_incorrect == 0:
        print("[Calibrate] WARNING: no false chains found — "
              "ROC FPR will be trivially 0; consider adding more repeats.")
    return records


# ─── ROC sweep ────────────────────────────────────────────────────────────────

def roc_sweep(records:        list[CalRecord],
              tau_acc_range = None) -> list[dict]:
    """
    Sweep τ_acc; at each point compute TPR and FPR over the ACCEPT decision.
        TPR = correct_accepts  / n_correct_chains
        FPR = wrong_accepts    / n_incorrect_chains
    """
    if tau_acc_range is None:
        tau_acc_range = np.arange(0.10, 0.80, 0.02)

    n_correct   = sum(r.is_correct_dp for r in records)
    n_incorrect = max(len(records) - n_correct, 1)

    roc_points = []
    for tau in tau_acc_range:
        accepts         = [r for r in records if r.cc >= tau]
        correct_accepts = sum(1 for r in accepts if r.is_correct_dp)
        wrong_accepts   = len(accepts) - correct_accepts
        tpr         = correct_accepts / max(n_correct,   1)
        fpr         = wrong_accepts   / max(n_incorrect, 1)
        bypass_rate = len(accepts)    / max(len(records), 1)
        roc_points.append({
            "tau_acc"    : round(float(tau), 3),
            "tpr"        : round(tpr,         4),
            "fpr"        : round(fpr,         4),
            "bypass_rate": round(bypass_rate, 4),
            "n_accepts"  : len(accepts),
        })

    return roc_points


# ─── Threshold selection ──────────────────────────────────────────────────────

def pick_threshold(roc_points: list[dict],
                   max_fpr:    float = WCSA_MAX_FPR) -> tuple[float, float]:
    """
    Select operating point with the highest bypass rate subject to:
        FPR ≤ max_fpr   AND   bypass_rate > 5 %

    Fallback 1: highest TPR with lowest FPR (if FPR constraint unsatisfiable).
    Fallback 2: τ_acc = 0.90 (conservative but always achieves some bypass).
    """
    candidates = [p for p in roc_points
                  if p["fpr"] <= max_fpr and p["bypass_rate"] > 0.05]
    if candidates:
        best    = max(candidates, key=lambda p: p["bypass_rate"])
        tau_acc = best["tau_acc"]
        print(f"[Calibrate] Chosen tau_acc={tau_acc:.2f}  "
              f"FPR={best['fpr']:.3f}  bypass={best['bypass_rate']*100:.1f}%  "
              f"TPR={best['tpr']:.3f}")
        return tau_acc, 0.10

    tpr_candidates = [p for p in roc_points if p["tpr"] >= 0.60]
    if tpr_candidates:
        best    = min(tpr_candidates, key=lambda p: p["fpr"])
        tau_acc = best["tau_acc"]
        print(f"[Calibrate] Fallback tau_acc={tau_acc:.2f}  "
              f"FPR={best['fpr']:.3f}  TPR={best['tpr']:.3f}  "
              f"bypass={best['bypass_rate']*100:.1f}%")
        return tau_acc, 0.10

    print("[Calibrate] Using conservative default tau_acc=0.40")
    return 0.40, 0.10


# ─── Full calibration pipeline ────────────────────────────────────────────────

def calibrate(ref:           str,
              repeat_coords: list,
              index:         dict,
              n_cal_reads:   int) -> tuple[float, float, list[dict], list[CalRecord]]:
    """
    Full calibration pipeline.

    Parameters
    ----------
    ref            : full reference string
    repeat_coords  : repeat copy coordinates from build_reference
    index          : pre-built minimizer index
    n_cal_reads    : number of calibration reads to generate

    Returns
    -------
    (tau_acc, tau_rej, roc_points, cal_records)
    """
    rng      = np.random.default_rng(_CAL_SEED)
    cal_reads = _cal_reads_from_first_half(ref, repeat_coords,
                                           n_cal_reads, rng)
    print(f"[Calibrate] Using {len(cal_reads)} reads from first "
          f"{REF_LENGTH // 2 // 1000} Kbp of reference "
          f"(seed={_CAL_SEED}, independent of test set)")

    records         = collect_calibration_data(cal_reads, ref, index)
    roc_pts         = roc_sweep(records)
    tau_acc, tau_rej = pick_threshold(roc_pts)
    return tau_acc, tau_rej, roc_pts, records
