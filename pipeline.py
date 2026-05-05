# =============================================================
#  pipeline.py  — run all 5 systems on the same test dataset
#
#  System definitions:
#   1. CPU          — sequential basecall + full DP, no features
#   2. CPU-CP       — chunk pipelining (overlap basecall + map)
#   3. CPU-GP       — CP + Early Rejection of low-quality chunks
#   4. GenPIP       — CP + ER + PIM (seed lookup accelerated)
#   5. GenPIP-WCSA  — CP + ER + PIM + WCSA block before DP
#
#  Sensitivity is computed over MAPPABLE reads only (reads where
#  at least one chain is found and DP is attempted), matching
#  the information-retrieval definition: TP / (TP + FN).
#  End-to-end and mapping-only speedups are both reported.
# =============================================================

from __future__ import annotations
import time
import numpy as np
from dataclasses import dataclass, field
from config import (
    POWER_WEIGHT, CORRECT_MAP_OVERLAP, CHUNK_SIZE,
    T_BASECALL, T_MINIMIZER, T_SEED_LOOKUP, T_CHAIN_DP,
    T_ANCHOR, T_ALIGN_DP, T_OVERHEAD,
    PIM_SEED_SPEEDUP, PIM_RCL_SPEEDUP,
    KMER_K, MINIMIZER_W,
)
from sim_reference import SimRead, is_correct_mapping
from sim_index import find_anchors
from sim_chaining import chain_anchors
from wcsa_block import wcsa_block
from dp_alignment import align_chain


# ─── Per-read result ──────────────────────────────────────────────────────────

@dataclass
class ReadResult:
    read_id         : int
    is_mapped       : bool  = False
    is_correct      : bool  = False
    is_er_rejected  : bool  = False   # True → all chunks rejected by ER
    has_chain        : bool  = False  # True → at least one chain was formed
    pred_ref_start  : int   = -1
    pred_ref_end    : int   = -1
    pred_strand     : int   = 0
    mapq            : int   = 0
    source          : str   = "UNMAPPED"  # WCSA_ACCEPT | DP | UNMAPPED

    # Timing (modelled, seconds)
    basecall_time   : float = 0.0
    seed_time       : float = 0.0
    chain_time      : float = 0.0
    wcsa_ce_time    : float = 0.0
    wcsa_rcl_time   : float = 0.0
    wcsa_wcs_time   : float = 0.0
    wcsa_adg_time   : float = 0.0
    wcsa_total_time : float = 0.0
    dp_time         : float = 0.0
    overhead_time   : float = 0.0
    total_map_time  : float = 0.0
    total_time      : float = 0.0

    # Call counts
    n_chains        : int   = 0
    n_dp_calls      : int   = 0
    n_wcsa_accept   : int   = 0
    n_wcsa_reject   : int   = 0
    n_wcsa_uncertain: int   = 0
    n_er_rejected   : int   = 0

    # WCSA scores
    cc_scores       : list  = field(default_factory=list)
    vr_scores       : list  = field(default_factory=list)


# ─── Pipeline for one read ────────────────────────────────────────────────────

def process_read(read:           SimRead,
                 ref:            str,
                 index:          dict,
                 use_pipelining: bool,
                 use_er:         bool,
                 use_pim:        bool,
                 use_wcsa:       bool,
                 tau_acc:        float = 0.85,
                 tau_rej:        float = 0.40) -> ReadResult:

    rr = ReadResult(read_id=read.read_id)
    rr.basecall_time = read.basecall_time

    # ── Early Rejection (chunk-level) ────────────────────────────────────
    n_er = sum(1 for c in read.chunks if c.is_low_qual)
    rr.n_er_rejected = n_er
    if use_er and n_er == len(read.chunks):
        rr.is_er_rejected = True
        rr.overhead_time  = len(read.sequence) * T_OVERHEAD
        rr.total_map_time = rr.overhead_time
        rr.total_time     = _pipeline_time(rr.basecall_time,
                                           rr.total_map_time,
                                           use_pipelining,
                                           len(read.chunks))
        return rr

    # ── Seed lookup ──────────────────────────────────────────────────────
    pim_s   = PIM_SEED_SPEEDUP if use_pim else 1.0
    anchors, seed_t = find_anchors(read.sequence, index,
                                   KMER_K, MINIMIZER_W,
                                   pim_speedup=pim_s)
    rr.seed_time = seed_t

    # ── Chaining DP ──────────────────────────────────────────────────────
    chains, chain_t = chain_anchors(anchors, read.sequence, ref, read.quality)
    rr.chain_time = chain_t
    rr.n_chains   = len(chains)

    if not chains:
        rr.overhead_time  = len(read.sequence) * T_OVERHEAD
        rr.total_map_time = rr.seed_time + rr.chain_time + rr.overhead_time
        rr.total_time     = _pipeline_time(rr.basecall_time, rr.total_map_time,
                                           use_pipelining, len(read.chunks))
        return rr

    rr.has_chain  = True
    best_chain    = chains[0]

    # ── WCSA or direct DP ────────────────────────────────────────────────
    if use_wcsa:
        w = wcsa_block(best_chain, tau_acc=tau_acc, tau_rej=tau_rej,
                       use_pim=use_pim)
        rr.wcsa_ce_time    = w["ce_time"]
        rr.wcsa_rcl_time   = w["rcl_time"]
        rr.wcsa_wcs_time   = w["wcs_time"]
        rr.wcsa_adg_time   = w["adg_time"]
        rr.wcsa_total_time = w["total_wcsa_time"]
        rr.cc_scores.append(w["cc"])
        rr.vr_scores.append(w["vr"])

        decision = w["decision"]

        if decision == "ACCEPT":
            rr.n_wcsa_accept  += 1
            rr.is_mapped       = True
            exp_s, exp_e       = _expand_coords(best_chain, read.length)
            rr.pred_ref_start  = exp_s
            rr.pred_ref_end    = exp_e
            rr.pred_strand     = w["strand"]
            rr.mapq            = w["mapq"]
            rr.source          = "WCSA_ACCEPT"

        elif decision == "REJECT":
            rr.n_wcsa_reject += 1
            rr.is_mapped      = False
            rr.source         = "WCSA_REJECT"

        else:   # UNCERTAIN → fall through to DP
            rr.n_wcsa_uncertain += 1
            rr.n_dp_calls       += 1
            dp = align_chain(best_chain)
            rr.dp_time         = dp["dp_time"]
            rr.is_mapped       = dp["is_mapped"]
            exp_s, exp_e       = _expand_coords(best_chain, read.length)
            rr.pred_ref_start  = exp_s
            rr.pred_ref_end    = exp_e
            rr.pred_strand     = dp["strand"]
            rr.mapq            = dp["mapq"]
            rr.source          = "DP"

    else:
        rr.n_dp_calls += 1
        dp = align_chain(best_chain)
        rr.dp_time         = dp["dp_time"]
        rr.is_mapped       = dp["is_mapped"]
        exp_s, exp_e       = _expand_coords(best_chain, read.length)
        rr.pred_ref_start  = exp_s
        rr.pred_ref_end    = exp_e
        rr.pred_strand     = dp["strand"]
        rr.mapq            = dp["mapq"]
        rr.source          = "DP"

    # ── Correctness check ────────────────────────────────────────────────
    if rr.is_mapped:
        rr.is_correct = is_correct_mapping(
            rr.pred_ref_start, rr.pred_ref_end,
            read.true_ref_start, read.true_ref_end,
            CORRECT_MAP_OVERLAP,
        )

    rr.overhead_time  = len(read.sequence) * T_OVERHEAD
    rr.total_map_time = (rr.seed_time + rr.chain_time +
                         rr.wcsa_total_time + rr.dp_time +
                         rr.overhead_time)
    rr.total_time     = _pipeline_time(rr.basecall_time, rr.total_map_time,
                                       use_pipelining, len(read.chunks))
    return rr


def _expand_coords(chain, read_len: int) -> tuple[int, int]:
    """
    Expand chain ref coordinates to cover the full read.

    A chain spans only the anchored region [q_start, q_end] of the read.
    Bases before q_start and after q_end are overhangs not represented in
    chain.ref_start / chain.ref_end.  Projecting the overhangs onto the
    reference gives the predicted full-read locus, making the correctness
    check comparable to the true [ref_start, ref_end] interval.
    """
    left_oh  = chain.q_start
    right_oh = max(0, read_len - chain.q_end)
    return (max(0, chain.ref_start - left_oh),
            chain.ref_end + right_oh)


def _pipeline_time(bc_t: float, map_t: float,
                   pipelining: bool, n_chunks: int) -> float:
    """
    Model wall-clock time with / without chunk pipelining.

    With pipelining, basecalling and mapping overlap chunk-by-chunk.
    Throughput is limited by the slower of the two stages.
    The slower stage dominates from chunk 2 onward; the first chunk
    pays the startup latency of the slower stage.
    """
    if not pipelining or n_chunks == 0:
        return bc_t + map_t
    bc_per  = bc_t  / n_chunks
    mp_per  = map_t / n_chunks
    bottleneck = max(bc_per, mp_per)
    # One bottleneck chunk of startup + n_chunks * bottleneck throughput
    return bottleneck + n_chunks * bottleneck


# ─── Full system run ──────────────────────────────────────────────────────────

def run_system(name:           str,
               reads:          list[SimRead],
               ref:            str,
               index:          dict,
               use_pipelining: bool,
               use_er:         bool,
               use_pim:        bool,
               use_wcsa:       bool,
               tau_acc:        float = 0.85,
               tau_rej:        float = 0.40) -> dict:

    print(f"  [{name}] processing {len(reads)} reads...", flush=True)
    t0 = time.perf_counter()

    read_results = [
        process_read(r, ref, index,
                     use_pipelining, use_er, use_pim, use_wcsa,
                     tau_acc, tau_rej)
        for r in reads
    ]

    elapsed = time.perf_counter() - t0

    # ── Aggregate ────────────────────────────────────────────────────────
    n = len(read_results)

    total_time  = sum(r.total_time    for r in read_results)
    total_bc    = sum(r.basecall_time for r in read_results)
    total_map   = sum(r.total_map_time for r in read_results)
    total_dp    = sum(r.dp_time       for r in read_results)
    total_seed  = sum(r.seed_time     for r in read_results)
    total_chain = sum(r.chain_time    for r in read_results)
    total_wcsa  = sum(r.wcsa_total_time for r in read_results)
    total_wce   = sum(r.wcsa_ce_time  for r in read_results)
    total_wrcl  = sum(r.wcsa_rcl_time for r in read_results)
    total_wwcs  = sum(r.wcsa_wcs_time for r in read_results)
    total_wadg  = sum(r.wcsa_adg_time for r in read_results)
    total_oh    = sum(r.overhead_time for r in read_results)

    n_mapped     = sum(1 for r in read_results if r.is_mapped)
    n_correct    = sum(1 for r in read_results if r.is_correct)
    n_dp_calls   = sum(r.n_dp_calls        for r in read_results)
    n_accept     = sum(r.n_wcsa_accept     for r in read_results)
    n_reject     = sum(r.n_wcsa_reject     for r in read_results)
    n_uncertain  = sum(r.n_wcsa_uncertain  for r in read_results)
    n_er_chunks  = sum(r.n_er_rejected     for r in read_results)
    n_er_reads   = sum(1 for r in read_results if r.is_er_rejected)
    total_chunks = sum(len(r2.chunks) for r2 in reads)

    # Mappable reads: those not discarded by ER and that produced ≥1 chain
    # (i.e., the aligner was given a genuine attempt).  Sensitivity is
    # TP / (TP + FN) over this set.
    n_attempted   = sum(1 for r in read_results if r.has_chain)
    n_mappable    = n_attempted   # reads where alignment was attempted

    sensitivity_mappable = (100.0 * n_correct / n_mappable
                            if n_mappable > 0 else 0.0)
    sensitivity_all      = 100.0 * n_correct / n   # legacy; kept for plots

    mapping_rate = 100.0 * n_mapped / n

    # WCSA bypass
    n_wcsa_total = n_accept + n_reject + n_uncertain
    bypass_pct   = 100.0 * (n_accept + n_reject) / max(n_wcsa_total, 1)
    accept_pct   = 100.0 * n_accept               / max(n_wcsa_total, 1)
    reject_pct   = 100.0 * n_reject               / max(n_wcsa_total, 1)

    cc_all = [cc for r in read_results for cc in r.cc_scores]
    vr_all = [vr for r in read_results for vr in r.vr_scores]

    print(f"  [{name}] done in {elapsed:.1f}s  "
          f"sensitivity(mappable)={sensitivity_mappable:.1f}%  "
          f"dp_calls={n_dp_calls}  "
          f"bypass={bypass_pct:.1f}%")

    return {
        "system"                   : name,
        "n_reads"                  : n,
        "n_mappable"               : n_mappable,
        "n_er_reads"               : n_er_reads,
        "total_time_s"             : total_time,
        "total_basecall_s"         : total_bc,
        "total_map_s"              : total_map,
        "total_dp_s"               : total_dp,
        "total_seed_s"             : total_seed,
        "total_chain_s"            : total_chain,
        "total_wcsa_s"             : total_wcsa,
        "wcsa_ce_s"                : total_wce,
        "wcsa_rcl_s"               : total_wrcl,
        "wcsa_wcs_s"               : total_wwcs,
        "wcsa_adg_s"               : total_wadg,
        "total_overhead_s"         : total_oh,
        "n_mapped"                 : n_mapped,
        "n_correct"                : n_correct,
        "sensitivity_pct"          : sensitivity_mappable,   # primary metric
        "sensitivity_all_pct"      : sensitivity_all,        # over all reads
        "mapping_rate_pct"         : mapping_rate,
        "n_dp_calls"               : n_dp_calls,
        "n_wcsa_accept"            : n_accept,
        "n_wcsa_reject"            : n_reject,
        "n_wcsa_uncertain"         : n_uncertain,
        "n_er_rejected_chunks"     : n_er_chunks,
        "n_er_rejected_reads"      : n_er_reads,
        "total_chunks"             : total_chunks,
        "wcsa_bypass_pct"          : bypass_pct,
        "wcsa_accept_pct"          : accept_pct,
        "wcsa_reject_pct"          : reject_pct,
        "er_rate_pct"              : 100.0 * n_er_chunks / max(total_chunks, 1),
        "power_weight"             : POWER_WEIGHT[name],
        "cc_distribution"          : cc_all,
        "vr_distribution"          : vr_all,
        "stage_seed"               : total_seed,
        "stage_chain"              : total_chain,
        "stage_wcsa_ce"            : total_wce,
        "stage_wcsa_rcl"           : total_wrcl,
        "stage_wcsa_wcs"           : total_wwcs,
        "stage_wcsa_adg"           : total_wadg,
        "stage_alignment"          : total_dp,
        "stage_overhead"           : total_oh,
        "read_results"             : read_results,
    }


def run_all_systems(test_reads: list[SimRead],
                    ref:        str,
                    index:      dict,
                    tau_acc:    float,
                    tau_rej:    float) -> dict:
    """Run all 5 systems; attach end-to-end and mapping-only speedups."""
    print("\n[Pipeline] Running all 5 systems...\n")

    results = {}

    results["CPU"] = run_system(
        "CPU", test_reads, ref, index,
        use_pipelining=False, use_er=False, use_pim=False, use_wcsa=False)

    results["CPU-CP"] = run_system(
        "CPU-CP", test_reads, ref, index,
        use_pipelining=True, use_er=False, use_pim=False, use_wcsa=False)

    results["CPU-GP"] = run_system(
        "CPU-GP", test_reads, ref, index,
        use_pipelining=True, use_er=True, use_pim=False, use_wcsa=False)

    results["GenPIP"] = run_system(
        "GenPIP", test_reads, ref, index,
        use_pipelining=True, use_er=True, use_pim=True, use_wcsa=False)

    results["GenPIP-WCSA"] = run_system(
        "GenPIP-WCSA", test_reads, ref, index,
        use_pipelining=True, use_er=True, use_pim=True, use_wcsa=True,
        tau_acc=tau_acc, tau_rej=tau_rej)

    # ── End-to-end speedup (relative to CPU) ─────────────────────────────
    cpu_t     = results["CPU"]["total_time_s"]
    cpu_map_t = results["CPU"]["total_map_s"]

    for name, r in results.items():
        r["speedup"]          = cpu_t     / r["total_time_s"]
        r["speedup_map_only"] = cpu_map_t / r["total_map_s"] \
                                if r["total_map_s"] > 0 else 1.0
        r["energy_score"]     = r["power_weight"] * r["total_time_s"] / cpu_t

    return results
