# =============================================================
#  wcsa_block.py  — Weighted Column Sum Array pre-alignment filter
#
#  Pipeline: CE → RCL → WCS → ADG
#
#  CE  (Column Encoder)         : encode W-base query windows into
#                                 Int8 base-4 codes (shift-add ops).
#  RCL (Reference Code Lookup)  : fetch matching reference codes at
#                                 anchor-guided diagonal positions;
#                                 benefits from PIM bandwidth.
#  WCS (Weighted Confidence Sc.): compare codes; weight by Phred Q.
#  ADG (Alignment Decision Gate): threshold on CC / VR.
#
#  The live implementation matches the architecture description.
# =============================================================

from __future__ import annotations
import numpy as np
from sim_chaining import Chain
from sim_reference import reverse_complement, BASE_IDX, COMP
from config import (
    WCSA_WINDOW_W, WCSA_WEIGHTS,
    T_WCSA_CE, T_WCSA_RCL, T_WCSA_WCS, T_WCSA_ADG,
    PIM_RCL_SPEEDUP, KMER_K,
)

_WEIGHTS = np.array(WCSA_WEIGHTS, dtype=np.int32)
W        = WCSA_WINDOW_W


# ─── CE: Column Encoder ───────────────────────────────────────────────────────

def encode_sequence(seq: str) -> np.ndarray:
    """
    Encode a DNA string into Int8 window codes via base-4 positional encoding.

        code(i) = Σ_{j=0}^{W-1}  base_value[i+j] × 4^j

    base values: A=0, C=1, G=2, T=3.  Codes are in [0, 4^W − 1].
    For W=2: range [0, 15] — fits in a nibble (Int8 safe).

    Returns shape (len(seq) − W + 1,) array of Int32 codes.
    """
    if len(seq) < W:
        return np.array([], dtype=np.int32)

    base_ints = np.array([BASE_IDX.get(b, 0) for b in seq], dtype=np.int32)
    n_windows = len(seq) - W + 1
    shape   = (n_windows, W)
    strides = (base_ints.strides[0], base_ints.strides[0])
    windows = np.lib.stride_tricks.as_strided(base_ints,
                                               shape=shape,
                                               strides=strides)
    return (windows @ _WEIGHTS).astype(np.int32)


# ─── RCL: classify code pairs ────────────────────────────────────────────────

def classify_codes(q_codes: np.ndarray,
                   r_codes: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare query codes against reference codes position by position.
    Returns (match_flags, snp_flags, indel_flags), each shape (n,) int8.

    MATCH  : q_code == r_code  (all W bases identical)
    SNP    : exactly one base-4 digit differs
             diff == m × 4^k  for some k in [0,W-1], m in [1,2,3]
    INDEL  : consecutive-window code-difference discontinuity;
             proxy for frame-shifted insertion/deletion signal.

    Flags are mutually exclusive: MATCH > SNP > INDEL.
    """
    n = min(len(q_codes), len(r_codes))
    q = q_codes[:n].astype(np.int32)
    r = r_codes[:n].astype(np.int32)

    match_flags = (q == r).astype(np.int8)

    diff      = np.abs(q - r)
    snp_flags = np.zeros(n, dtype=np.int8)
    for k in range(W):
        unit = int(4 ** k)
        for m in [1, 2, 3]:
            snp_flags |= (diff == m * unit).astype(np.int8)
    snp_flags &= ~match_flags

    indel_flags = np.zeros(n, dtype=np.int8)
    if n > 2:
        qdiff  = np.diff(q)
        rdiff  = np.diff(r)
        breaks = (qdiff != rdiff).astype(np.int8)
        indel_flags[:-1] |= breaks
        indel_flags[1:]  |= breaks
    indel_flags &= ~match_flags & ~snp_flags

    return match_flags, snp_flags, indel_flags


# ─── WCS: quality-weighted confidence scorer ──────────────────────────────────

def weighted_confidence(match_flags:  np.ndarray,
                        snp_flags:    np.ndarray,
                        indel_flags:  np.ndarray,
                        window_quals: np.ndarray) -> tuple[float, float]:
    """
    Compute consensus confidence (CC) and variant ratio (VR).

        CC = Σ(w_i × match_i) / Σ(w_i)
        VR = (n_snp + n_indel) / n_windows

    Weights:  w_i = clip(1 − 10^(−Q_i/10), 0.01, 1.0)   (Phred probability)
    window_quals : per-window mean Phred quality, shape (n,).
    """
    n = len(match_flags)
    if n == 0:
        return 0.0, 1.0
    weights = np.clip(1.0 - 10.0 ** (-window_quals / 10.0), 0.01, 1.0)
    total_w = float(np.sum(weights))
    CC      = float(np.sum(match_flags * weights)) / max(total_w, 1e-9)
    VR      = (int(np.sum(snp_flags)) + int(np.sum(indel_flags))) / n
    return CC, VR


# ─── ADG: MAPQ from CC ────────────────────────────────────────────────────────

def cc_to_mapq(cc: float) -> int:
    """Approximate MAPQ from consensus confidence."""
    return int(min(60, -10.0 * np.log10(max(1.0 - cc, 1e-6))))


# ─── Full WCSA block ──────────────────────────────────────────────────────────

def wcsa_block(chain:    Chain,
               tau_acc:  float = 0.85,
               tau_rej:  float = 0.10,
               use_pim:  bool  = False) -> dict:
    """
    WCSA block: CE → RCL → WCS → ADG.

    Inter-anchor window comparison
    --------------------------------
    For each consecutive anchor pair (a_prev, a_next), WCSA compares only
    the *inter-anchor* bases — the region strictly between the two anchor
    k-mers, excluding the anchors themselves.

    Anchor k-mers are exact matches by hash construction, so including them
    forces CC → 1.0 for every chain regardless of correctness.  Comparing
    only the inter-anchor windows allows CC to reflect the true per-base
    match rate.  Because inter-anchor gaps are enriched for error-containing
    positions (anchor formation requires error-free k-mers), the observed CC
    is lower than the genome-wide error rate (typical correct chain CC ≈ 0.35–
    0.50 at 8 % ONT error rate); false chains at unrelated loci yield CC ≈
    0.05–0.20 (random-DNA base rate).  Calibration selects tau_acc from data.

    Forward strand
        query window : query_seq[q_prev + k : q_next]
        ref window   : ref_seq  [r_prev + k : r_next]

    Reverse strand  (r_pos DECREASES as q_pos grows)
        query window : query_seq[q_prev + k : q_next]         (original)
        ref window   : RC(ref_seq[r_next + k : r_prev])       (flip to query orientation)

    chain.ref_seq is the forward-strand reference subsequence
    [chain.ref_start, chain.ref_end), populated by sim_chaining.
    For reverse chains chain.ref_start equals the smallest r_pos anchor.
    """
    assert chain.ref_seq, \
        "chain.ref_seq is empty — must be populated from reference before WCSA"

    if len(chain.anchors) < 2:
        return _uncertain_result(chain, 0.0, 0.0)

    k              = KMER_K
    sorted_anchors = sorted(chain.anchors, key=lambda a: a.q_pos)

    all_match : list[np.ndarray] = []
    all_snp   : list[np.ndarray] = []
    all_indel : list[np.ndarray] = []
    all_qual  : list[np.ndarray] = []

    for idx in range(len(sorted_anchors) - 1):
        a_prev = sorted_anchors[idx]
        a_next = sorted_anchors[idx + 1]

        # Local offsets within chain.query_seq / chain.ref_seq
        q_local_p = a_prev.q_pos - chain.q_start
        q_local_n = a_next.q_pos - chain.q_start
        r_local_p = a_prev.r_pos - chain.ref_start
        r_local_n = a_next.r_pos - chain.ref_start

        q_ia_start = q_local_p + k
        q_ia_end   = q_local_n

        if q_ia_end - q_ia_start < W:
            continue

        qw = chain.query_seq[q_ia_start:q_ia_end]

        if chain.strand == +1:
            # Forward: reference advances in the same direction as query
            r_ia_start = r_local_p + k
            r_ia_end   = r_local_n
            if r_ia_end - r_ia_start < W:
                continue
            rw = chain.ref_seq[r_ia_start:r_ia_end]
        else:
            # Reverse: r_local_p > r_local_n  (larger q_pos → smaller r_pos)
            # Inter-anchor ref on forward strand: [r_local_n + k, r_local_p)
            # RC to align with query orientation.
            r_ia_start = r_local_n + k
            r_ia_end   = r_local_p
            if r_ia_end - r_ia_start < W:
                continue
            rw = reverse_complement(chain.ref_seq[r_ia_start:r_ia_end])

        q_codes_ia = encode_sequence(qw)
        r_codes_ia = encode_sequence(rw)

        if len(q_codes_ia) == 0 or len(r_codes_ia) == 0:
            continue

        n_ia = min(len(q_codes_ia), len(r_codes_ia))
        m, s, idl = classify_codes(q_codes_ia[:n_ia], r_codes_ia[:n_ia])

        # Per-window Phred quality — use original orientation (qw is not RC'd)
        q_qual_seg = np.ascontiguousarray(
            chain.query_qual[q_ia_start : min(q_ia_end, len(chain.query_qual))])
        n_qual_w = max(0, len(q_qual_seg) - W + 1)
        if n_qual_w > 0:
            n_win   = min(n_qual_w, n_ia)
            shape   = (n_win, W)
            strides = (q_qual_seg.strides[0], q_qual_seg.strides[0])
            q_wins  = np.lib.stride_tricks.as_strided(
                q_qual_seg[:n_win + W - 1], shape=shape, strides=strides)
            w_qual  = np.mean(q_wins, axis=1).astype(np.float32)
        else:
            w_qual = np.array([], dtype=np.float32)

        if len(w_qual) < n_ia:
            pad    = np.full(n_ia - len(w_qual),
                             float(w_qual[-1]) if len(w_qual) else 10.0,
                             dtype=np.float32)
            w_qual = np.concatenate([w_qual, pad])

        all_match.append(m)
        all_snp.append(s)
        all_indel.append(idl)
        all_qual.append(w_qual[:n_ia])

    if not all_match:
        return _uncertain_result(chain, 0.0, 0.0)

    match_flags = np.concatenate(all_match)
    snp_flags   = np.concatenate(all_snp)
    indel_flags = np.concatenate(all_indel)
    qual_arr    = np.concatenate(all_qual)
    n           = len(match_flags)

    t_ce  = n * W * T_WCSA_CE
    t_rcl = n * T_WCSA_RCL / (PIM_RCL_SPEEDUP if use_pim else 1.0)
    t_wcs = n * T_WCSA_WCS
    t_adg = T_WCSA_ADG * 1000

    CC, VR = weighted_confidence(match_flags, snp_flags, indel_flags, qual_arr)

    if CC >= tau_acc:
        decision = "ACCEPT"
        mapq     = cc_to_mapq(CC)
    elif CC < tau_rej:
        # Very low CC ≪ random-DNA base-rate (0.25) → definitely misaligned
        decision = "REJECT"
        mapq     = 0
    else:
        decision = "UNCERTAIN"
        mapq     = -1

    total = t_ce + t_rcl + t_wcs + t_adg

    return {
        "decision"       : decision,
        "cc"             : float(CC),
        "vr"             : float(VR),
        "mapq"           : mapq,
        "ref_start"      : chain.ref_start,
        "ref_end"        : chain.ref_end,
        "strand"         : chain.strand,
        "snp_count"      : int(np.sum(snp_flags)),
        "indel_count"    : int(np.sum(indel_flags)),
        "ce_time"        : t_ce,
        "rcl_time"       : t_rcl,
        "wcs_time"       : t_wcs,
        "adg_time"       : t_adg,
        "total_wcsa_time": total,
        "n_windows"      : n,
    }


def _uncertain_result(chain: Chain, t_ce: float, t_rcl: float) -> dict:
    total = t_ce + t_rcl + T_WCSA_ADG * 1000
    return {
        "decision": "UNCERTAIN", "cc": 0.0, "vr": 0.0,
        "mapq": -1,
        "ref_start": chain.ref_start, "ref_end": chain.ref_end,
        "strand": chain.strand,
        "snp_count": 0, "indel_count": 0,
        "ce_time": t_ce, "rcl_time": t_rcl,
        "wcs_time": 0.0, "adg_time": T_WCSA_ADG * 1000,
        "total_wcsa_time": total, "n_windows": 0,
    }
