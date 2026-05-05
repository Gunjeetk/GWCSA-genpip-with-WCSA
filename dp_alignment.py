# =============================================================
#  dp_alignment.py  — banded Smith-Waterman with affine gap costs
#
#  Three-matrix formulation (H, E, F) as in Gotoh (1982):
#    H[i,j] = best score ending at (query[i], ref[j])
#    E[i,j] = best score with a gap open in the reference (insertion in query)
#    F[i,j] = best score with a gap open in the query   (deletion)
#
#  Only cells within ±bw of the main diagonal are computed.
#  Time: O(n × bw)  per alignment.
# =============================================================

from __future__ import annotations
import numpy as np
from config import (
    SW_BANDWIDTH, SW_MATCH, SW_MISMATCH, SW_GAP_OPEN, SW_GAP_EXT,
    T_ALIGN_DP,
)
from sim_chaining import Chain
from sim_reference import reverse_complement, BASE_IDX


# Pre-built translation table for fast base comparison
_ORD = np.zeros(256, dtype=np.int8)
for _b, _v in BASE_IDX.items():
    _ORD[ord(_b)] = _v


def banded_sw(query: str, ref: str,
              bw:       int = SW_BANDWIDTH,
              match:    int = SW_MATCH,
              mismatch: int = SW_MISMATCH,
              gap_open: int = SW_GAP_OPEN,
              gap_ext:  int = SW_GAP_EXT) -> dict:
    """
    Affine-gap banded Smith-Waterman (Gotoh, 1982).

    H[i, bw+dj] holds the H-score at query position i, ref position i+dj.
    E tracks gaps in the reference (vertical moves).
    F is propagated left-to-right within each row (horizontal moves).

    Returns score, identity proxy, and alignment end-points.
    """
    n, m = len(query), len(ref)
    if n == 0 or m == 0:
        return {"score": 0, "identity": 0.0, "end_i": 0, "end_j": 0}

    NINF = np.int32(-10 ** 6)
    W    = 2 * bw + 1

    # Previous and current rows for H and E (space-efficient)
    H_prev = np.full(W, NINF, dtype=np.int32)
    E_prev = np.full(W, NINF, dtype=np.int32)
    H_prev[bw] = np.int32(0)          # origin: (0, 0) on the diagonal

    # Integer arrays for fast per-row base comparison
    q_arr = np.frombuffer(query.encode('ascii'), dtype=np.uint8)
    r_arr = np.frombuffer(ref.encode('ascii'),   dtype=np.uint8)

    best_score = 0
    best_i = best_j = 0

    for i in range(1, n + 1):
        H_cur = np.full(W, NINF, dtype=np.int32)
        E_cur = np.full(W, NINF, dtype=np.int32)

        dj_arr  = np.arange(-bw, bw + 1, dtype=np.int32)
        j_arr   = np.int32(i) + dj_arr          # ref positions (1-indexed)
        valid   = (j_arr >= 1) & (j_arr <= m)

        # ── E: gap in reference (insertion in query) ──────────────────────
        # E[i, bw+dj] = max( H[i-1, bw+(dj+1)] + gap_open,
        #                    E[i-1, bw+(dj+1)] + gap_ext )
        # The vertical predecessor at (i-1, j) has index bw + (j - (i-1)) = bw+dj+1
        up_idx = np.clip(np.arange(1, W + 1), 0, W - 1)
        E_from_H = np.where(H_prev[up_idx] > NINF // 2,
                             H_prev[up_idx] + gap_open, NINF)
        E_from_E = np.where(E_prev[up_idx] > NINF // 2,
                             E_prev[up_idx] + gap_ext,  NINF)
        E_cur[:] = np.where(valid, np.maximum(E_from_H, E_from_E), NINF)

        # ── Diagonal: match / mismatch ────────────────────────────────────
        # Predecessor at (i-1, j-1) has same dj → index bw+dj in H_prev
        r_idx    = np.clip(j_arr - 1, 0, m - 1)
        is_match = np.where(valid, r_arr[r_idx] == q_arr[i - 1], False)
        ms_score = np.where(is_match, match, mismatch)
        from_diag = np.where(valid & (H_prev > NINF // 2),
                             H_prev + ms_score, NINF)

        # ── Combine diagonal and E (before F) ─────────────────────────────
        H_cur[:] = np.where(valid,
                            np.maximum(np.int32(0),
                                       np.maximum(from_diag, E_cur)),
                            NINF)

        # ── F: gap in query (deletion), left-to-right propagation ─────────
        # F[i, k] = max( H[i, k-1] + gap_open, F[i, k-1] + gap_ext )
        # Must be computed sequentially; inner loop over bw is small (≤65).
        f_score = NINF
        for k in range(W):
            if not valid[k]:
                f_score = NINF
                continue
            if k > 0 and valid[k - 1]:
                f_from_h = (int(H_cur[k - 1]) + gap_open
                            if H_cur[k - 1] > NINF // 2 else int(NINF))
                f_from_f = (int(f_score) + gap_ext
                            if f_score > NINF // 2 else int(NINF))
                f_score = max(int(NINF), max(f_from_h, f_from_f))
            else:
                f_score = int(NINF)
            if f_score > int(H_cur[k]):
                H_cur[k] = np.int32(f_score)

        # ── Track global best ─────────────────────────────────────────────
        if valid.any():
            local_max = int(H_cur[valid].max())
            if local_max > best_score:
                best_score = local_max
                local_argmax = int(np.argmax(H_cur * valid.astype(np.int32)))
                best_i = i
                best_j = int(j_arr[local_argmax])

        H_prev = H_cur
        E_prev = E_cur

    max_possible = best_i * match
    identity     = best_score / max(max_possible, 1)
    return {
        "score"    : best_score,
        "identity" : float(np.clip(identity, 0.0, 1.0)),
        "end_i"    : best_i,
        "end_j"    : best_j,
    }


def align_chain(chain: Chain, use_pim: bool = False) -> dict:
    """
    Run affine-gap banded SW on a chain's query vs reference subsequence.
    Handles reverse-complement for reverse-strand chains.
    """
    query = chain.query_seq
    ref   = chain.ref_seq

    if chain.strand == -1:
        query = reverse_complement(query)

    sw = banded_sw(query, ref)

    dp_time = len(chain.query_seq) * T_ALIGN_DP

    return {
        "score"     : sw["score"],
        "identity"  : sw["identity"],
        "ref_start" : chain.ref_start,
        "ref_end"   : chain.ref_end,
        "strand"    : chain.strand,
        "mapq"      : _sw_score_to_mapq(sw["score"], len(chain.query_seq)),
        "dp_time"   : dp_time,
        "is_mapped" : sw["score"] > 10,
    }


def _sw_score_to_mapq(score: int, query_len: int) -> int:
    """
    Approximate MAPQ from SW score.
    fraction = score / (query_len × match_score_per_base)
    → MAPQ ≈ -10 log10(1 − fraction), capped at 60.
    """
    if query_len == 0:
        return 0
    fraction = score / max(query_len * SW_MATCH, 1)
    if fraction >= 0.99:
        return 60
    if fraction < 0.05:
        return 0
    return int(np.clip(-10 * np.log10(1.0 - fraction + 1e-6), 0, 60))
