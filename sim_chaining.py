# =============================================================
#  sim_chaining.py  — anchor chaining DP (Minimap2 style)
#
#  Forward-strand chains : sorted by r_pos; require r_gap > 0 & q_gap > 0.
#  Reverse-strand chains : sorted by q_pos; require q_gap > 0 &
#                          r_gap_rev > 0  (r_pos DECREASES as q_pos grows).
#
#  Chains carry the actual reference subsequence so WCSA can
#  encode both query and reference directly — no circular lookup.
# =============================================================

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from config import (
    MAX_CHAIN_GAP, MIN_CHAIN_ANCHORS, MAX_CHAINS_KEPT,
    T_ANCHOR, T_CHAIN_DP, KMER_K,
)
from sim_index import Anchor


# ─── Data class ───────────────────────────────────────────────────────────────

@dataclass
class Chain:
    chain_id   : int
    anchors    : list[Anchor]
    chain_score: float
    ref_start  : int
    ref_end    : int
    q_start    : int
    q_end      : int
    strand     : int           # +1 or −1
    ref_seq    : str           # forward-strand reference subsequence
    query_seq  : str           # query subsequence spanning this chain
    query_qual : np.ndarray    # per-base Phred quality for query_seq


# ─── Chaining DP ──────────────────────────────────────────────────────────────

def chain_anchors(anchors:    list[Anchor],
                  read_seq:   str,
                  ref:        str,
                  query_qual: np.ndarray,
                  max_gap:    int = MAX_CHAIN_GAP,
                  min_anch:   int = MIN_CHAIN_ANCHORS,
                  max_keep:   int = MAX_CHAINS_KEPT,
                  ) -> tuple[list[Chain], float]:
    """
    Minimap2-style co-linear anchor chaining DP.

    Forward strand (strand=+1)
    --------------------------
    Anchors sorted by r_pos ascending.  Co-linearity:
        r_gap > 0  and  q_gap > 0  and  r_gap ≤ max_gap.

    Reverse strand (strand=−1)
    --------------------------
    For an RC-strand alignment, as query position increases, the
    reference position DECREASES.  Anchors are therefore sorted by
    q_pos ascending; co-linearity requires:
        q_gap > 0  and  r_gap_rev = sa[j].r_pos − sa[i].r_pos > 0
        and  q_gap ≤ max_gap.

    Chain ref coordinates:
      Forward: ref_start = path[0].r_pos, ref_end = path[-1].r_pos + k
      Reverse: ref_start = path[-1].r_pos, ref_end = path[0].r_pos + k
               (path[-1] has the largest q_pos → smallest r_pos)
    """
    if len(anchors) < min_anch:
        n = len(read_seq)
        return [], n * (T_ANCHOR + T_CHAIN_DP)

    fwd_anchors = [a for a in anchors if a.strand == +1]
    rev_anchors = [a for a in anchors if a.strand == -1]

    chains: list[Chain] = []
    chain_id = 0

    for strand_val, strand_anchors in [(+1, fwd_anchors), (-1, rev_anchors)]:
        if len(strand_anchors) < min_anch:
            continue

        # ── Sort ─────────────────────────────────────────────────────────
        if strand_val == +1:
            sa = sorted(strand_anchors, key=lambda a: (a.r_pos, a.q_pos))
        else:
            # Sorted by query position; reference goes in reverse order.
            sa = sorted(strand_anchors, key=lambda a: (a.q_pos, -a.r_pos))

        n  = len(sa)
        scores = np.array([float(a.length) for a in sa], dtype=np.float64)
        prev   = [-1] * n

        # ── Chaining DP ──────────────────────────────────────────────────
        for i in range(1, n):
            for j in range(i - 1, max(i - 80, -1), -1):
                if strand_val == +1:
                    r_gap = sa[i].r_pos - sa[j].r_pos
                    q_gap = sa[i].q_pos - sa[j].q_pos
                    if r_gap <= 0 or q_gap <= 0 or r_gap > max_gap:
                        continue
                    eff_gap = r_gap
                else:
                    q_gap   = sa[i].q_pos - sa[j].q_pos
                    r_gap   = sa[j].r_pos - sa[i].r_pos   # j has LARGER r_pos
                    if q_gap <= 0 or r_gap <= 0 or q_gap > max_gap:
                        continue
                    eff_gap = q_gap

                gap_diff = abs(r_gap - q_gap)
                gap_cost = 0.01 * gap_diff + (0.5 if gap_diff > 0 else 0.0)
                cand = scores[j] + sa[i].length - gap_cost
                if cand > scores[i]:
                    scores[i] = cand
                    prev[i]   = j

        # ── Traceback ─────────────────────────────────────────────────────
        used = [False] * n
        for end_idx in np.argsort(-scores):
            end_idx = int(end_idx)
            if used[end_idx]:
                continue
            if scores[end_idx] < min_anch * KMER_K * 0.5:
                continue

            path = []
            cur  = end_idx
            while cur != -1:
                path.append(cur)
                cur = prev[cur]
            path.reverse()

            if len(path) < min_anch:
                continue

            for idx in path:
                used[idx] = True

            path_anchors = [sa[i] for i in path]

            # ── Chain coordinate calculation ──────────────────────────────
            q_start = path_anchors[0].q_pos
            q_end   = path_anchors[-1].q_pos + path_anchors[-1].length

            if strand_val == +1:
                r_start = path_anchors[0].r_pos
                r_end   = path_anchors[-1].r_pos + path_anchors[-1].length
            else:
                # path[-1] has largest q_pos → smallest r_pos (leftmost on ref)
                r_start = path_anchors[-1].r_pos
                r_end   = path_anchors[0].r_pos + path_anchors[0].length

            r_start = max(0, r_start)
            r_end   = min(r_end, len(ref))
            q_end   = min(q_end, len(read_seq))

            ref_subseq   = ref[r_start:r_end]
            query_subseq = read_seq[q_start:q_end]
            qual_subseq  = query_qual[q_start: min(q_end, len(query_qual))]
            if len(qual_subseq) < len(query_subseq):
                pad = np.full(len(query_subseq) - len(qual_subseq),
                              10.0, dtype=np.float32)
                qual_subseq = np.concatenate([qual_subseq, pad])

            chains.append(Chain(
                chain_id    = chain_id,
                anchors     = path_anchors,
                chain_score = float(scores[end_idx]),
                ref_start   = r_start,
                ref_end     = r_end,
                q_start     = q_start,
                q_end       = q_end,
                strand      = strand_val,
                ref_seq     = ref_subseq,
                query_seq   = query_subseq,
                query_qual  = qual_subseq,
            ))
            chain_id += 1

            if len(chains) >= max_keep:
                break
        if len(chains) >= max_keep:
            break

    chains.sort(key=lambda c: -c.chain_score)
    chains = chains[:max_keep]

    chain_time = len(read_seq) * (T_ANCHOR + T_CHAIN_DP)
    return chains, chain_time
