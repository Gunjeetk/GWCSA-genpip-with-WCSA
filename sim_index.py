# =============================================================
#  sim_index.py  — build and query minimizer index on reference
#  Replaces the fake DRAM-table approach.
#  Index is built ONCE from the real reference string and reused
#  identically by all 5 pipeline systems.
# =============================================================

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from config import KMER_K, MINIMIZER_W, MAX_SEED_HITS
from sim_reference import BASE_IDX, reverse_complement


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Anchor:
    """One seed match: query position ↔ reference position."""
    q_pos   : int
    r_pos   : int
    strand  : int    # +1 or -1
    length  : int = KMER_K


# ─── K-mer hashing ────────────────────────────────────────────────────────────

def kmer_hash(kmer: str, k: int = KMER_K) -> int:
    """Map a DNA k-mer to an integer. Forward strand only."""
    h = 0
    for b in kmer:
        h = h * 4 + BASE_IDX.get(b, 0)
    return h


def canonical_hash(kmer: str, k: int = KMER_K) -> tuple[int, int]:
    """
    Return (hash, strand) where hash is the lexicographically smaller
    of forward and reverse-complement hashes (canonical form).
    """
    fwd = kmer_hash(kmer, k)
    rev = kmer_hash(reverse_complement(kmer), k)
    if fwd <= rev:
        return fwd, +1
    return rev, -1


# ─── Minimizer selection ──────────────────────────────────────────────────────

def extract_minimizers(seq: str, k: int = KMER_K,
                       w: int = MINIMIZER_W) -> list[tuple[int, int, int]]:
    """
    Extract minimizers from seq.
    A minimizer is the k-mer with the smallest canonical hash in any
    window of w consecutive k-mers.

    Returns list of (hash, seq_pos, strand).
    Duplicates (same hash at adjacent windows) are deduplicated.
    """
    n = len(seq)
    if n < k + w - 1:
        return []

    minimizers = []
    prev_min_pos = -1

    for i in range(n - k - w + 2):
        window = []
        for j in range(w):
            pos  = i + j
            kmer = seq[pos: pos + k]
            if len(kmer) < k or 'N' in kmer:
                continue
            h, s = canonical_hash(kmer, k)
            window.append((h, pos, s))
        if not window:
            continue
        # Pick minimum hash in window
        min_h, min_pos, min_s = min(window, key=lambda x: x[0])
        if min_pos != prev_min_pos:
            minimizers.append((min_h, min_pos, min_s))
            prev_min_pos = min_pos

    return minimizers


# ─── Index construction ───────────────────────────────────────────────────────

def build_index(ref: str, k: int = KMER_K,
                w: int = MINIMIZER_W) -> dict[int, list[tuple[int, int]]]:
    """
    Build a minimizer index on the reference.
    Returns dict: hash → list of (ref_pos, strand).
    Filters out k-mers with > MAX_SEED_HITS hits (repetitive).
    """
    print(f"[Index] Building minimizer index (k={k}, w={w}) "
          f"on {len(ref):,} bp reference...", end=" ", flush=True)

    raw: dict[int, list] = {}
    for h, pos, s in extract_minimizers(ref, k, w):
        if h not in raw:
            raw[h] = []
        raw[h].append((pos, s))

    # Filter high-frequency k-mers
    index = {h: hits for h, hits in raw.items()
             if len(hits) <= MAX_SEED_HITS}

    total_kmers  = len(raw)
    kept         = len(index)
    filtered     = total_kmers - kept
    total_hits   = sum(len(v) for v in index.values())
    print(f"done.  {kept:,} distinct k-mers "
          f"({filtered} repetitive filtered), "
          f"{total_hits:,} total positions.")
    return index


# ─── Seed lookup ──────────────────────────────────────────────────────────────

def find_anchors(read_seq: str, index: dict,
                 k: int = KMER_K, w: int = MINIMIZER_W,
                 pim_speedup: float = 1.0) -> tuple[list[Anchor], float]:
    """
    Extract minimizers from the query read, look up each in the reference
    index, and return all seed anchors.

    pim_speedup: divides the modelled seed-lookup time (PIM benefit).

    Returns (anchors, modelled_lookup_time_seconds).
    """
    from config import T_SEED_LOOKUP, T_MINIMIZER
    query_minimizers = extract_minimizers(read_seq, k, w)

    anchors: list[Anchor] = []
    for h, q_pos, q_strand in query_minimizers:
        if h in index:
            for r_pos, r_strand in index[h]:
                # Strand of the anchor: XOR of query and ref strand flags
                strand = +1 if q_strand == r_strand else -1
                anchors.append(Anchor(
                    q_pos  = q_pos,
                    r_pos  = r_pos,
                    strand = strand,
                    length = k,
                ))

    n = len(read_seq)
    minimizer_time  = n * T_MINIMIZER
    seed_lookup_time = n * T_SEED_LOOKUP / pim_speedup

    return anchors, minimizer_time + seed_lookup_time
