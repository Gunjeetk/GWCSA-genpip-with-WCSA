# =============================================================
#  sim_reference.py  — synthetic reference genome + ONT reads
#
#  Reference contains diverged tandem-repeat families that
#  produce genuine multi-mapping candidates for ROC evaluation.
#  Every read has a known ground-truth position so that
#  accuracy is computed, not hardcoded.
# =============================================================

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from config import (
    REF_LENGTH, REF_GC_PROB, ONT_ERROR_RATE,
    ONT_SUB_FRAC, ONT_INS_FRAC, ONT_DEL_FRAC,
    ONT_QUAL_MEAN, ONT_QUAL_STD,
    READ_LEN_MEAN, READ_LEN_STD, N_READS_TOTAL,
    RANDOM_SEED, CHUNK_SIZE, ER_LOW_QUAL_FRAC,
    REF_REPEAT_N_FAMILIES, REF_REPEAT_COPY_LEN,
    REF_REPEAT_N_COPIES, REF_REPEAT_DIVERGENCE,
)

BASES    = ['A', 'C', 'G', 'T']
COMP     = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
BASE_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id   : int
    sequence   : str
    quality    : np.ndarray
    length     : int
    mean_q     : float
    frac_low_q : float
    is_low_qual: bool


@dataclass
class SimRead:
    read_id        : int
    sequence       : str
    quality        : np.ndarray
    length         : int
    chunks         : list[Chunk]
    true_ref_start : int
    true_ref_end   : int
    true_strand    : int          # +1 forward, -1 reverse complement
    basecall_time  : float
    mean_quality   : float
    in_repeat      : bool = False # True when origin overlaps a repeat copy


# ─── Utilities ────────────────────────────────────────────────────────────────

def reverse_complement(seq: str) -> str:
    return ''.join(COMP.get(b, 'N') for b in reversed(seq))


def base_idx(b: str) -> int:
    return BASE_IDX.get(b, 0)


# ─── Reference genome with repeat families ────────────────────────────────────

def _diverge(unit: list[str], divergence: float, rng) -> list[str]:
    """Apply random substitutions at `divergence` rate to create a copy."""
    copy = list(unit)
    for i in range(len(copy)):
        if rng.random() < divergence:
            copy[i] = rng.choice(BASES)
    return copy


def build_reference(length: int = REF_LENGTH,
                    seed:   int = RANDOM_SEED
                    ) -> tuple[str, list[tuple[int, int, int, int]]]:
    """
    Build a reference genome with diverged tandem-repeat families.

    Returns
    -------
    ref : str
        Full reference sequence.
    repeat_coords : list of (start, end, family_id, copy_id)
        Coordinates of every repeat copy inserted into the reference.
        Used downstream to identify multi-mapping reads.
    """
    rng = np.random.default_rng(seed)
    ref = list(rng.choice(BASES, size=length, p=REF_GC_PROB))

    repeat_coords: list[tuple[int, int, int, int]] = []

    # Spacing between family anchors — spread evenly across the reference,
    # reserving 10 % margins so reads at the edges still have valid templates.
    margin      = length // 10
    usable      = length - 2 * margin
    family_step = usable // max(REF_REPEAT_N_FAMILIES, 1)

    for fam in range(REF_REPEAT_N_FAMILIES):
        # Canonical sequence for this family
        canon = list(rng.choice(BASES, size=REF_REPEAT_COPY_LEN, p=REF_GC_PROB))

        # Place N_COPIES copies with 5 Kbp unique spacers between them
        copy_stride = REF_REPEAT_COPY_LEN + 5_000
        family_origin = margin + fam * family_step

        for cp in range(REF_REPEAT_N_COPIES):
            pos = family_origin + cp * copy_stride
            end = pos + REF_REPEAT_COPY_LEN
            if end > length:
                break

            # Each copy diverges independently from the canonical sequence
            copy_seq = _diverge(canon, REF_REPEAT_DIVERGENCE, rng)
            for i, base in enumerate(copy_seq):
                ref[pos + i] = base

            repeat_coords.append((pos, end, fam, cp))

    ref_str = ''.join(ref)
    n_repeat_bp = sum(e - s for s, e, _, _ in repeat_coords)
    print(f"[Reference] {length:,} bp  |  "
          f"{len(repeat_coords)} repeat copies  |  "
          f"{n_repeat_bp / length * 100:.1f}% repetitive")
    return ref_str, repeat_coords


# ─── ONT error model ──────────────────────────────────────────────────────────

def _apply_ont_errors(template: str, rng,
                      error_rate: float = ONT_ERROR_RATE) -> str:
    seq = []
    i   = 0
    while i < len(template):
        if rng.random() < error_rate:
            etype = rng.random()
            if etype < ONT_SUB_FRAC:
                seq.append(rng.choice(BASES))
                i += 1
            elif etype < ONT_SUB_FRAC + ONT_INS_FRAC:
                seq.append(rng.choice(BASES))
                # do NOT advance i — ref base still emitted next iteration
            else:
                i += 1                            # deletion: skip ref base
        else:
            seq.append(template[i])
            i += 1
    return ''.join(seq)


def _simulate_quality(length: int, rng,
                      mean_q: float = ONT_QUAL_MEAN,
                      std_q:  float = ONT_QUAL_STD) -> np.ndarray:
    raw = rng.normal(mean_q, std_q, size=length)
    return np.clip(raw, 2, 40).astype(np.float32)


# ─── Chunking ─────────────────────────────────────────────────────────────────

def _make_chunks(seq: str, qual: np.ndarray,
                 chunk_size: int = CHUNK_SIZE) -> list[Chunk]:
    chunks = []
    for start in range(0, len(seq), chunk_size):
        end     = min(start + chunk_size, len(seq))
        cseq    = seq[start:end]
        cqual   = qual[start:end]
        mean_q  = float(np.mean(cqual))
        frac_lq = float(np.mean(cqual < 10))
        chunks.append(Chunk(
            chunk_id    = len(chunks),
            sequence    = cseq,
            quality     = cqual,
            length      = len(cseq),
            mean_q      = mean_q,
            frac_low_q  = frac_lq,
            is_low_qual = frac_lq > ER_LOW_QUAL_FRAC,
        ))
    return chunks


# ─── Read generator ───────────────────────────────────────────────────────────

def simulate_ont_read(ref:            str,
                      repeat_coords:  list,
                      read_id:        int,
                      rng,
                      read_len_mean:  int = READ_LEN_MEAN,
                      read_len_std:   int = READ_LEN_STD) -> SimRead:
    """
    Generate one simulated ONT read from a random position on the reference.
    Reads drawn from repeat-covered regions are flagged `in_repeat=True`
    so that multi-mapping FP candidates can be assessed in calibration.
    """
    ref_len  = len(ref)
    read_len = max(300, int(rng.normal(read_len_mean, read_len_std)))
    strand   = int(rng.choice([1, -1]))

    # Bimodal quality: 75% good reads (Q~14), 25% low-quality (Q~6)
    is_bad_read = rng.random() < 0.25
    q_mean = 6.0 if is_bad_read else 14.0

    # Draw start position leaving room for the full template
    max_start = ref_len - read_len - 1
    if max_start <= 0:
        max_start = 1
    start  = int(rng.integers(0, max_start))
    end    = start + read_len

    template = ref[start:end]
    if strand == -1:
        template = reverse_complement(template)

    seq  = _apply_ont_errors(template, rng)
    qual = _simulate_quality(len(seq), rng, mean_q=q_mean)

    from config import T_BASECALL
    bc_time = len(seq) * T_BASECALL

    # Flag whether the read's origin overlaps any repeat copy
    in_repeat = any(s < end and e > start
                    for (s, e, _, _) in repeat_coords)

    return SimRead(
        read_id        = read_id,
        sequence       = seq,
        quality        = qual,
        length         = len(seq),
        chunks         = _make_chunks(seq, qual),
        true_ref_start = start,
        true_ref_end   = end,
        true_strand    = strand,
        basecall_time  = bc_time,
        mean_quality   = float(np.mean(qual)),
        in_repeat      = in_repeat,
    )


def generate_dataset(ref:           str,
                     repeat_coords: list,
                     n_reads:       int = N_READS_TOTAL,
                     seed:          int = RANDOM_SEED) -> list[SimRead]:
    """Generate n_reads reads; deterministic given seed."""
    rng   = np.random.default_rng(seed)
    reads = [simulate_ont_read(ref, repeat_coords, i, rng)
             for i in range(n_reads)]
    n_rep = sum(1 for r in reads if r.in_repeat)
    print(f"[Dataset] {n_reads} reads  "
          f"(mean len {np.mean([r.length for r in reads]):.0f} bp, "
          f"mean Q {np.mean([r.mean_quality for r in reads]):.1f}, "
          f"{n_rep} from repeat regions)")
    return reads


# ─── Correctness helper ───────────────────────────────────────────────────────

def is_correct_mapping(pred_start: int, pred_end: int,
                       true_start: int, true_end: int,
                       threshold:  float = 0.90) -> bool:
    """
    Correct if overlap / true_length ≥ threshold.
    Default 0.90 (90 % reciprocal overlap) matches paftools.js eval.
    """
    overlap  = max(0, min(pred_end, true_end) - max(pred_start, true_start))
    true_len = max(true_end - true_start, 1)
    return (overlap / true_len) >= threshold
