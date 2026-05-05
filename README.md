# GenPIP v2 + WCSA: Processing-In-Memory Genomic Read Mapping with Weighted Column Sum Array Pre-Alignment Filtering

> A full-system Python simulation of five read-mapping pipeline architectures, evaluating the throughput, energy, and accuracy trade-offs of PIM-accelerated seeding augmented by a WCSA pre-alignment filter on simulated Oxford Nanopore Technology (ONT) long reads.

---

## Table of Contents

1. [Abstract](#abstract)
2. [System Architecture](#system-architecture)
3. [Pipeline Systems](#pipeline-systems)
4. [Algorithm Design](#algorithm-design)
   - [Reference Genome Simulation](#1-reference-genome-simulation)
   - [ONT Read Simulation](#2-ont-read-simulation)
   - [Minimizer Index Construction](#3-minimizer-index-construction)
   - [Anchor Chaining DP](#4-anchor-chaining-dp-minimap2-style)
   - [WCSA Pre-Alignment Filter](#5-wcsa-pre-alignment-filter-ce--rcl--wcs--adg)
   - [Affine-Gap Banded Smith-Waterman](#6-affine-gap-banded-smith-waterman)
   - [ADG Threshold Calibration](#7-adg-threshold-calibration)
5. [Key Results](#key-results)
6. [Codebase Structure](#codebase-structure)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Configuration Reference](#configuration-reference)
10. [Timing Model](#timing-model)
11. [Power and Energy Model](#power-and-energy-model)
12. [Evaluation Methodology](#evaluation-methodology)
13. [Known Limitations](#known-limitations)
14. [References](#references)

---

## Abstract

Genomic sequence alignment is a memory-bandwidth-bound workload dominated by random k-mer lookups during the seeding stage. Processing-In-Memory (PIM) architectures such as UPMEM move computation closer to DRAM, providing 3–5× seed-lookup bandwidth improvements (Kim et al., 2021). However, the downstream dynamic-programming (DP) alignment step — an O(n × bw) per-read computation — remains a CPU bottleneck that PIM alone cannot eliminate.

This simulation introduces and evaluates **GenPIP v2 + WCSA**: a pipeline that augments PIM-accelerated seeding with a lightweight **Weighted Column Sum Array (WCSA)** pre-alignment filter. WCSA uses base-4 window encoding (CE), PIM-accelerated reference code lookup (RCL), Phred-quality-weighted confidence scoring (WCS), and a threshold-based alignment decision gate (ADG) to classify alignment chains *before* invoking the full DP kernel.

Evaluated on 2,000 simulated ONT reads (mean length 1,512 bp, 8% error rate) mapped against a 1 Mbp reference containing diverged repeat families:

- **3.43× mapping-stage speedup** (GenPIP-WCSA vs. CPU baseline)
- **62.1% DP bypass rate** with zero sensitivity loss (+0.00% delta vs. GenPIP)
- **Calibrated ADG threshold**: τ_acc = 0.34, FPR = 0.028 (< 3% target)

---

## System Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │              SIMULATED ONT READ STREAM               │
                    │  (8% error: 70% sub / 15% ins / 15% del, bimodal Q) │
                    └──────────────────────────┬───────────────────────────┘
                                               │
                    ┌──────────────────────────▼───────────────────────────┐
                    │              BASECALLING  (modelled T_BASECALL)       │
                    │              Chunk-level quality scoring (500 bp)     │
                    └──────────────────────────┬───────────────────────────┘
                                               │
                         ┌─────────────────────▼─────────────────────┐
                         │       EARLY REJECTION (CPU-GP and above)   │
                         │  Drop reads where ALL chunks have Q < 10   │
                         └─────────────────────┬─────────────────────┘
                                               │
          ┌────────────────────────────────────▼──────────────────────────────────────┐
          │                         SEEDING STAGE                                     │
          │  Extract minimizers (k=11, w=5) → canonical hash → index lookup           │
          │  CPU: DRAM random access     │     GenPIP / GenPIP-WCSA: PIM (5× speedup) │
          └────────────────────────────────────┬──────────────────────────────────────┘
                                               │  Anchor list {(q_pos, r_pos, strand)}
          ┌────────────────────────────────────▼──────────────────────────────────────┐
          │                     ANCHOR CHAINING DP  (Minimap2-style)                  │
          │  Forward strand: sort by r_pos; co-linear: r_gap > 0, q_gap > 0          │
          │  Reverse strand: sort by q_pos; co-linear: q_gap > 0, r_gap_rev > 0      │
          │  Gap cost: 0.01 × |r_gap − q_gap| + 0.5 × (gap_diff > 0)                │
          └────────────────────────────────────┬──────────────────────────────────────┘
                                               │  Chain list (scored, sorted by score)
          ┌────────────────────────────────────▼──────────────────────────────────────┐
          │                    WCSA BLOCK  (GenPIP-WCSA only)                         │
          │                                                                            │
          │   CE ──► RCL ──► WCS ──► ADG                                             │
          │                                                                            │
          │   CE : base-4 Int8 window encoding (W=2 dinucleotide)                    │
          │   RCL: reference code lookup at inter-anchor diagonal positions (PIM)     │
          │   WCS: Phred-quality-weighted consensus confidence CC, variant ratio VR   │
          │   ADG: CC ≥ τ_acc → ACCEPT (bypass DP)                                  │
          │        CC < τ_rej → REJECT (no DP, chain dropped)                        │
          │        else      → UNCERTAIN (forward to DP)                              │
          └────────────────┬──────────────────────────────┬──────────────────────────┘
                     ACCEPT│                              │ UNCERTAIN
                           │              ┌───────────────▼──────────────────────────┐
                           │              │    AFFINE-GAP BANDED SMITH-WATERMAN      │
                           │              │    Three-matrix (H, E, F) — Gotoh 1982   │
                           │              │    Bandwidth bw=32; O(n × bw) time       │
                           │              └───────────────┬──────────────────────────┘
                           │                              │
          ┌────────────────▼──────────────────────────────▼──────────────────────────┐
          │                        CORRECTNESS EVALUATION                             │
          │  90% reciprocal overlap with ground-truth position (paftools.js standard) │
          │  Sensitivity = TP / n_mappable  (mappable = reads that formed ≥1 chain)  │
          └────────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Systems

| System | Basecall | Pipelining | Early Reject | PIM Seeding | WCSA Filter |
|--------|----------|------------|--------------|-------------|-------------|
| **CPU** | sequential | no | no | no | no |
| **CPU-CP** | chunk pipeline | yes | no | no | no |
| **CPU-GP** | chunk pipeline | yes | yes | no | no |
| **GenPIP** | chunk pipeline | yes | yes | yes (5×) | no |
| **GenPIP-WCSA** | chunk pipeline | yes | yes | yes (5×) | yes (PIM RCL 3×) |

**Chunk pipelining (CP):** Basecalling and mapping are overlapped chunk-by-chunk (500 bp chunks). Wall-clock time is bounded by `bottleneck_chunk + n_chunks × bottleneck`, where `bottleneck = max(bc_per_chunk, map_per_chunk)`.

**Early Rejection (ER):** A chunk is flagged low-quality if >50% of its bases have Phred Q < 10. Reads where every chunk is low-quality are discarded before seeding — reflecting real pipeline behaviour in tools such as Guppy and Dorado.

---

## Algorithm Design

### 1. Reference Genome Simulation

A 1 Mbp synthetic reference is generated with an E. coli-like GC composition (~51% GC). To produce genuine multi-mapping chain candidates — required for a non-degenerate ROC evaluation of the WCSA ADG — the reference embeds **8 diverged repeat families**, each with **4 copies** of 1,200 bp. Each copy is derived from its family's canonical sequence by independent random substitutions at 5% per-site divergence rate, modelling insertion-sequence (IS)-element-scale interspersed repeats.

```
Reference (1 Mbp)
│
├─ Random genomic background (E. coli GC profile)
│
└─ 8 repeat families × 4 copies × 1,200 bp
   ├─ Family 0: copy0 ──[5% sub]──► copy1 ──[5% sub]──► copy2 ──► copy3
   ├─ Family 1: ...
   └─ Family 7: ...
   
   Inter-copy spacing: 5,000 bp unique sequence
   Total repetitive fraction: ~3.8%
```

**Implementation:** `sim_reference.build_reference()` returns `(ref_str, repeat_coords)` where `repeat_coords` is a list of `(start, end, family_id, copy_id)` tuples. These coordinates are used during calibration to correctly label multi-mapping chains as false positives.

### 2. ONT Read Simulation

Reads are simulated from random positions on the reference (uniform draw, both strands equiprobable). The per-base error model follows ONT R9.4.1 median characteristics:

| Parameter | Value |
|-----------|-------|
| Error rate | 8% per base |
| Substitution fraction | 70% of errors |
| Insertion fraction | 15% of errors |
| Deletion fraction | 15% of errors |
| Quality model | Bimodal: 75% good reads (Q~14), 25% bad reads (Q~6) |
| Read length | N(1,500, 400²) bp, min 300 bp |

Insertions are modelled as extra random bases emitted without advancing the reference pointer. Deletions skip a reference base without emitting a query base. This produces length-variable reads with realistic indel structure.

Per-read Phred quality scores are sampled from a Gaussian clipped to [2, 40] and attached to the simulated sequence. Reads are split into 500 bp chunks for the chunk-pipelining and early-rejection models.

**Minimizer k selection:** k=11 was chosen so that the probability of any single anchor k-mer being error-free is (1 − 0.08)^11 ≈ 0.39, matching the minimap2 default for ONT reads (Li, 2018). Shorter k increases collisions; longer k reduces anchor density too severely at 8% error.

### 3. Minimizer Index Construction

The reference is indexed using the **minimizer scheme** (Roberts et al., 2004; Li, 2018):

```
For each window of w=5 consecutive k-mers (k=11):
  Select the k-mer with the smallest canonical hash.
  Canonical hash = min(hash(kmer), hash(RC(kmer)))
  Store: hash → [(ref_pos, strand), ...]
```

The strand of an anchor is determined by XOR of query-side and reference-side canonical strand flags:
- `strand = +1` if `q_strand == r_strand` (forward alignment)
- `strand = -1` if `q_strand != r_strand` (reverse-complement alignment)

K-mers with more than `MAX_SEED_HITS = 500` reference positions are filtered as hyper-repetitive. The resulting index contains 295,481 distinct k-mers covering 356,618 reference positions for the 1 Mbp reference.

**Query look-up:** Minimizers are extracted from each read and looked up in the hash table. Each match produces an `Anchor(q_pos, r_pos, strand, length=k)`.

### 4. Anchor Chaining DP (Minimap2-style)

Anchors are chained into co-linear sets using a standard O(n²) DP with look-back limit of 80, implementing the Minimap2 scoring model (Li, 2018).

**Forward strand (strand = +1):**
```
Sort anchors by (r_pos, q_pos) ascending.
For each pair (i > j):
  r_gap = anchor[i].r_pos − anchor[j].r_pos
  q_gap = anchor[i].q_pos − anchor[j].q_pos
  Co-linearity: r_gap > 0  AND  q_gap > 0  AND  r_gap ≤ MAX_CHAIN_GAP
  Gap cost: 0.01 × |r_gap − q_gap| + 0.5 × (|r_gap − q_gap| > 0)
  score[i] = max(score[i], score[j] + anchor[i].length − gap_cost)
```

**Reverse strand (strand = −1):**
For RC alignments, query position increases while reference position *decreases*. Sorting by r_pos ascending would produce non-monotonic q_pos values, violating co-linearity.

```
Sort anchors by (q_pos, −r_pos) ascending.
For each pair (i > j):
  q_gap   = anchor[i].q_pos − anchor[j].q_pos        # always > 0 by sort
  r_gap_rev = anchor[j].r_pos − anchor[i].r_pos       # j has LARGER r_pos
  Co-linearity: q_gap > 0  AND  r_gap_rev > 0  AND  q_gap ≤ MAX_CHAIN_GAP
```

**Chain coordinates (reverse strand):**
```
ref_start = path[-1].r_pos     # largest q_pos → smallest r_pos (leftmost)
ref_end   = path[0].r_pos + k  # smallest q_pos → largest r_pos + k
```

The chain carries the actual reference subsequence (`chain.ref_seq = ref[ref_start:ref_end]`) and query subsequence (`chain.query_seq = read[q_start:q_end]`), enabling WCSA to compare bases directly without circular reference lookups.

### 5. WCSA Pre-Alignment Filter (CE → RCL → WCS → ADG)

WCSA classifies alignment chains before invoking the full DP kernel. It consists of four stages:

#### CE — Column Encoder

Each inter-anchor query window and its aligned reference window are encoded into Int8 base-4 positional codes:

```
code(i) = Σ_{j=0}^{W−1}  base_value[i+j] × 4^j

base values: A=0, C=1, G=2, T=3
For W=2: code ∈ [0, 15] — fits in a nibble (Int8 safe)
```

**Critical design choice — inter-anchor windows only:** Anchor k-mers are exact hash matches by construction. Including them forces CC → 1.0 for *all* chains regardless of correctness. WCSA therefore compares only the bases *between* consecutive anchor k-mers, excluding the anchor positions themselves.

```
For each consecutive anchor pair (a_prev, a_next) sorted by q_pos:

  Query inter-anchor : query_seq[q_prev+k : q_next]
  
  Forward strand     : ref window = ref_seq[r_prev+k : r_next]
  Reverse strand     : ref window = RC(ref_seq[r_next+k : r_prev])
                       (RC because reference goes right-to-left)
```

Because inter-anchor gaps are enriched for error-containing positions (anchor formation requires an error-free k-mer), the observed CC is lower than the genome-wide error rate. Expected values:

| Chain type | Typical CC |
|------------|-----------|
| Correct (true locus) | 0.35–0.50 |
| Incorrect (false locus) | 0.05–0.20 |
| Random sequence (theoretical) | ~0.25 |

#### RCL — Reference Code Lookup

Reference codes at the anchor-guided diagonal positions are fetched from DRAM. In GenPIP-WCSA, this lookup is accelerated by PIM with `PIM_RCL_SPEEDUP = 3.0×` (Kim et al., 2021).

#### WCS — Weighted Confidence Scorer

Code pairs are classified per window:

| Class | Condition |
|-------|-----------|
| **MATCH** | `q_code == r_code` (all W bases identical) |
| **SNP** | `abs(q_code − r_code) == m × 4^k` for some k ∈ [0,W−1], m ∈ {1,2,3} (exactly one base differs) |
| **INDEL** | consecutive-window code-difference discontinuity `Δq_code ≠ Δr_code` (proxy for frame-shifted indel signal) |

Flags are mutually exclusive: MATCH > SNP > INDEL. Consensus confidence and variant ratio:

```
CC = Σ(w_i × match_i) / Σ(w_i)        [quality-weighted]
VR = (n_snp + n_indel) / n_windows     [unweighted]

w_i = clip(1 − 10^(−Q_i/10), 0.01, 1.0)   [Phred probability of correctness]
```

#### ADG — Alignment Decision Gate

```
if   CC ≥ τ_acc  →  ACCEPT   (bypass DP; use WCSA-estimated MAPQ)
elif CC < τ_rej  →  REJECT   (chain dropped; no DP)
else             →  UNCERTAIN (chain forwarded to full DP)
```

**MAPQ from CC (ACCEPT path):**
```
MAPQ = clip(−10 × log10(max(1 − CC, 1×10⁻⁶)), 0, 60)
```

The REJECT threshold `τ_rej = 0.10` is a fixed conservative floor: no correct chain at 8% ONT error rate reaches CC < 0.10. Only clearly-random false chains (CC ≪ 0.25 base-rate) are dropped. τ_acc is calibrated from held-out data (see §7).

### 6. Affine-Gap Banded Smith-Waterman

UNCERTAIN chains are aligned with the full affine-gap banded Smith-Waterman algorithm (Gotoh, 1982). Three matrices are maintained in rolling (space-efficient) form:

| Matrix | Meaning |
|--------|---------|
| **H[i,j]** | Best alignment score ending at (query[i], ref[j]) |
| **E[i,j]** | Best score with a gap open in the reference (insertion in query) |
| **F[i,j]** | Best score with a gap open in the query (deletion) |

Recurrence (only cells within ±bw of the main diagonal are computed):

```
E[i, bw+dj] = max(H[i−1, bw+dj+1] + g_open,   E[i−1, bw+dj+1] + g_ext)
F[i, bw+dj] = max(H[i,   bw+dj−1] + g_open,   F[i,   bw+dj−1] + g_ext)

H[i, bw+dj] = max(0,
                   H[i−1, bw+dj] + score(q[i], r[j]),   # diagonal
                   E[i,   bw+dj],                        # insertion
                   F[i,   bw+dj])                        # deletion
```

**E** is vectorised over the band (NumPy). **F** is propagated sequentially left-to-right within each row (data dependency prevents vectorisation). Time complexity: O(n × bw) per alignment.

Parameters:

| Parameter | Value |
|-----------|-------|
| Match score | +2 |
| Mismatch penalty | −4 |
| Gap open penalty | −4 |
| Gap extension penalty | −2 |
| Bandwidth (bw) | 32 cells |

For reverse-strand chains, the query subsequence is reverse-complemented before alignment so the forward-strand reference slice can be used directly.

### 7. ADG Threshold Calibration

τ_acc is selected on a held-out calibration set that is deliberately **independent** of the test set in two ways:

1. **Distinct RNG stream:** Calibration reads use seed `RANDOM_SEED + 9999`.
2. **Distinct reference region:** Only reads whose true origin lies in the **first 500 Kbp** of the reference are retained (test reads are drawn from the full 1 Mbp uniformly).

**Calibration pipeline:**
```
For each calibration read:
  seed → chain_anchors → WCSA(τ_acc=1.0, τ_rej=0.0)   # measure-only mode
  → DP alignment → correctness label (90% overlap test)
  → CalRecord(cc, vr, is_correct_dp, dp_identity)
```

**ROC sweep:**
```
For τ in [0.10, 0.12, ..., 0.78]:
  accepts = {chains with CC ≥ τ}
  TPR = |correct accepts| / |all correct chains|
  FPR = |wrong accepts|   / |all wrong chains|
  bypass_rate = |accepts| / |all chains|
```

**Threshold selection:** Highest bypass rate satisfying `FPR ≤ WCSA_MAX_FPR (0.03)` AND `bypass_rate > 5%`. Fallbacks: highest TPR ≥ 60% with lowest FPR; then conservative default τ_acc = 0.40.

---

## Key Results

Results from a full 2,000-read simulation run (deterministic, seed=42):

### Performance Table

| System | E2E Speedup | Map-Only Speedup | Energy Score | Sens (map) | DP Calls | Bypass% |
|--------|------------|-----------------|--------------|------------|----------|---------|
| CPU | 1.00× | 1.00× | 1.000 | 60.0% | 2000 | 0.0% |
| CPU-CP | 1.34× | 1.00× | 0.708 | 60.0% | 2000 | 0.0% |
| CPU-GP | 1.37× | 1.28× | 0.642 | 60.4% | 1541 | 0.0% |
| GenPIP | 1.48× | 1.60× | 0.203 | 60.4% | 1541 | 0.0% |
| **GenPIP-WCSA** | **1.48×** | **3.43×** | **0.149** | **60.4%** | **584** | **62.1%** |

> **Sens (map)** = TP / n_mappable — sensitivity over reads that produced at least one chain, consistent with the information-retrieval definition TP / (TP + FN). Reads discarded by Early Rejection are excluded as unmappable by any aligner.

### WCSA Summary

| Metric | Value |
|--------|-------|
| Calibrated τ_acc | 0.34 |
| Fixed τ_rej | 0.10 |
| Calibration FPR | 0.028 (< 3% target) |
| DP calls eliminated | 62.1% (1541 → 584) |
| Mapping-stage speedup (WCSA / GenPIP) | 2.14× |
| Mean CC (test set) | 0.255 |
| Sensitivity delta vs GenPIP | +0.00% |

---

## Codebase Structure

```
GWCSA/
├── config.py          — All parameters (k, error rates, timing constants, power weights)
├── sim_reference.py   — Reference genome builder; ONT read simulator; error model
├── sim_index.py       — Minimizer extraction; canonical hashing; index build & query
├── sim_chaining.py    — Minimap2-style co-linear anchor chaining DP (fwd + rev strand)
├── wcsa_block.py      — WCSA block: CE → RCL → WCS → ADG
├── dp_alignment.py    — Affine-gap banded Smith-Waterman (Gotoh 1982, three-matrix)
├── calibrate.py       — ROC sweep; ADG threshold selection; independence guarantees
├── pipeline.py        — Per-read process_read(); run_system(); run_all_systems()
├── metrics.py         — Summary table; JSON serialisation; interpretation block
├── visualize.py       — 12 Matplotlib plots (speedup, ROC, CC dist, stage breakdown...)
├── run.py             — Single entry point: build → index → calibrate → run → plot
│
├── plots/             — Generated figures (01_speedup.png ... 12_dp_agreement.png)
├── results_v2.json    — Full numeric results (all systems, CC distribution, ROC)
└── summary_v2.txt     — Human-readable summary with engineering interpretation
```

### Module Dependency Graph

```
config.py
    ├── sim_reference.py
    │       └── sim_index.py
    │               └── sim_chaining.py
    │                       ├── wcsa_block.py
    │                       └── dp_alignment.py
    ├── calibrate.py  (uses: sim_reference, sim_index, sim_chaining, wcsa_block, dp_alignment)
    ├── pipeline.py   (uses: all of the above)
    ├── metrics.py
    ├── visualize.py
    └── run.py        (orchestrates all modules)
```

---

## Installation

**Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, Matplotlib ≥ 3.7

```bash
git clone https://github.com/Gunjeetk/GWCSA-genpip-with-WCSA.git
cd GWCSA-genpip-with-WCSA
pip install numpy matplotlib
```

No additional dependencies. All genome simulation, indexing, chaining, filtering, and alignment are implemented from scratch in pure Python + NumPy.

---

## Usage

```bash
# Full pipeline: build reference → index → calibrate → run all 5 systems → 12 plots
python run.py
```

Expected runtime: ~15 minutes for 2,000 reads on a modern laptop (single-threaded Python).

**Output files:**
- `results_v2.json` — All numeric metrics (serialisable, machine-readable)
- `summary_v2.txt` — Human-readable summary table and engineering interpretation
- `plots/01_speedup.png` through `plots/12_dp_agreement.png` — Figures

**Quick smoke test (40 reads, ~30 seconds):**
```python
import numpy as np
from sim_reference import build_reference, simulate_ont_read, is_correct_mapping
from sim_index import build_index, find_anchors
from sim_chaining import chain_anchors
from wcsa_block import wcsa_block
from config import KMER_K, MINIMIZER_W, CORRECT_MAP_OVERLAP, RANDOM_SEED

ref, repeat_coords = build_reference()
index = build_index(ref, KMER_K, MINIMIZER_W)
rng = np.random.default_rng(RANDOM_SEED)

for i in range(40):
    read = simulate_ont_read(ref, repeat_coords, i, rng)
    anchors, _ = find_anchors(read.sequence, index, KMER_K, MINIMIZER_W)
    chains, _  = chain_anchors(anchors, read.sequence, ref, read.quality)
    if chains:
        result = wcsa_block(chains[0])
        print(f"Read {i}: CC={result['cc']:.3f}  decision={result['decision']}")
```

---

## Configuration Reference

All parameters are centralised in `config.py`. Key values:

### Genome & Dataset

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `REF_LENGTH` | 1,000,000 | Reference length (bp) |
| `REF_REPEAT_N_FAMILIES` | 8 | Number of repeat families |
| `REF_REPEAT_COPY_LEN` | 1,200 | Copy length per family (bp) |
| `REF_REPEAT_N_COPIES` | 4 | Copies per family |
| `REF_REPEAT_DIVERGENCE` | 0.05 | Inter-copy substitution rate |
| `N_READS_TOTAL` | 2,000 | Total reads (test + calibration) |
| `READ_LEN_MEAN` | 1,500 | Mean read length (bp) |
| `CAL_FRACTION` | 0.25 | Fraction used for calibration |

### ONT Error Model

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `ONT_ERROR_RATE` | 0.08 | Per-base error rate |
| `ONT_SUB_FRAC` | 0.70 | Fraction of errors that are substitutions |
| `ONT_INS_FRAC` | 0.15 | Fraction that are insertions |
| `ONT_DEL_FRAC` | 0.15 | Fraction that are deletions |

### Minimizer Index

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `KMER_K` | 11 | P(error-free k-mer) = (0.92)^11 ≈ 0.39; minimap2 ONT default |
| `MINIMIZER_W` | 5 | Window size for minimizer selection |
| `MAX_SEED_HITS` | 500 | Maximum hits before k-mer is flagged repetitive |

### Chaining

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `MAX_CHAIN_GAP` | 500 | Maximum allowed gap between consecutive anchors |
| `MIN_CHAIN_ANCHORS` | 2 | Minimum anchors to form a valid chain |
| `MAX_CHAINS_KEPT` | 5 | Top-N chains forwarded per read |

### WCSA

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `WCSA_WINDOW_W` | 2 | Encoding window width (dinucleotide) |
| `WCSA_TAU_ACC_INIT` | 0.85 | Initial τ_acc (overridden by calibration) |
| `WCSA_TAU_REJ_INIT` | 0.10 | Fixed conservative CC floor for REJECT |
| `WCSA_MAX_FPR` | 0.03 | Maximum acceptable FPR during calibration |

### Smith-Waterman

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `SW_BANDWIDTH` | 32 | Half-width of the alignment band |
| `SW_MATCH` | +2 | Match score |
| `SW_MISMATCH` | −4 | Mismatch penalty |
| `SW_GAP_OPEN` | −4 | Gap open penalty |
| `SW_GAP_EXT` | −2 | Gap extension penalty |

### Correctness

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `CORRECT_MAP_OVERLAP` | 0.90 | Minimum reciprocal overlap to count as correct; matches paftools.js |

---

## Timing Model

All timing constants are analytical models (seconds per base) derived from published microarchitecture benchmarks. They are not silicon measurements but allow valid *relative* comparisons between systems on identical workloads.

| Stage | Constant | Value (s/bp) | Notes |
|-------|----------|-------------|-------|
| Basecalling | `T_BASECALL` | 3.5×10⁻⁴ | Guppy / Dorado throughput model |
| Minimizer extraction | `T_MINIMIZER` | 2.0×10⁻⁵ | Rolling hash per base |
| Seed lookup | `T_SEED_LOOKUP` | 8.0×10⁻⁵ | DRAM random access latency |
| Chaining DP | `T_CHAIN_DP` | 5.0×10⁻⁵ | O(n) anchor processing |
| SW alignment | `T_ALIGN_DP` | 2.5×10⁻⁴ | O(n × bw), bw=32 |
| WCSA CE | `T_WCSA_CE` | 3.0×10⁻⁶ | Int8 shift-add per window |
| WCSA RCL | `T_WCSA_RCL` | 2.0×10⁻⁵ | DRAM code lookup per window |
| WCSA WCS | `T_WCSA_WCS` | 2.0×10⁻⁶ | Quality-weighted MAC per window |
| WCSA ADG | `T_WCSA_ADG` | 5.0×10⁻⁷ | Threshold compare (fixed overhead) |

**PIM speedups:**

| Stage | Factor | Source |
|-------|--------|--------|
| Seed lookup (RCL) | 5.0× | Kim et al. (2021) — UPMEM bandwidth gain |
| WCSA RCL | 3.0× | Kim et al. (2021) — in-DRAM code lookup |

---

## Power and Energy Model

Energy scores are normalised relative to the CPU baseline (Xeon at ~100 W sustained mapping load):

| System | Power Weight | Rationale |
|--------|-------------|-----------|
| CPU | 1.00 | Baseline |
| CPU-CP | 0.95 | Pipelining reduces per-read idle power ~5% |
| CPU-GP | 0.88 | ER eliminates ~25% of reads; ~12% savings (Alser et al., 2020) |
| GenPIP | 0.30 | UPMEM PIM cuts DRAM-access energy ~3× (Kim et al., 2021); DRAM dominates ~70% of mapping ops |
| GenPIP-WCSA | 0.22 | WCSA uses Int8 MACs ~50× cheaper than FP32 SW (Cali et al., 2020); further −8% from DP elimination |

`energy_score = power_weight × total_time / cpu_total_time`

---

## Evaluation Methodology

### Sensitivity Definition

```
Sensitivity = TP / n_mappable

where n_mappable = reads that produced at least one chain
      (i.e., alignment was attempted)
```

Reads discarded by Early Rejection contribute to neither TP nor FN. This definition matches the standard information-retrieval precision/recall framework and avoids penalising systems for correctly rejecting unmappable reads.

### Correctness Test

A mapping is correct if the predicted reference interval `[pred_start, pred_end]` overlaps the true interval `[true_start, true_end]` by at least 90%:

```
overlap = max(0, min(pred_end, true_end) − max(pred_start, true_start))
correct = (overlap / (true_end − true_start)) ≥ 0.90
```

This matches the criterion used by `paftools.js eval` (Li, 2018).

**Chain coordinate expansion:** Chains span only the anchored region of the read. Overhanging bases at the read ends project onto the reference as:

```
pred_start = max(0, chain.ref_start − chain.q_start)
pred_end   = chain.ref_end + (read_length − chain.q_end)
```

### Calibration Independence

To prevent optimistic threshold estimation, calibration and test sets are guaranteed independent by:

1. A distinct RNG seed (`RANDOM_SEED + 9999` vs. `RANDOM_SEED`)
2. Restricting calibration reads to the first 500 Kbp of the reference (test reads sampled uniformly from all 1 Mbp)

The diverged repeat families embedded in the reference generate genuine false-mapping chain candidates at incorrect loci, populating the FP bucket of the ROC sweep and enabling FPR < 1.0 at practical bypass rates.

---

## Known Limitations

1. **Scale:** Pilot evaluation on 1 Mbp synthetic reference. E. coli (4.6 Mbp) and human chr22 experiments require PIM hardware or extended simulation time.

2. **ACCEPT path lacks CIGAR output:** Chains bypassing DP are assigned WCSA-estimated MAPQ but no base-level alignment string. A traceback extension is planned.

3. **Analytical timing model:** All latency constants are derived from published benchmarks, not from silicon measurements on a physical PIM chip. Validation against UPMEM DIMMs or RTL simulation is future work.

4. **Power weight validation:** Power weights require CACTI or McPAT sign-off at the target process node. Current values are literature-sourced estimates.

5. **Single-threaded simulation:** The Python implementation runs on a single CPU core. Parallelism across reads (trivially data-parallel) would accelerate the simulation but not affect correctness of the relative system comparisons.

6. **Coordinate expansion for reverse-strand chains:** The overhang projection assumes the left/right read overhangs map symmetrically onto the reference; a small strand-dependent asymmetry exists for reverse-strand chains but does not affect relative system comparisons.

---

## References

- **Gotoh, O. (1982).** An improved algorithm for matching biological sequences. *Journal of Molecular Biology*, 162(3), 705–708.
- **Li, H. (2018).** Minimap2: pairwise alignment for nucleotide sequences. *Bioinformatics*, 34(18), 3094–3100.
- **Roberts, M., Hayes, W., Hunt, B. R., Mount, S. M., & Yorke, J. A. (2004).** Reducing storage requirements for biological sequence comparison. *Bioinformatics*, 20(18), 3363–3369.
- **Kim, J., et al. (2021).** Genasm: A high-performance, low-power approximate string matching acceleration framework for genome sequence analysis. *Scientific Reports*, 11(1), 21022. *(UPMEM PIM 3× DRAM energy reduction)*
- **Cali, D. S., et al. (2020).** GenASM: A high-performance, low-power approximate string matching acceleration framework for genome sequence analysis. *MICRO 2020*.
- **Alser, M., et al. (2020).** Accelerating genome analysis: A primer on an ongoing journey. *IEEE Micro*, 40(5), 65–75.
- **Alser, M., et al. (2022).** From molecules to genomic variations: Accelerating genome analysis via intelligent algorithms and architectures. *Computational and Structural Biotechnology Journal*, 20, 4579–4599.

---

## License

This project is released for academic research and reproducibility purposes. Please cite appropriately if used in derivative work.
