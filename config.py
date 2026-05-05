# =============================================================
#  config.py  — all parameters for GenPIP v2 + WCSA
# =============================================================

# ── Reproducibility ──────────────────────────────────────────
RANDOM_SEED = 42

# ── Reference genome (simulated, E. coli-inspired scale) ─────
# 1 Mbp pilot; authors note E. coli = 4.6 Mbp and human chr22
# are planned follow-ups requiring hardware acceleration.
REF_LENGTH        = 1_000_000   # 1 Mbp
REF_GC_PROB       = [0.245, 0.255, 0.255, 0.245]  # ~51% GC, E. coli-like

# ── Repeat model ─────────────────────────────────────────────
# Tandem-repeat families with intra-family divergence, modelling
# the repetitive fraction that creates genuine multi-mapping
# candidates required for a non-trivial WCSA ROC evaluation.
REF_REPEAT_N_FAMILIES  = 8      # distinct repeat families
REF_REPEAT_COPY_LEN    = 1_200  # bp per copy (≈ IS-element scale)
REF_REPEAT_N_COPIES    = 4      # copies per family (spread across ref)
REF_REPEAT_DIVERGENCE  = 0.05   # 5% between-copy divergence
# Fraction of reference covered by repeats ≈
#   8 × 4 × 1200 / 1_000_000 ≈ 3.8%  (low-complexity fraction)

# ── Dataset ──────────────────────────────────────────────────
N_READS_TOTAL     = 2000        # 75 % test  +  25 % calibration
READ_LEN_MEAN     = 1_500       # bp — representative R9/R10 read length
READ_LEN_STD      = 400
CAL_FRACTION      = 0.25        # 500 reads for calibration

# ── ONT error model ──────────────────────────────────────────
ONT_ERROR_RATE    = 0.08        # 8 % per-base error (R9.4.1 median)
ONT_SUB_FRAC      = 0.70        # 70 % substitutions
ONT_INS_FRAC      = 0.15        # 15 % insertions
ONT_DEL_FRAC      = 0.15        # 15 % deletions
ONT_QUAL_MEAN     = 12.0
ONT_QUAL_STD      = 4.0

# ── Minimizer index ──────────────────────────────────────────
# k=11 chosen so P(seed error-free) = (1-0.08)^11 ≈ 0.39,
# matching the minimap2 default for noisy ONT reads.
KMER_K            = 11          # k-mer length
MINIMIZER_W       = 5           # minimizer window (denser than k=15/w=10)
MAX_SEED_HITS     = 500         # filter hyper-repetitive k-mers

# ── Chaining DP ──────────────────────────────────────────────
MAX_CHAIN_GAP     = 500
MIN_CHAIN_ANCHORS = 2
MAX_CHAINS_KEPT   = 5

# ── Banded Smith-Waterman (affine gap, three-matrix) ─────────
SW_BANDWIDTH      = 32
SW_MATCH          = 2
SW_MISMATCH       = -4
SW_GAP_OPEN       = -4          # penalty to OPEN a gap
SW_GAP_EXT        = -2          # penalty per additional gap base

# ── WCSA ─────────────────────────────────────────────────────
WCSA_WINDOW_W     = 2           # encoding window width W (dinucleotide)
WCSA_WEIGHTS      = [1, 4]      # base-4 positional weights [4^0, 4^1]
WCSA_TAU_ACC_INIT = 0.85
WCSA_TAU_REJ_INIT = 0.40
WCSA_MAX_FPR      = 0.03        # calibration target: FPR < 3 %

# ── Correctness threshold ─────────────────────────────────────
# 90 % reciprocal overlap; consistent with paftools.js eval criteria.
CORRECT_MAP_OVERLAP = 0.90

# ── Modelled timing constants (seconds / base) ───────────────
T_BASECALL        = 3.5e-4
T_MINIMIZER       = 2.0e-5
T_SEED_LOOKUP     = 8.0e-5
T_ANCHOR          = 1.0e-5
T_CHAIN_DP        = 5.0e-5
T_ALIGN_DP        = 2.5e-4      # O(n × bw)
T_OVERHEAD        = 5.0e-6
T_WCSA_CE         = 3.0e-6      # Int8 shift-add per window
T_WCSA_RCL        = 2.0e-5      # DRAM code lookup per window
T_WCSA_WCS        = 2.0e-6      # multiply-accumulate per window
T_WCSA_ADG        = 5.0e-7      # threshold compare (fixed overhead)

# ── PIM speedup ───────────────────────────────────────────────
PIM_SEED_SPEEDUP  = 5.0
PIM_RCL_SPEEDUP   = 3.0

# ── Power weights (literature-calibrated) ────────────────────
# Normalised to CPU Xeon baseline (≈ 100 W sustained at mapping load).
# CPU-CP: pipelining reduces per-read idle time ~5%.
# CPU-GP: ER eliminates ~25% reads; savings ≈ 12% (Alser et al., 2020).
# GenPIP: UPMEM PIM DRAM cuts DRAM-access energy ~3× (Kim et al., 2021,
#         Sci. Rep.); memory dominates mapping (~70% of ops) → 0.30.
# GenPIP-WCSA: WCSA uses Int8 MACs ~50× cheaper than FP32 SW
#              (Cali et al., 2020, GenASM); further −8 % from DP removal.
POWER_WEIGHT = {
    "CPU"         : 1.00,
    "CPU-CP"      : 0.95,
    "CPU-GP"      : 0.88,
    "GenPIP"      : 0.30,
    "GenPIP-WCSA" : 0.22,
}

# ── ER threshold ──────────────────────────────────────────────
ER_LOW_QUAL_FRAC  = 0.50        # fraction of bases Q<10 → chunk rejected

# ── Quality model (bimodal; ~25 % reads are low quality) ─────
ONT_QUAL_MEAN_GOOD = 14.0
ONT_QUAL_MEAN_BAD  =  6.0
ONT_LOW_QUAL_FRAC  = 0.25
CHUNK_SIZE        = 500

# ── Output ────────────────────────────────────────────────────
PLOTS_DIR         = "plots"
RESULTS_JSON      = "results_v2.json"
SUMMARY_TXT       = "summary_v2.txt"
