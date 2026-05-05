# =============================================================
#  visualize.py  — all plots (9 original + 3 new critical ones)
# =============================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import PLOTS_DIR
from calibrate import CalRecord

os.makedirs(PLOTS_DIR, exist_ok=True)

SYSTEMS = ["CPU", "CPU-CP", "CPU-GP", "GenPIP", "GenPIP-WCSA"]
COLORS  = {
    "CPU":          "#4a90d9",
    "CPU-CP":       "#5cb85c",
    "CPU-GP":       "#f0ad4e",
    "GenPIP":       "#d9534f",
    "GenPIP-WCSA":  "#9b59b6",
}
DARK = "#1a1a2e"
MID  = "#16213e"
TEXT = "#e0e0e0"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": MID,
    "axes.edgecolor": "#444466", "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT, "text.color": TEXT,
    "grid.color": "#333355", "grid.linestyle": "--", "grid.alpha": 0.4,
    "font.family": "monospace", "legend.facecolor": "#0f1030",
    "legend.edgecolor": "#444466",
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] {path}")


def _bar_group(ax, data_dict, ylabel, title):
    x    = np.arange(len(SYSTEMS))
    vals = [data_dict[s] for s in SYSTEMS]
    cols = [COLORS[s] for s in SYSTEMS]
    bars = ax.bar(x, vals, color=cols, width=0.6, edgecolor="#111", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(SYSTEMS, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(title, fontsize=11, fontweight="bold")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    hi = SYSTEMS.index("GenPIP-WCSA")
    bars[hi].set_edgecolor("white"); bars[hi].set_linewidth(2.0)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + max(vals)*0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)


# ── Plot 1: End-to-end and mapping-only speedup ───────────────────────────────
def plot_speedup(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: end-to-end speedup
    ax = axes[0]
    _bar_group(ax, {s: results[s]["speedup"] for s in SYSTEMS},
               "Speedup (×) vs CPU", "End-to-End Speedup")
    ax.axhline(1.0, color="#888", linestyle=":", lw=1)

    # Right panel: mapping-only speedup (excludes basecalling)
    ax2 = axes[1]
    _bar_group(ax2, {s: results[s]["speedup_map_only"] for s in SYSTEMS},
               "Speedup (×) vs CPU mapping", "Mapping-Only Speedup (excl. basecall)")
    ax2.axhline(1.0, color="#888", linestyle=":", lw=1)

    fig.suptitle("System Speedup Over CPU Baseline", fontsize=12, fontweight="bold")
    _save(fig, "01_speedup.png")


# ── Plot 2: Energy ────────────────────────────────────────────────────────────
def plot_energy(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    _bar_group(ax, {s: results[s]["energy_score"] for s in SYSTEMS},
               "Normalised Energy Score", "Energy Efficiency (↓ better)")
    _save(fig, "02_energy.png")


# ── Plot 3: Sensitivity over mappable reads (COMPUTED from ground truth) ──────
def plot_accuracy(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = {s: results[s]["sensitivity_pct"] for s in SYSTEMS}
    _bar_group(ax, vals,
               "Sensitivity (%) — TP/(TP+FN), mappable reads, overlap ≥ 90%",
               "Mapping Sensitivity — Computed over Mappable Reads")
    # Auto y-axis: floor at the lowest bar minus 5 %, ceiling at 105 %
    lo = max(0, min(vals.values()) - 5)
    ax.set_ylim(lo, 105)
    _save(fig, "03_accuracy.png")


# ── Plot 4: Stage-wise time breakdown ─────────────────────────────────────────
def plot_time_breakdown(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    x, w = np.arange(len(SYSTEMS)), 0.35
    bc = [results[s]["total_basecall_s"] for s in SYSTEMS]
    mp = [results[s]["total_map_s"]      for s in SYSTEMS]
    ax.bar(x-w/2, bc, w, label="Basecall", color="#4a90d9", alpha=0.85)
    ax.bar(x+w/2, mp, w, label="Mapping",  color="#d9534f", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(SYSTEMS, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Time (s)"); ax.set_title("Basecall vs Mapping Time", fontweight="bold")
    ax.legend(fontsize=8); ax.yaxis.grid(True); ax.set_axisbelow(True)
    _save(fig, "04_time_breakdown.png")


# ── Plot 5: Speedup vs Energy scatter ────────────────────────────────────────
def plot_speedup_energy(results):
    fig, ax = plt.subplots(figsize=(8, 6))
    for s in SYSTEMS:
        r   = results[s]
        sz  = 220 if s == "GenPIP-WCSA" else 120
        ax.scatter(r["speedup"], r["energy_score"], color=COLORS[s], s=sz,
                   zorder=3, edgecolors="white" if s == "GenPIP-WCSA" else "#555", lw=1.5)
        ax.annotate(s, (r["speedup"], r["energy_score"]),
                    xytext=(8, 4), textcoords="offset points", fontsize=8, color=COLORS[s])
    ax.set_xlabel("Speedup (×)"); ax.set_ylabel("Energy Score (↓)")
    ax.set_title("Speedup vs Energy Trade-off", fontweight="bold")
    ax.grid(True); ax.set_axisbelow(True)
    _save(fig, "05_speedup_energy.png")


# ── Plot 6: DP calls — baseline vs WCSA ──────────────────────────────────────
def plot_dp_calls(results):
    fig, ax = plt.subplots(figsize=(9, 5))
    gp  = results["GenPIP"]
    gw  = results["GenPIP-WCSA"]
    acc = gw["n_wcsa_accept"]
    rej = gw["n_wcsa_reject"]
    unc = gw["n_wcsa_uncertain"]

    ax.bar(0, gp["n_dp_calls"], 0.5, color="#d9534f", label="Full DP (GenPIP)")
    ax.bar(1, acc,       0.5, color="#2ecc71",               label="WCSA ACCEPT (no DP)")
    ax.bar(1, rej, 0.5,  bottom=acc, color="#e74c3c",        label="WCSA REJECT (no DP)")
    ax.bar(1, unc, 0.5,  bottom=acc+rej, color="#f39c12",    label="UNCERTAIN → DP runs")

    bypass = 100*(acc+rej)/max(acc+rej+unc, 1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["GenPIP (baseline)", "GenPIP-WCSA"])
    ax.set_ylabel("DP alignment calls")
    ax.set_title(f"DP Call Reduction — {bypass:.1f}% bypass rate", fontweight="bold")
    ax.legend(fontsize=8); ax.yaxis.grid(True); ax.set_axisbelow(True)
    _save(fig, "06_dp_calls.png")


# ── Plot 7: Speedup waterfall ─────────────────────────────────────────────────
def plot_waterfall(results):
    fig, ax = plt.subplots(figsize=(10, 5))
    speedups = [results[s]["speedup"] for s in SYSTEMS]
    x        = np.arange(len(SYSTEMS))
    bars = ax.bar(x, speedups, color=[COLORS[s] for s in SYSTEMS], width=0.55)
    for i in range(len(SYSTEMS)-1):
        ax.annotate("", xy=(i+1, speedups[i+1]), xytext=(i, speedups[i]),
                    arrowprops=dict(arrowstyle="->", color="#aaaacc", lw=1.2))
    ax.set_xticks(x); ax.set_xticklabels(SYSTEMS, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Speedup (×)"); ax.set_title("Incremental Speedup Progression", fontweight="bold")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    for b, v in zip(bars, speedups):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.2f}×",
                ha="center", va="bottom", fontsize=8)
    _save(fig, "07_waterfall.png")


# ── Plot 8: Stage-level breakdown GenPIP vs GenPIP-WCSA ──────────────────────
def plot_stage_breakdown(results):
    fig, ax = plt.subplots(figsize=(11, 6))
    gp = results["GenPIP"];      gw = results["GenPIP-WCSA"]
    n  = gp["n_reads"]

    def s(r, k): return r.get(k, 0.0) / max(n, 1)

    gp_stgs = {"Seeding": s(gp,"stage_seed"), "Chaining": s(gp,"stage_chain"),
               "Alignment": s(gp,"stage_alignment"), "Overhead": s(gp,"stage_overhead")}
    gw_stgs = {"Seeding": s(gw,"stage_seed"), "Chaining": s(gw,"stage_chain"),
               "CE": s(gw,"stage_wcsa_ce"), "RCL": s(gw,"stage_wcsa_rcl"),
               "WCS": s(gw,"stage_wcsa_wcs"), "ADG": s(gw,"stage_wcsa_adg"),
               "Align\n(reduced)": s(gw,"stage_alignment"), "Overhead": s(gw,"stage_overhead")}

    stg_colors = {
        "Seeding":"#4a90d9","Chaining":"#5cb85c","Alignment":"#d9534f",
        "CE":"#8e44ad","RCL":"#9b59b6","WCS":"#a569bd","ADG":"#bb8fce",
        "Align\n(reduced)":"#e74c3c","Overhead":"#95a5a6",
    }
    x_pos = [0, 1.6]
    bottoms = [0.0, 0.0]
    for i, (label, stages) in enumerate([("GenPIP", gp_stgs), ("GenPIP-WCSA", gw_stgs)]):
        for stage, val in stages.items():
            ax.bar(x_pos[i], val, 0.6, bottom=bottoms[i],
                   color=stg_colors.get(stage, "#888"), edgecolor="#111", lw=0.5)
            if val > max(max(gp_stgs.values()), 1e-6)*0.03:
                ax.text(x_pos[i], bottoms[i]+val/2, stage,
                        ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
            bottoms[i] += val

    pct = 100*(bottoms[1]-bottoms[0])/max(bottoms[0], 1e-9)
    mid_x = (x_pos[0]+x_pos[1])/2; mid_y = (bottoms[0]+bottoms[1])/2
    ax.annotate("", xy=(x_pos[1], bottoms[1]), xytext=(x_pos[0], bottoms[0]),
                arrowprops=dict(arrowstyle="<->", color="#f1c40f", lw=2))
    ax.text(mid_x, mid_y, f"{pct:+.1f}%\ntotal", ha="center", fontsize=9,
            color="#f1c40f", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=DARK, ec="#f1c40f", alpha=0.85))
    ax.set_xticks(x_pos); ax.set_xticklabels(["GenPIP","GenPIP-WCSA"], fontsize=10, fontweight="bold")
    ax.set_ylabel("Time per read (s)"); ax.set_title("Stage-Level Time Breakdown", fontweight="bold")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    patches = [mpatches.Patch(color=stg_colors.get(s,"#888"), label=s) for s in gw_stgs]
    ax.legend(handles=patches, fontsize=7, loc="upper right", ncol=2)
    _save(fig, "08_stage_breakdown.png")


# ── Plot 9: WCSA score distribution ─────────────────────────────────────────
def plot_score_distribution(results, cal_records: list[CalRecord],
                             tau_acc: float):
    """
    Histogram of CC values split by whether DP confirmed correct mapping.
    Expected: correct chains peak near 1.0; wrong near 0.
    CRITICAL diagnostic — proves WCSA score is meaningful.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, 1, 41)

    cc_correct   = [r.cc for r in cal_records if r.is_correct_dp]
    cc_incorrect = [r.cc for r in cal_records if not r.is_correct_dp]

    ax.hist(cc_correct,   bins=bins, alpha=0.75, color="#2ecc71",
            label=f"DP-confirmed correct (n={len(cc_correct)})")
    ax.hist(cc_incorrect, bins=bins, alpha=0.75, color="#e74c3c",
            label=f"DP-confirmed wrong (n={len(cc_incorrect)})")
    ax.axvline(tau_acc, color="white", linestyle="--", lw=2,
               label=f"τ_acc = {tau_acc:.2f}")
    ax.set_xlabel("WCSA Consensus Confidence (CC)", fontsize=10)
    ax.set_ylabel("Count of chains", fontsize=10)
    ax.set_title("WCSA Score Distribution — Correct vs Incorrect Chains",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    _save(fig, "09_score_distribution.png")


# ── Plot 10: ROC curve ────────────────────────────────────────────────────────
def plot_roc(roc_pts: list[dict], tau_acc: float):
    """
    ROC of ADG threshold sweep.
    Shows the operating point chosen by calibration.
    AUC > 0.85 means WCSA scoring is informative.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    fprs = [p["fpr"] for p in roc_pts]
    tprs = [p["tpr"] for p in roc_pts]
    taus = [p["tau_acc"] for p in roc_pts]

    ax.plot(fprs, tprs, "b-o", markersize=5, lw=2)
    ax.plot([0,1],[0,1],"k--", alpha=0.4, label="Random")

    # Mark chosen operating point
    try:
        idx = next(i for i,p in enumerate(roc_pts) if abs(p["tau_acc"]-tau_acc)<0.015)
        ax.scatter([fprs[idx]], [tprs[idx]], color="#f1c40f", s=180, zorder=5,
                   label=f"Chosen τ={tau_acc:.2f}")
    except StopIteration:
        pass

    # Annotate every 5th point with tau value
    for i, (fpr, tpr, tau) in enumerate(zip(fprs, tprs, taus)):
        if i % 5 == 0:
            ax.annotate(f"{tau:.2f}", (fpr, tpr),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)

    ax.set_xlabel("False Positive Rate (wrong ACCEPTs)", fontsize=10)
    ax.set_ylabel("True Positive Rate (correct ACCEPTs)", fontsize=10)
    ax.set_title("ROC Curve — WCSA ADG Threshold Sweep", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.yaxis.grid(True); ax.xaxis.grid(True); ax.set_axisbelow(True)
    _save(fig, "10_roc_curve.png")


# ── Plot 11: Accuracy vs bypass tradeoff ─────────────────────────────────────
def plot_tradeoff(roc_pts: list[dict], cal_records: list[CalRecord],
                  tau_acc: float):
    """
    Sweeps tau_acc: as threshold decreases (more aggressive bypass),
    how does sensitivity trade off against bypass rate?
    This is the CORE empirical claim of the paper.
    """
    taus     = sorted(set(p["tau_acc"] for p in roc_pts))
    n_total  = len(cal_records)
    n_correct_total = sum(1 for r in cal_records if r.is_correct_dp)

    acc_vals    = []
    bypass_vals = []

    for tau in taus:
        # Simulate: chains above tau are accepted (bypassing DP)
        # Measure: sensitivity = correct_accepted / n_correct_total
        accepted = [r for r in cal_records if r.cc >= tau]
        correct  = sum(1 for r in accepted if r.is_correct_dp)
        sensitivity = 100.0 * correct / max(n_correct_total, 1)
        bypass      = 100.0 * len(accepted) / max(n_total, 1)
        acc_vals.append(sensitivity)
        bypass_vals.append(bypass)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(taus, acc_vals,    "b-o", lw=2, ms=5, label="Sensitivity (%)")
    ax2.plot(taus, bypass_vals, "r-s", lw=2, ms=5, label="DP Bypass Rate (%)")
    ax1.axvline(tau_acc, color="#f1c40f", linestyle="--", lw=2, label=f"Chosen τ={tau_acc:.2f}")

    ax1.set_xlabel("ADG Accept Threshold (τ_acc)", fontsize=10)
    ax1.set_ylabel("Sensitivity (%)", color="#4a90d9", fontsize=10)
    ax2.set_ylabel("DP Bypass Rate (%)", color="#d9534f", fontsize=10)
    ax1.tick_params(axis='y', labelcolor="#4a90d9")
    ax2.tick_params(axis='y', labelcolor="#d9534f")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="center left", fontsize=9)
    ax1.set_title("WCSA: Sensitivity vs DP Bypass Trade-off",
                  fontsize=11, fontweight="bold")
    ax1.yaxis.grid(True); ax1.set_axisbelow(True)
    _save(fig, "11_accuracy_bypass_tradeoff.png")


# ── Plot 12: WCSA–DP agreement ────────────────────────────────────────────────
def plot_dp_agreement(results):
    """
    For UNCERTAIN chains (passed to DP): what % of WCSA CC-ordering
    agreed with DP's final correct/incorrect decision?
    Proves WCSA is a reliable signal even when it passes chains through.
    """
    gw = results["GenPIP-WCSA"]
    rr = gw.get("read_results", [])

    # For each read where source=="DP" (came through WCSA UNCERTAIN),
    # check if the chain was correct (is_correct) — this represents
    # a case where WCSA said "I am not sure" and DP confirmed/denied.
    dp_reads = [r for r in rr if r.source == "DP"]
    if not dp_reads:
        print("  [Plot 12] No UNCERTAIN→DP reads to plot")
        return

    # Collect CC for UNCERTAIN reads and compare to is_correct
    uncertain_correct   = [r for r in dp_reads if r.is_correct]
    uncertain_incorrect = [r for r in dp_reads if not r.is_correct]

    cc_unc_correct   = [cc for r in uncertain_correct   for cc in r.cc_scores]
    cc_unc_incorrect = [cc for r in uncertain_incorrect for cc in r.cc_scores]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 31)
    if cc_unc_correct:
        ax.hist(cc_unc_correct,   bins=bins, alpha=0.7, color="#2ecc71",
                label=f"UNCERTAIN→DP correct (n={len(cc_unc_correct)})")
    if cc_unc_incorrect:
        ax.hist(cc_unc_incorrect, bins=bins, alpha=0.7, color="#e74c3c",
                label=f"UNCERTAIN→DP wrong (n={len(cc_unc_incorrect)})")
    ax.set_xlabel("WCSA CC score at UNCERTAIN decision"); ax.set_ylabel("Count")
    ax.set_title("CC Distribution of UNCERTAIN Chains (passed to DP)",
                 fontweight="bold")
    ax.legend(fontsize=8); ax.yaxis.grid(True); ax.set_axisbelow(True)
    _save(fig, "12_dp_agreement.png")


def generate_all_plots(results, cal_records, roc_pts, tau_acc):
    print("\n[Plots] Generating all 12 plots...")
    plot_speedup(results)
    plot_energy(results)
    plot_accuracy(results)
    plot_time_breakdown(results)
    plot_speedup_energy(results)
    plot_dp_calls(results)
    plot_waterfall(results)
    plot_stage_breakdown(results)
    plot_score_distribution(results, cal_records, tau_acc)
    plot_roc(roc_pts, tau_acc)
    plot_tradeoff(roc_pts, cal_records, tau_acc)
    plot_dp_agreement(results)
    print(f"[Plots] All 12 plots -> ./{PLOTS_DIR}/\n")
