# 07_visualise_rq1.py
# RQ1: Moffitt Populist Style Markers + Visual Production Features across AfD TikTok Tiers
#
# Moffitt (2016) variables (new prompt — all 1=PRESENT, 0=ABSENT):
#   Appeal_to_the_People  (Dim 1 — people-centric)
#   Anti_Elitism          (Dim 1 — anti-elite)
#   Bad_Manners           (Dim 2 — transgressive style)
#   Crisis_Breakdown_Threat (Dim 3 — crisis framing)
#
# Inputs:
#   results/predictions_gemma3-27b_2026-03-04.csv  — Tier, visual cols (Dress/Setting/Production/Format)
#   results/predictions_gemma3-27b_2026-03-16.csv  — old People_Centrism / Enemy_Construction (fallback)
#   results/predictions_gemma3-27b_<latest>.csv    — new Moffitt 4-var predictions (preferred)
#
# Outputs (results/figures/):
#   rq1_summary_table.{png,pdf}
#   rq1_bar_chart.{png,pdf}
#   rq1_heatmap.{png,pdf}
#   rq1_radar.{png,pdf}
#   rq1_h1a_pairwise.{png,pdf}
#   rq1_h1b_bad_manners.{png,pdf}
#
# NA handling: NaN is preserved throughout; every statistic uses only valid
# (non-missing) observations for that specific variable. Denominators therefore
# vary per variable and are shown as "n/N (pct%)" in all tables.

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.path import Path as MplPath
from pathlib import Path
from itertools import combinations
from scipy.stats import chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 150

TIER_COLORS = {1: "#2C6E9E", 2: "#E08B2E", 3: "#D95F30"}
TIER_LABELS = {
    1: "Tier 1\n(Weidel)",
    2: "Tier 2\n(Bundestag)",
    3: "Tier 3\n(Krah/Siegmund)"
}
TIER_LABELS_SHORT = {1: "T1: Weidel", 2: "T2: Bundestag", 3: "T3: Krah/Siegmund"}

GROUP_A_VARS   = ['Bad_Manners_pred', 'Appeal_to_the_People_pred',
                  'Anti_Elitism_pred', 'Crisis_Breakdown_Threat_pred']
GROUP_A_LABELS = ['Bad Manners', 'Appeal to the People',
                  'Anti-Elitism', 'Crisis/Breakdown/Threat']

GROUP_B_VARS   = ['Dress', 'Setting', 'Production', 'Format']
GROUP_B_LABELS = ['Dress', 'Setting', 'Production', 'Format']

ALL_VARS   = GROUP_A_VARS + GROUP_B_VARS
ALL_LABELS = GROUP_A_LABELS + GROUP_B_LABELS

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(results_dir: Path) -> pd.DataFrame:
    """
    Load predictions for RQ1 analysis.

    Tries to find the NEW Moffitt 4-var predictions first (Appeal_to_the_People_pred,
    Anti_Elitism_pred, Bad_Manners_pred, Crisis_Breakdown_Threat_pred).
    Falls back to old variable names with mapping:
      - Bad_Manners_pred  ← 1 - Manners_pred
      - Appeal_to_the_People_pred ← People_Centrism_pred
      - Anti_Elitism_pred ← NaN (no old equivalent)
      - Crisis_Breakdown_Threat_pred ← Enemy_Construction_pred

    Visual production variables (Dress, Setting, Production, Format) and Tier
    always come from the base predictions file.
    """
    base_path = results_dir / "predictions_gemma3-27b_2026-03-04.csv"

    df = pd.read_csv(base_path)
    df.columns = df.columns.str.strip()

    if 'Unnamed: 14' in df.columns:
        df = df.drop(columns=['Unnamed: 14'])

    # ── Visual columns — preserve NaN ────────────────────────────────────────
    for col in ['Dress', 'Setting', 'Production', 'Format']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── Tier (never missing) ─────────────────────────────────────────────────
    df['Tier'] = df['Tier'].astype(int)

    # ── Coerce old pred columns to numeric (for fallback) ────────────────────
    for col in ['Manners_pred', 'Nativism_pred', 'Econ_Grievance_pred',
                'Status_Grievance_pred']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ── Try to load new Moffitt predictions ──────────────────────────────────
    moffitt_cols = ['Appeal_to_the_People_pred', 'Anti_Elitism_pred',
                    'Bad_Manners_pred', 'Crisis_Breakdown_Threat_pred']

    # Find the newest predictions file that has Moffitt columns
    moffitt_path = None
    for p in sorted(results_dir.glob("predictions_gemma3-27b_*.csv"),
                    key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            cols = pd.read_csv(p, nrows=0).columns.str.strip().tolist()
            if 'Bad_Manners_pred' in cols and 'Appeal_to_the_People_pred' in cols:
                moffitt_path = p
                break
        except Exception:
            continue

    if moffitt_path is not None:
        print(f"  [Moffitt] Using {moffitt_path.name}")
        df_moffitt = pd.read_csv(moffitt_path)
        df_moffitt.columns = df_moffitt.columns.str.strip()
        merge_cols = ['ID'] + [c for c in moffitt_cols if c in df_moffitt.columns]
        df = df.merge(df_moffitt[merge_cols], on='ID', how='left')
        for col in moffitt_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                df[col] = np.nan
    else:
        # ── Fallback: derive from old variable names ─────────────────────────
        print("  [Fallback] No Moffitt predictions found — deriving from old variables")

        # Bad_Manners_pred ← 1 - Manners_pred
        if 'Manners_pred' in df.columns:
            df['Bad_Manners_pred'] = np.where(
                df['Manners_pred'].notna(), 1 - df['Manners_pred'], np.nan)
        else:
            df['Bad_Manners_pred'] = np.nan

        # Appeal_to_the_People_pred ← People_Centrism_pred (from 2026-03-16 file)
        old_path = results_dir / "predictions_gemma3-27b_2026-03-16.csv"
        if old_path.exists():
            df_old = pd.read_csv(old_path)
            df_old.columns = df_old.columns.str.strip()
            if 'People_Centrism_pred' in df_old.columns:
                df = df.merge(df_old[['ID', 'People_Centrism_pred']], on='ID', how='left')
                df['Appeal_to_the_People_pred'] = pd.to_numeric(
                    df['People_Centrism_pred'], errors='coerce')
            else:
                df['Appeal_to_the_People_pred'] = np.nan
            if 'Enemy_Construction_pred' in df_old.columns:
                df = df.merge(df_old[['ID', 'Enemy_Construction_pred']], on='ID', how='left')
                df['Crisis_Breakdown_Threat_pred'] = pd.to_numeric(
                    df['Enemy_Construction_pred'], errors='coerce')
            else:
                df['Crisis_Breakdown_Threat_pred'] = np.nan
        else:
            df['Appeal_to_the_People_pred'] = np.nan
            df['Crisis_Breakdown_Threat_pred'] = np.nan

        # Anti_Elitism_pred — no old equivalent
        df['Anti_Elitism_pred'] = np.nan

    # ── Ensure all expected columns exist ────────────────────────────────────
    for col in moffitt_cols + ['Dress', 'Setting', 'Production', 'Format']:
        if col not in df.columns:
            df[col] = np.nan

    return df


# ── Statistics helpers ────────────────────────────────────────────────────────

def wilson_ci(p: float, n: int, z: float = 1.96):
    """Wilson score 95% CI — handles p=0 and p=1 gracefully."""
    if n == 0:
        return 0.0, 0.0
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    return np.sqrt(chi2 / (n * (min(r, c) - 1))) if (n > 0 and min(r, c) > 1) else 0.0


def compute_stats(df: pd.DataFrame, var: str) -> dict:
    """
    Compute chi-square, Cramér's V, and Bonferroni pairwise Fisher tests for
    one binary variable.  NaN rows are excluded BEFORE any calculation so that
    percentages use the correct (non-missing) denominator.

    Returns a dict with:
      counts    — {tier: {'present': n1, 'absent': n0, 'n': n_valid, 'pct': float}}
      pcts      — {tier: float}  (convenience alias for counts[t]['pct'])
      chi2, p_global, cramers_v, pairwise, n_valid, low_expected
    """
    # ── per-variable NA exclusion ────────────────────────────────────────────
    df_v = df[df[var].notna()].copy()

    if df_v.empty:
        # All observations are missing for this variable
        tiers = sorted(df['Tier'].unique())
        empty_counts = {t: {'present': 0, 'absent': 0, 'n': 0, 'pct': 0.0} for t in tiers}
        return {
            'counts': empty_counts,
            'pcts': {t: 0.0 for t in tiers},
            'chi2': 0.0, 'p_global': 1.0, 'cramers_v': 0.0,
            'pairwise': {pair: 1.0 for pair in combinations(tiers, 2)},
            'n_valid': 0, 'low_expected': True,
        }

    df_v[var] = df_v[var].astype(int)

    tiers = sorted(df_v['Tier'].unique())

    counts = {}
    for t in tiers:
        sub      = df_v[df_v['Tier'] == t][var]
        n_valid  = len(sub)
        n_pres   = int((sub == 1).sum())
        n_abs    = n_valid - n_pres
        pct      = (n_pres / n_valid * 100) if n_valid > 0 else 0.0
        counts[t] = {'present': n_pres, 'absent': n_abs, 'n': n_valid, 'pct': pct}

    n_total = sum(c['n'] for c in counts.values())

    # 2 × num_tiers contingency table (rows: present / absent)
    table = np.array([
        [counts[t]['present'] for t in tiers],
        [counts[t]['absent']  for t in tiers]
    ])

    chi2_stat, p_global, _, expected = chi2_contingency(table, correction=False)
    v = cramers_v(chi2_stat, n_total, 2, len(tiers))

    # Pairwise Fisher with Bonferroni ×3
    pairwise = {}
    for t_a, t_b in combinations(tiers, 2):
        tbl22 = np.array([
            [counts[t_a]['present'], counts[t_a]['absent']],
            [counts[t_b]['present'], counts[t_b]['absent']]
        ])
        _, p_raw = fisher_exact(tbl22)
        pairwise[(t_a, t_b)] = min(1.0, p_raw * 3)

    return {
        'counts':       counts,
        'pcts':         {t: counts[t]['pct'] for t in tiers},
        'chi2':         chi2_stat,
        'p_global':     p_global,
        'cramers_v':    v,
        'pairwise':     pairwise,
        'n_valid':      n_total,
        'low_expected': (expected < 5).any(),
    }


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def fmt_p(p: float) -> str:
    if p < 0.001: return "<.001***"
    if p < 0.01:  return f"{p:.3f}**"
    if p < 0.05:  return f"{p:.3f}*"
    return f"{p:.3f}"


def _pct_valid(df: pd.DataFrame, tier: int, var: str) -> float:
    """Return % present for var in tier, NaN-safe."""
    sub = df[df['Tier'] == tier][var].dropna()
    return (sub == 1).sum() / len(sub) * 100 if len(sub) > 0 else 0.0


def build_stats_table(df: pd.DataFrame) -> list:
    rows = []
    for var, label in zip(ALL_VARS, ALL_LABELS):
        s = compute_stats(df, var)
        rows.append({
            'var':      var,
            'label':    label,
            'stats':    s,
            'counts':   s['counts'],
            'chi2':     s['chi2'],
            'p_global': s['p_global'],
            'cramers_v': s['cramers_v'],
            'p_T1vT2':  s['pairwise'].get((1,2), 1.0),
            'p_T1vT3':  s['pairwise'].get((1,3), 1.0),
            'p_T2vT3':  s['pairwise'].get((2,3), 1.0),
        })
    return rows


# ── Diagnostic helpers ────────────────────────────────────────────────────────

def print_na_diagnostics(df: pd.DataFrame):
    """Print a table of valid N per variable per tier to verify NA exclusion."""
    tiers = sorted(df['Tier'].unique())
    total_n = len(df)
    print(f"  {'Variable':<22} | " + " | ".join(f"T{t} valid" for t in tiers) +
          f" | Total valid | Missing")
    print("  " + "-"*72)
    for var, label in zip(ALL_VARS, ALL_LABELS):
        tier_ns = [df[(df['Tier']==t) & df[var].notna()].shape[0] for t in tiers]
        total_v = sum(tier_ns)
        missing = total_n - total_v
        tier_str = " | ".join(f"{n:^9}" for n in tier_ns)
        print(f"  {label:<22} | {tier_str} | {total_v:^11} | {missing:^7}")
    print()


# ── Figure helpers ────────────────────────────────────────────────────────────

def save_fig(fig, name: str, out_dir: Path):
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches='tight')
    fig.savefig(out_dir / f"{name}.png", bbox_inches='tight')
    plt.close(fig)


# ── Figure 1: Summary Statistics Table ───────────────────────────────────────

def fig_summary_table(df: pd.DataFrame, out_dir: Path):
    stats = build_stats_table(df)

    # Headers: no tier-level N (varies per variable); N shown inside cells
    col_headers = [
        "Variable",
        "T1\nn/N (%)",
        "T2\nn/N (%)",
        "T3\nn/N (%)",
        "chi2",
        "p",
        "V",
        "T1 vs T2\n(adj.)",
        "T1 vs T3\n(adj.)",
        "T2 vs T3\n(adj.)",
    ]

    def cell_pct(counts, tier):
        c = counts.get(tier)
        if c is None:
            return "—"
        return f"{c['present']}/{c['n']} ({c['pct']:.1f}%)"

    cell_rows = []

    # Group A
    for r in stats[:3]:
        cell_rows.append([
            r['label'],
            cell_pct(r['counts'], 1),
            cell_pct(r['counts'], 2),
            cell_pct(r['counts'], 3),
            f"{r['chi2']:.2f}",
            fmt_p(r['p_global']),
            f"{r['cramers_v']:.2f}",
            fmt_p(r['p_T1vT2']),
            fmt_p(r['p_T1vT3']),
            fmt_p(r['p_T2vT3']),
        ])

    cell_rows.append([""] * len(col_headers))   # separator

    # Group B
    for r in stats[3:]:
        cell_rows.append([
            r['label'],
            cell_pct(r['counts'], 1),
            cell_pct(r['counts'], 2),
            cell_pct(r['counts'], 3),
            f"{r['chi2']:.2f}",
            fmt_p(r['p_global']),
            f"{r['cramers_v']:.2f}",
            fmt_p(r['p_T1vT2']),
            fmt_p(r['p_T1vT3']),
            fmt_p(r['p_T2vT3']),
        ])

    fig_h = 1.2 + len(cell_rows) * 0.58
    fig, ax = plt.subplots(figsize=(13.5, fig_h))
    ax.axis('off')

    table = ax.table(
        cellText=cell_rows, colLabels=col_headers,
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.2)

    # Header styling
    for j in range(len(col_headers)):
        c = table[0, j]
        c.set_facecolor('#2C3E50')
        c.set_text_props(color='white', fontweight='bold')

    # Row colours
    group_a_color = '#EAF4FB'
    group_b_color = '#FEF9EC'
    sep_color     = '#BDC3C7'

    for i, row_data in enumerate(cell_rows):
        display_row = i + 1
        is_sep = all(c == "" for c in row_data)
        for j in range(len(col_headers)):
            cell = table[display_row, j]
            if is_sep:
                cell.set_facecolor(sep_color)
                cell.set_height(cell.get_height() * 0.22)
            elif i < 3:
                cell.set_facecolor(group_a_color)
            else:
                cell.set_facecolor(group_b_color)

    # Left-align variable name column
    for i in range(len(cell_rows) + 1):
        table[i, 0].set_text_props(ha='left')

    # Group label annotations
    ax.text(0.01, 0.97, "Populist Style Markers (Moffitt 2016)",
            transform=ax.transAxes, fontsize=9, style='italic',
            color='#2C6E9E', va='top')
    ax.text(0.01, 0.52, "Visual Production Features",
            transform=ax.transAxes, fontsize=9, style='italic',
            color='#2C6E9E', va='top')

    ax.set_title(
        "Table RQ1. Populist Style Markers and Visual Production Features by Tier\n"
        "(n/N = present/valid; % based on valid observations only; N varies per variable due to missing data exclusion;\n"
        "chi2 with correction=False; V = Cramer's V; pairwise p Bonferroni-corrected x3; "
        "* p<0.05, ** p<0.01, *** p<0.001)",
        fontsize=9.5, pad=16
    )

    fig.text(0.5, 0.01,
             "Note: N per cell shows valid observations for that variable/tier combination. "
             "Missing values excluded listwise per variable.",
             ha='center', fontsize=8.5, style='italic', color='#555555')

    save_fig(fig, "rq1_summary_table", out_dir)


# ── Figure 2: Horizontal Grouped Bar Chart ────────────────────────────────────

def fig_bar_chart(df: pd.DataFrame, out_dir: Path):
    stats  = build_stats_table(df)
    tiers  = [1, 2, 3]

    n_vars = len(ALL_VARS)
    bar_h  = 0.22
    y_pos  = np.arange(n_vars)

    # Build valid-N labels for legend (use total valid across all vars as approx)
    tier_n_approx = df.groupby('Tier').size().to_dict()

    fig, ax = plt.subplots(figsize=(9.5, 7.5))

    offsets = [-bar_h, 0, bar_h]
    for tier, offset in zip(tiers, offsets):
        # Use NaN-safe per-variable pct
        vals = [_pct_valid(df, tier, s['var']) for s in stats]
        bars = ax.barh(
            y_pos + offset, vals, bar_h * 0.9,
            color=TIER_COLORS[tier],
            label=f"{TIER_LABELS_SHORT[tier]} (N~{tier_n_approx.get(tier,'?')})",
            alpha=0.88
        )

    # Group separator
    sep_y = n_vars - len(GROUP_B_VARS) - 0.5
    ax.axhline(sep_y, color='#AAAAAA', linewidth=1.0, linestyle='--')
    ax.text(102, sep_y + 0.05, "Visual Production", fontsize=7.5,
            color='#666666', va='bottom')
    ax.text(102, sep_y - 0.1,  "Populist Style",    fontsize=7.5,
            color='#666666', va='top')

    # Left of y-axis: global significance star
    for i, s in enumerate(stats):
        star  = sig_stars(s['p_global'])
        color = '#C0392B' if star != 'n.s.' else '#999999'
        ax.text(-3, y_pos[i], star, ha='right', va='center',
                fontsize=9.5, color=color, fontweight='bold')

    # Right of bars: compact pairwise post-hoc annotations
    for i, s in enumerate(stats):
        pw = s['stats']['pairwise']
        parts = []
        for (ta, tb), p in sorted(pw.items()):
            star = sig_stars(p)
            if star != 'n.s.':
                parts.append(f"T{ta}#T{tb}{star}")
            else:
                parts.append(f"T{ta}=T{tb}")
        annotation = "  ".join(parts)
        ax.text(102, y_pos[i], annotation, ha='left', va='center',
                fontsize=7, color='#444444')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ALL_LABELS, fontsize=10)
    ax.set_xlabel("% of valid videos scoring 1", fontsize=10)
    ax.set_xlim(-8, 135)
    ax.set_ylim(-0.6, n_vars - 0.4)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.grid(axis='x', alpha=0.3, linestyle='--', color='grey')

    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.set_title(
        "Figure RQ1-2. Distribution of Feature Prevalence across Tiers (N=124)\n"
        "(Bars = % valid observations scoring 1; left stars = global chi2 significance;\n"
        "right annotations = pairwise Bonferroni-adj. Fisher; # = significant difference)",
        fontsize=10
    )
    plt.tight_layout()
    save_fig(fig, "rq1_bar_chart", out_dir)


# ── Figure 3: Heatmap Tier × Variable ────────────────────────────────────────

def fig_heatmap(df: pd.DataFrame, out_dir: Path):
    tiers = sorted(df['Tier'].unique())
    # NaN-safe per-variable pct
    matrix = np.array([
        [_pct_valid(df, t, v) for v in ALL_VARS]
        for t in tiers
    ])

    # Build row labels with variable-specific N
    def tier_label_with_n(tier):
        ns = [df[(df['Tier']==tier) & df[v].notna()].shape[0] for v in ALL_VARS]
        n_min = min(ns); n_max = max(ns)
        if n_min == n_max:
            return f"{TIER_LABELS_SHORT[tier]}\n(N={n_min})"
        return f"{TIER_LABELS_SHORT[tier]}\n(N={n_min}-{n_max})"

    fig, ax = plt.subplots(figsize=(10.5, 3.8))

    im = ax.imshow(matrix, cmap='RdYlBu_r', vmin=0, vmax=100, aspect='auto')

    for i in range(len(tiers)):
        for j in range(len(ALL_VARS)):
            val        = matrix[i, j]
            text_color = 'white' if (val > 70 or val < 25) else 'black'
            ax.text(j, i, f"{val:.0f}%", ha='center', va='center',
                    fontsize=10, color=text_color, fontweight='bold')

    row_labels = [tier_label_with_n(t) for t in tiers]
    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xticks(range(len(ALL_VARS)))
    ax.set_xticklabels(ALL_LABELS, fontsize=9, rotation=30, ha='right')

    sep_x = len(GROUP_A_VARS) - 0.5
    ax.axvline(sep_x, color='white', linewidth=2)
    ax.text((len(GROUP_A_VARS)-1)/2, -0.85, "Populist Style Markers",
            ha='center', va='top', fontsize=9, style='italic', color='#2C3E50',
            transform=ax.get_xaxis_transform())
    ax.text(len(GROUP_A_VARS) + (len(GROUP_B_VARS)-1)/2, -0.85, "Visual Production Features",
            ha='center', va='top', fontsize=9, style='italic', color='#2C3E50',
            transform=ax.get_xaxis_transform())

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("% scoring 1", fontsize=8)

    ax.spines[:].set_visible(False)
    ax.set_title(
        "Figure RQ1-3. Heatmap: Feature Prevalence by Tier\n"
        "(Cell values = % valid observations scoring 1; N varies per cell due to missing data exclusion)",
        fontsize=10, pad=30
    )
    plt.tight_layout()
    save_fig(fig, "rq1_heatmap", out_dir)


# ── Figure 4: Radar / Spider Chart ───────────────────────────────────────────

def fig_radar(df: pd.DataFrame, out_dir: Path):
    tiers  = sorted(df['Tier'].unique())
    labels = ALL_LABELS
    num_v  = len(labels)
    angles = np.linspace(0, 2*np.pi, num_v, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))

    for tier in tiers:
        # NaN-safe per-variable pct
        vals = [_pct_valid(df, tier, v) for v in ALL_VARS]
        vals += vals[:1]
        ax.plot(angles, vals, color=TIER_COLORS[tier], linewidth=2, alpha=0.9,
                label=TIER_LABELS_SHORT[tier])
        ax.fill(angles, vals, color=TIER_COLORS[tier], alpha=0.18)

    sep_angle = angles[len(GROUP_A_VARS)]
    ax.axvline(sep_angle, color='#AAAAAA', linewidth=0.8, linestyle='--', alpha=0.6)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_rlabel_position(0)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color='grey', size=7)
    ax.set_ylim(0, 100)
    ax.grid(True, color='lightgrey', linestyle='--', alpha=0.5)

    mid_a = np.mean(angles[:len(GROUP_A_VARS)])
    mid_b = np.mean(angles[len(GROUP_A_VARS):len(ALL_VARS)])
    ax.annotate("Populist Style", xy=(mid_a, 112), ha='center', fontsize=8,
                style='italic', color='#2C3E50', xycoords=('data', 'data'),
                annotation_clip=False)
    ax.annotate("Visual Prod.", xy=(mid_b, 112), ha='center', fontsize=8,
                style='italic', color='#2C3E50', xycoords=('data', 'data'),
                annotation_clip=False)

    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title(
        "Figure RQ1-4. Feature Profiles by Tier (Radar Chart)\n"
        "(Axes = % valid observations scoring 1; missing data excluded per variable)",
        pad=22, fontsize=10
    )
    plt.tight_layout()
    save_fig(fig, "rq1_radar", out_dir)


# ── Figure 5: H1a — Visual Features Pairwise Heatmap ─────────────────────────

def fig_h1a_pairwise(df: pd.DataFrame, out_dir: Path):
    """
    H1a: T1 ≈ T2 ≠ T3 for visual production features.
    Pairwise Fisher exact p-values (Bonferroni ×3) as a heatmap.
    Includes a summary row counting variables that match the H1a pattern.
    """
    pairs      = [(1,2), (1,3), (2,3)]
    pair_labels = ["T1 vs T2", "T1 vs T3", "T2 vs T3"]
    p_matrix   = np.ones((len(GROUP_B_VARS), len(pairs)))
    all_stats  = []

    for vi, var in enumerate(GROUP_B_VARS):
        s = compute_stats(df, var)
        all_stats.append(s)
        for pi, pair in enumerate(pairs):
            p_matrix[vi, pi] = s['pairwise'].get(pair, 1.0)

    # H1a pattern: T1 vs T2 n.s. AND T1 vs T3 sig AND T2 vs T3 sig
    h1a_match = [
        (p_matrix[vi, 0] >= 0.05 and
         p_matrix[vi, 1] <  0.05 and
         p_matrix[vi, 2] <  0.05)
        for vi in range(len(GROUP_B_VARS))
    ]
    n_match = sum(h1a_match)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    ax.imshow(np.ones_like(p_matrix), cmap='Greys', vmin=0, vmax=3,
              aspect='auto', alpha=0.12)
    sig_im = ax.imshow(
        np.where(p_matrix < 0.05, p_matrix, np.nan),
        cmap='YlGn_r', vmin=0, vmax=0.05, aspect='auto', alpha=0.85
    )

    for vi in range(len(GROUP_B_VARS)):
        for pi in range(len(pairs)):
            p    = p_matrix[vi, pi]
            star = sig_stars(p)
            text  = f"p={p:.3f}\n{star}"
            color = 'white' if (p < 0.05 and p < 0.02) else 'black'
            ax.text(pi, vi, text, ha='center', va='center', fontsize=9, color=color)

        # Mark H1a-matching rows
        if h1a_match[vi]:
            ax.annotate("H1a", xy=(2.55, vi), ha='left', va='center',
                        fontsize=8, color='#27AE60', fontweight='bold',
                        annotation_clip=False)

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pair_labels, fontsize=10)
    ax.set_yticks(range(len(GROUP_B_VARS)))
    ax.set_yticklabels(GROUP_B_LABELS, fontsize=10)
    ax.spines[:].set_visible(False)

    # H1a prediction annotation
    ax.text(1, -0.82,
            "H1a predicts: T1 vs T2 n.s.  |  T1 vs T3 sig  |  T2 vs T3 sig",
            ha='center', va='top', fontsize=8, style='italic', color='#555555',
            transform=ax.get_xaxis_transform())

    # Summary: how many variables show H1a pattern
    ax.text(1, -1.25,
            f"Variables matching H1a pattern: {n_match}/{len(GROUP_B_VARS)}",
            ha='center', va='top', fontsize=9, fontweight='bold',
            color='#27AE60' if n_match >= 3 else '#C0392B',
            transform=ax.get_xaxis_transform())

    cbar = fig.colorbar(sig_im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("adj. p (significant cells only)", fontsize=8)

    ax.set_title(
        "Figure RQ1-5 (H1a). Pairwise Fisher Tests — Visual Production Features\n"
        "(Bonferroni-corrected p-values ×3; green = p<0.05; grey = n.s.;\n"
        "H1a = T1 approx T2 ≠ T3; missing data excluded per variable)",
        fontsize=10
    )
    plt.tight_layout()
    save_fig(fig, "rq1_h1a_pairwise", out_dir)


# ── Figure 6: H1b — Bad Manners Bar Chart with CI + Pairwise Brackets ─────────

def fig_h1b_bad_manners(df: pd.DataFrame, out_dir: Path):
    """
    H1b: Bad_Manners increases T1 → T2 → T3.
    Bar chart per tier with Wilson 95% CI (NaN-safe N) + pairwise brackets.
    """
    tiers = sorted(df['Tier'].unique())

    means  = []
    lo_err = []
    hi_err = []
    valid_ns = []

    for t in tiers:
        # NaN-safe: dropna before computing
        sub = df[df['Tier']==t]['Bad_Manners_pred'].dropna()
        n   = len(sub)
        p   = sub.mean() if n > 0 else 0.0
        lo, hi = wilson_ci(p, n)
        means.append(p * 100)
        lo_err.append((p - lo) * 100)
        hi_err.append((hi - p) * 100)
        valid_ns.append(n)

    fig, ax = plt.subplots(figsize=(6, 5.5))

    x    = np.arange(len(tiers))
    bars = ax.bar(
        x, means,
        yerr=[lo_err, hi_err],
        capsize=5, width=0.5,
        color=[TIER_COLORS[t] for t in tiers],
        alpha=0.85,
        error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': '#333333'}
    )

    # Value + n/N labels on bars
    for i, (bar, m, n) in enumerate(zip(bars, means, valid_ns)):
        ax.text(bar.get_x() + bar.get_width()/2,
                m + hi_err[i] + 1.5,
                f"{m:.0f}%\n(n={int(round(m/100*n))}/{n})",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Pairwise brackets — use corrected N from compute_stats
    s     = compute_stats(df, 'Bad_Manners_pred')
    y_max = max(m + e for m, e in zip(means, hi_err))

    drawn_pairs = [
        ((1,2), s['pairwise'].get((1,2), 1.0), y_max + 9),
        ((1,3), s['pairwise'].get((1,3), 1.0), y_max + 19),
        ((2,3), s['pairwise'].get((2,3), 1.0), y_max + 9),
    ]

    for (t_a, t_b), p_val, y_br in drawn_pairs:
        ia   = tiers.index(t_a)
        ib   = tiers.index(t_b)
        star = sig_stars(p_val)
        ax.plot([ia, ia, ib, ib],
                [means[ia]+hi_err[ia]+1, y_br, y_br, means[ib]+hi_err[ib]+1],
                lw=1.2, color='#333333')
        p_str = f"p={p_val:.3f} {star}"
        ax.text((ia+ib)/2, y_br + 0.5, p_str,
                ha='center', va='bottom', fontsize=8,
                color='#C0392B' if star != 'n.s.' else '#888888')

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{TIER_LABELS[t]}\n(N_valid={valid_ns[i]})" for i, t in enumerate(tiers)],
        fontsize=9
    )
    ax.set_ylabel("% Bad Manners present", fontsize=10)
    ax.set_ylim(0, y_max + 32)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    ax.annotate("", xy=(2.3, max(means)*0.45),
                xytext=(-.3, max(means)*0.45),
                arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=1.5))
    ax.text(1.0, max(means)*0.45 + 2,
            "H1b: expects T1 < T2 < T3", ha='center', fontsize=8,
            style='italic', color='#888888')

    ax.set_title(
        "Figure RQ1-6 (H1b). Bad Manners Prevalence by Tier\n"
        "(Bars = %; n/N = present/valid; error bars = Wilson 95% CI;\n"
        "brackets = Bonferroni pairwise Fisher p; N per tier shows valid observations)",
        fontsize=10
    )
    plt.tight_layout()
    save_fig(fig, "rq1_h1b_bad_manners", out_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    RESULTS_DIR = Path("results")
    FIGURES_DIR = RESULTS_DIR / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df     = load_data(RESULTS_DIR)
    tier_n = df.groupby('Tier').size().to_dict()
    print(f"  N={len(df)} | Tier distribution: {tier_n}")
    print(f"  Columns: {[c for c in df.columns if not c.endswith('_reasoning')]}\n")

    # ── Diagnostic: valid N per variable per tier ─────────────────────────────
    print("NA exclusion diagnostics:")
    print_na_diagnostics(df)

    # ── Cross-check against reference values ─────────────────────────────────
    print("Cross-check (reference values in parentheses):")
    for var, label in zip(ALL_VARS, ALL_LABELS):
        s = compute_stats(df, var)
        tiers = sorted(df['Tier'].unique())
        cells = [f"T{t}: {s['counts'].get(t,{}).get('present','?')}/{s['counts'].get(t,{}).get('n','?')} "
                 f"({s['counts'].get(t,{}).get('pct',0):.1f}%)" for t in tiers]
        print(f"  {label:<22}: {' | '.join(cells)} | chi2={s['chi2']:.2f}, p={s['p_global']:.3f}, V={s['cramers_v']:.3f}")
    print()

    figures = [
        ("rq1_summary_table",   lambda: fig_summary_table(df, FIGURES_DIR)),
        ("rq1_bar_chart",       lambda: fig_bar_chart(df, FIGURES_DIR)),
        ("rq1_heatmap",         lambda: fig_heatmap(df, FIGURES_DIR)),
        ("rq1_radar",           lambda: fig_radar(df, FIGURES_DIR)),
        ("rq1_h1a_pairwise",    lambda: fig_h1a_pairwise(df, FIGURES_DIR)),
        ("rq1_h1b_bad_manners", lambda: fig_h1b_bad_manners(df, FIGURES_DIR)),
    ]

    for name, fn in figures:
        try:
            print(f"[GEN]  {name}...")
            fn()
            print(f"[OK]   {name}")
        except Exception as e:
            import traceback
            print(f"[WARN] {name} failed: {e}")
            traceback.print_exc()

    print(f"\nAll done. Figures in: {FIGURES_DIR.resolve()}")
