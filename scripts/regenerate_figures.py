#!/usr/bin/env python3
# Rebuild the seven manuscript RQ figures from archived CSVs (no hardcoded results).
from io import StringIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

BASE=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--results-dir',default=str(BASE/'results/archived_experiment'))
parser.add_argument('--out-dir',default=str(BASE/'verification/regenerated_figures'))
args=parser.parse_args()
RES=Path(args.results_dir)
OUT=Path(args.out_dir)
OUT.mkdir(parents=True,exist_ok=True)
DPI=450
DOWNLOAD_ZIP=False
SHOW_FIGURES=False
raw=pd.read_csv(RES/'project_rq2_analysis.csv')
project=raw.rename(columns={'proximate_logloss_y0':'ll0',
                            'proximate_logloss_y1':'ll1',
                            'full_logloss':'full_ll'}).copy()
assert len(project)==19 and project[['family','target','R0','R1','U','ll0','ll1','full_ll','K0','K1']].notna().all().all()
assert abs(float(project.loc[(project.family=='jira')&(project.target=='lucene-2.3.0'),'U'].iloc[0])-.0819875776398)<1e-11
raw_abl=pd.read_csv(RES/'ablation_associations.csv')
method_map={'TREAD':'TREAD','source_frequency':'Source freq.',
            'single_component':'Single comp.',
            'transfer_only':'Transfer only','ambiguity_only':'Ambiguity only'}
rows=[]
for cls in (0,1):
  for method, display in method_map.items():
    vals={}
    for fam in ('aeeem','jira','promise'):
        d=raw_abl[(raw_abl.type==f'RQ3_y{cls}')&
                  (raw_abl.diagnostic==method)&(raw_abl.family==fam)]
        assert len(d)==1,(cls,method,fam)
        vals[fam]=float(d.iloc[0].spearman_rho)
    rows.append({'class':cls,'method':display,**vals})
rq3=pd.DataFrame(rows)
FAMILIES=['aeeem','jira','promise']
FAMILY_LABELS=['AEEEM','JIRA','PROMISE']
COLORS={'aeeem':'#197A7B','jira':'#D48729','promise':'#7255A8'}
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11.5,
    'axes.titlesize': 12.5,
    'axes.labelsize': 11.6,
    'xtick.labelsize': 10.8,
    'ytick.labelsize': 10.8,
    'legend.fontsize': 10,
    'axes.linewidth': 0.9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'savefig.facecolor': 'white',
})

saved = []

def save_panel(fig, name):
    filepath = OUT / name
    fig.savefig(filepath, dpi=DPI, facecolor='white', bbox_inches='tight', pad_inches=0.13)
    assert filepath.exists() and filepath.stat().st_size > 10_000
    saved.append(filepath)
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    print('Saved:', filepath)

def clean_axes(ax):
    ax.grid(axis='both', linewidth=0.65, alpha=0.16)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6)) if ax.has_data() and ax.get_xlabel() not in ('Benchmark family',) else None

def family_legend(ax, loc='upper right'):
    handles = [Line2D([0], [0], marker='o', linestyle='None', markersize=7.5,
                      markerfacecolor=COLORS[f], markeredgecolor='white',
                      label=lab)
               for f, lab in zip(FAMILIES, FAMILY_LABELS)]
    ax.legend(handles=handles, ncol=3, frameon=False, loc='upper center',
              bbox_to_anchor=(0.5, -0.22), columnspacing=1.3, handletextpad=0.3)

# ------------------------------------------------------------------
# RQ1(a)  Component count > 1
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.45))
x = np.arange(len(FAMILIES)); w = 0.35
for cls, dx, col in [(0, -w/2, '#477B94'), (1, w/2, '#DF7E37')]:
    pct = [100 * project.loc[project.family == f, f'K{cls}'].gt(1).mean() for f in FAMILIES]
    bars = ax.bar(x + dx, pct, width=w, color=col, edgecolor='white', linewidth=0.6,
                  label=f'Class {cls}', zorder=3)
    for bar, val in zip(bars, pct):
        ax.annotate(f'{val:.0f}%', (bar.get_x() + bar.get_width()/2, max(val, 0)),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    fontsize=10, fontweight='semibold')
ax.set_xticks(x, FAMILY_LABELS)
ax.set_ylim(0, 125)
ax.set_ylabel(r'Runs with $K_y>1$ (%)')
ax.set_xlabel('Benchmark family')
ax.set_title('Multi-component representations', pad=11)
ax.legend(ncol=2, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.02))
ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
clean_axes(ax)
fig.tight_layout()
save_panel(fig, 'fig_rq1a_multi_component_representations.png')

# ------------------------------------------------------------------
# RQ1(b)  Source-proximity exceedance, one dot per target project
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.6, 3.5))
for i, fam in enumerate(FAMILIES):
    values = project.loc[project.family == fam].sort_values('target').U.to_numpy()
    xs = i + np.linspace(-0.14, 0.14, len(values))
    ax.scatter(xs, values, s=61, color=COLORS[fam], edgecolor='white', linewidth=0.65, zorder=3)
    ax.hlines(np.median(values), i-0.22, i+0.22, color='#252525', linewidth=2.1, zorder=4)
ax.set_xticks(range(len(FAMILIES)), FAMILY_LABELS)
ax.set_ylim(0, 0.43)
ax.set_ylabel(r'Proximity exceedance $U$')
ax.set_xlabel('Benchmark family')
ax.set_title('Across-project proximity', pad=10)
ax.text(0.98, 0.96, 'Bar: family median', transform=ax.transAxes,
        ha='right', va='top', fontsize=9.3, color='#555555')
clean_axes(ax)
fig.tight_layout()
save_panel(fig, 'fig_rq1b_proximity_variation.png')

# ------------------------------------------------------------------
# RQ2(a), RQ2(b), RQ2(c): separate scatterplots, separate PNGs
# ------------------------------------------------------------------
SCATTER_PANELS = [
    ('R0', 'll0', r'TREAD score $R_0$', 'Class-0 proximate log loss',
     'Class 0', 'fig_rq2a_R0_vs_class0_loss.png'),
    ('R1', 'll1', r'TREAD score $R_1$', 'Class-1 proximate log loss',
     'Class 1', 'fig_rq2b_R1_vs_class1_loss.png'),
    ('U', 'full_ll', r'Proximity exceedance $U$', 'Full-target log loss',
     'Target proximity', 'fig_rq2c_U_vs_full_target_loss.png'),
]
for xcol, ycol, xlabel, ylabel, title, filename in SCATTER_PANELS:
    fig, ax = plt.subplots(figsize=(4.6, 3.65))
    for fam in FAMILIES:
        d = project.loc[project.family == fam]
        ax.scatter(d[xcol], d[ycol], s=66, color=COLORS[fam], alpha=0.93,
                   linewidth=0.65, edgecolors='white', zorder=3)
    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    clean_axes(ax)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    family_legend(ax)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.90, bottom=0.32)
    save_panel(fig, filename)

# ------------------------------------------------------------------
# RQ3(a), RQ3(b): separate family-wise Spearman heatmaps.
# Compact method names preserve readability in IEEE subfigures.
# ------------------------------------------------------------------
methods = ['TREAD', 'Source freq.', 'Single comp.', 'Transfer only', 'Ambiguity only']
for cls, filename in [(0, 'fig_rq3a_class0_component_comparison.png'),
                      (1, 'fig_rq3b_class1_component_comparison.png')]:
    data = rq3[rq3['class'] == cls].set_index('method').loc[methods, FAMILIES]
    values = data.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(4.7, 3.45))
    im = ax.imshow(values, cmap='RdBu_r', vmin=-1, vmax=1,
                   aspect='auto', interpolation='nearest')
    ax.set_title(f'Class {cls}: rank associations', pad=10)
    ax.set_xticks(range(3), FAMILY_LABELS)
    ax.set_yticks(range(len(methods)), methods)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center',
                    color='white' if abs(v) >= 0.6 else '#202020',
                    fontsize=10.8, fontweight='bold' if i == 0 else 'normal')
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.6)
    ax.tick_params(which='minor', bottom=False, left=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.035)
    cb.set_label(r'Spearman $\rho$', fontsize=10)
    cb.ax.tick_params(labelsize=9)
    fig.tight_layout(pad=0.9)
    save_panel(fig, filename)

# ------------------------------------------------------------------
# All individual figures and an optional single ZIP for easy retrieval
# ------------------------------------------------------------------
assert len(saved) == 7 and len({f.name for f in saved}) == 7
zip_path = OUT / 'TREAD_MSR_Seven_Individual_PNGs_Recreated.zip'
with ZipFile(zip_path, 'w', ZIP_DEFLATED) as archive:
    for file in saved:
        archive.write(file, arcname=file.name)

print('\nVERIFIED: seven independent PNG files at', DPI, 'DPI')
for file in saved:
    print(' ', file.name, f'({file.stat().st_size/1024:.0f} KB)')
print('ZIP:', zip_path)

if DOWNLOAD_ZIP:
    try:
        from google.colab import files
        files.download(str(zip_path))
    except ImportError:
        print('Outside Colab: figures and ZIP were saved in the working directory.')
