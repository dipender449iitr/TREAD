# ============================================================
# TREAD MSR | ONE-CELL GOOGLE COLAB FIGURE REPRODUCTION
# Paste ALL of this Python code into ONE Google Colab cell and Run.
# No CSV upload, Drive mount, experiment rerun, or extra cell needed.
# Produces SEVEN separate, high-resolution PNG subfigures.
# ============================================================

from io import StringIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

DPI = 450
DOWNLOAD_ZIP = True          # One download containing seven SEPARATE PNGs.
SHOW_FIGURES = True          # Show each subfigure as it is created.
OUT = Path('TREAD_MSR_Individual_PNGs')
OUT.mkdir(parents=True, exist_ok=True)

# This is the completed run's PROJECT-LEVEL table, embedded in the code.
# ll0/ll1 = mean true-class loss on source-proximate modules;
# full_ll = mean log loss across all target modules.
DATA = '''family,target,K0,K1,U,R0,R1,full_ll,ll0,ll1
aeeem,equinox,4,5,0.0895061728395,0.200031449747,0.650936511958,0.840856790038,0.0672382440224,2.19779973299
aeeem,jdt,2,5,0.368104312939,0.25047605276,0.678671514901,0.517574111549,0.242699313747,1.50206893954
aeeem,lucene,4,5,0.0231548480463,0.209584627107,0.665659141716,0.265746178931,0.0938124698614,2.07669313045
aeeem,mylyn,5,4,0.016111707841,0.216220557451,0.700592210114,0.217049871141,0.0965353920523,1.01293952363
aeeem,pde,3,4,0.0374081496326,0.24191545255,0.663733291191,0.314672115661,0.121933516686,1.53780903382
jira,activemq-5.0.0,1,1,0.0265392781316,0.218860919992,0.680999290344,0.34534857089,0.144414590282,1.51008997216
jira,camel-2.10.0,1,1,0.00859236795552,0.267167595156,0.647675317688,0.124314896118,0.065000337757,2.08886707175
jira,derby-10.2.1.6,2,1,0.214467651554,0.201339759951,0.695903774227,0.58194162705,0.101150656944,1.84408522866
jira,groovy-1_5_7,1,1,0.0581241743725,0.220615744888,0.669507367207,0.124777479549,0.059743316238,2.5119232433
jira,hbase-0.94.0,1,1,0.115203021719,0.216403421681,0.68181699768,0.374397366199,0.211854948545,1.08503959441
jira,hive-0.10.0,1,1,0.075641025641,0.221566805549,0.675388966269,0.257735961948,0.0595137154748,2.06025889698
jira,jruby-1.1,1,1,0.113543091655,0.218772985862,0.673194813131,0.267114982422,0.155853788517,1.37206920767
jira,lucene-2.3.0,1,1,0.0819875776398,0.217163271497,0.676807495487,0.573137037736,0.052598398277,2.35803430544
jira,wicket-1.3.0-incubating-beta-1,1,1,0.061004784689,0.221900830362,0.668789572959,0.365652467546,0.322428934767,0.786473858398
promise,ant-1.7,1,5,0.0469798657718,0.597647243678,0.542392114237,0.758143148705,0.768273029457,0.745440699012
promise,camel-1.6,1,5,0.0290155440415,0.596121064199,0.553037245948,0.723363431404,0.694350531483,0.8617088364
promise,jedit-4.3,3,5,0.0772357723577,0.577396202654,0.495201002678,1.15130934026,1.05568184985,0.853665934031
promise,xalan-2.7,1,4,0.151815181518,0.355496863696,0.626242758421,1.16921358081,0.330889935853,1.27301513504
promise,xerces-1.4,1,5,0.0544217687075,0.442135174866,0.608466694091,0.805852140942,0.521896519352,0.935117388736'''

# RQ3: signed, within-family Spearman rho values, taken from
# ablation_associations.csv from the same completed experiment.
# Values are NOT correlations recomputed over individual modules.
RQ3_DATA = '''class,method,aeeem,jira,promise
0,TREAD,1,-0.05,0.7
0,Source freq.,-0.4,-0.05,0.7
0,Single comp.,-0.4,-0.05,0.7
0,Transfer only,0.5,0.05,0.7
0,Ambiguity only,0.9,-0.1,0.7
1,TREAD,-0.9,-0.1666666666666667,0.9
1,Source freq.,-0.9,-0.1666666666666667,0.9
1,Single comp.,-1,-0.1666666666666667,0.9
1,Transfer only,-0.6,-0.1666666666666667,0.9
1,Ambiguity only,-1,-0.2,0.7'''

project = pd.read_csv(StringIO(DATA))
rq3 = pd.read_csv(StringIO(RQ3_DATA))
FAMILIES = ['aeeem', 'jira', 'promise']
FAMILY_LABELS = ['AEEEM', 'JIRA', 'PROMISE']
COLORS = {'aeeem': '#197A7B', 'jira': '#D48729', 'promise': '#7255A8'}

# Fail early if embedded results are accidentally altered.
assert len(project) == 19 and not project.isna().any().any()
assert len(rq3) == 10
assert int((project.K0.eq(1) | project.K1.eq(1)).sum()) >= 1
assert int(project.K0.eq(1).sum() + project.K1.eq(1).sum()) == 21
jl = project.loc[(project.family == 'jira') & (project.target == 'lucene-2.3.0')]
assert len(jl) == 1 and np.isclose(float(jl.iloc[0].U), 0.0819875776398)
assert len(project[['family', 'target']].drop_duplicates()) == len(project)
# Verify independently derived Spearman ranks agree with embedded RQ3 TREAD row.
for cls in (0, 1):
    for fam in FAMILIES:
        d = project[project.family == fam]
        actual = d[f'R{cls}'].rank().corr(d[f'll{cls}'].rank())
        expected = float(rq3.loc[(rq3['class'] == cls) & (rq3.method == 'TREAD'), fam].iloc[0])
        assert np.isclose(actual, expected, atol=1e-9), (cls, fam, actual, expected)

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
zip_path = Path('TREAD_MSR_Seven_Individual_PNGs.zip')
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
