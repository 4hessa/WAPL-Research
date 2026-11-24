"""Paper figures from audited stage-71 results (light mode, print).

Palette: categorical slots 1-2 of the reference palette, validated with the dataviz
validator (light surface #fcfcfb, all checks pass). Series also differ by marker
shape and are direct-labelled, so identity never relies on colour alone.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('research/paper/figures')
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE = '#2a78d6', '#eb6834'


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis='y', color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc='left', color=INK, fontsize=11, pad=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_xlabel('Fraction of off-target (English) candidates, ρ', color=INK2, fontsize=9)


def load(path):
    s = json.loads(Path(path).read_text())['summary']
    rhos = [float(r) for r in s['per_rho']]
    return s, rhos, [s['per_rho'][str(r)] for r in rhos]


def crossover_figure():
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    for path, colour, marker, label in (('results/llm-gpu-threshold.json', BLUE, 'o', 'L4, r = 1.95 (primary)'),
                                         ('results/llm-gpu-threshold-t4.json', ORANGE, 's', 'T4, r = 3.24 (exploratory)')):
        s, rhos, e = load(path)
        y = [x['minus_random'] for x in e]
        lo = [m - x['ci95'][0] for m, x in zip(y, e)]
        hi = [x['ci95'][1] - m for m, x in zip(y, e)]
        ax.errorbar(rhos, y, yerr=[lo, hi], color=colour, marker=marker, markersize=6, linewidth=2,
                    capsize=3, elinewidth=1, label=label)
        ax.annotate(label.split(',')[0], (rhos[-1], y[-1]), xytext=(6, 0), textcoords='offset points',
                    color=INK, fontsize=9, va='center')
        c = s['primary']['crossover_rho']
        ax.axvline(c, color=colour, linewidth=1, linestyle=':')
    ax.axhline(0, color=INK2, linewidth=1)
    ax.text(0.0, 0.003, 'random better ↑', color=INK2, fontsize=8, va='bottom')
    ax.text(0.0, -0.003, 'alignment better ↓', color=INK2, fontsize=8, va='top')
    style(ax, 'Alignment minus random at equal cost (Qwen2.5-1.5B, stage 71)', 'Δ target loss (nats/token), 95% CI')
    ax.set_xlim(-.03, 1.0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc='lower left')
    fig.tight_layout()
    fig.savefig(OUT / 'stage71_crossover.png', facecolor=SURFACE)


def decomposition_figure():
    _, rhos, e = load('results/llm-gpu-threshold.json')
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    gain = [-x['selection_term'] for x in e]
    cost = [x['throughput_term'] for x in e]
    ax.plot(rhos, gain, color=BLUE, marker='o', markersize=6, linewidth=2, label='Selection gain (−selection term)')
    ax.plot(rhos, cost, color=ORANGE, marker='s', markersize=6, linewidth=2, label='Cost of fewer updates (throughput term)')
    ax.annotate('selection gain', (rhos[-1], gain[-1]), xytext=(6, 0), textcoords='offset points', color=INK, fontsize=9, va='center')
    ax.annotate('cost of fewer updates', (rhos[-1], cost[-1]), xytext=(6, 0), textcoords='offset points', color=INK, fontsize=9, va='center')
    style(ax, 'Why the crossover happens (stage 71, L4)', 'nats/token')
    ax.set_xlim(-.03, 1.12)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'stage71_decomposition.png', facecolor=SURFACE)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    crossover_figure()
    decomposition_figure()
