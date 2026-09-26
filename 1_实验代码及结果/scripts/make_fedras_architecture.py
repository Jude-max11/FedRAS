"""FedRAS system-architecture figure, v6 (server boxes with mini-diagrams).

Backend: Python (matplotlib), per the nature-figure backend gate (user-selected).
Single-panel schematic-led composite, 190 mm Elsevier double-column width.

v6 design contract (author request: the four server pipeline boxes should
carry small explanatory illustrations, not text alone):
- strict modular grid kept: four equal boxes on one row, equal client cards,
  all connectors orthogonal;
- each server box gains a mini-diagram in its lower half:
  A  Link states          - base-station mast + fluctuating rate sparkline;
  B  Prediction           - two mini-curves: on-time probability decays with
                             rank, capacity rises and saturates (the trade-off
                             the scheduler arbitrates);
  C  Scheduler (hero)     - byte-budget bar packed with rank blocks
                             [16][8][4][4] under a B_max bracket, admission
                             heads with "<= K";
  D  Aggregation          - per-client rank stacks -> averaged B-bar, with a
                             hatched frozen-A strip carrying a padlock.
- pictographic icons kept from v5 (rack, phone/drone/satellite, padlocks,
  clock); notation and every number match CN_reviewed.tex Section 3
  (eq. 3 hazard, eq. 5 capacity, 1,408 params/rank column = 5.63 KB,
  r16 0.090 MB vs 5.29 MB full model = 58.7x; Propositions 1 and 3 in caption).
Glyph floor: mathtext with sub/superscripts uses parents >= 7.4 pt (scripts
>= 5.0 pt); diagram captions avoid subscripts entirely.
Layout safety: every text artist is measured against its host region after
the final draw (check_fits) and the script fails loudly on overflow.

QA: render-time alignment gate (single panel -> NOT APPLICABLE recorded),
then validate_figure.py / audit_pdf_text.py / audit_figure_collisions.py.
"""
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

SKILL_SCRIPTS = os.environ.get(
    'NATURE_FIGURE_SCRIPTS', os.path.expanduser('~/.codex/skills/nature-figure/scripts'))
if os.path.isdir(SKILL_SCRIPTS):
    sys.path.insert(0, SKILL_SCRIPTS)
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(ROOT), '3_ComputerNetworks期刊latex版本',
                   'CN_package_v4', 'recolored_figures')
os.makedirs(OUT, exist_ok=True)
NAME = os.path.join(OUT, 'fig_architecture')

MM = 1 / 25.4
W, H = 190.0, 122.0

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
    'font.size': 7,
})

GRAY = '#555555'
DARK = '#333333'
NEUTRAL_EDGE = '#8a8a8a'
NEUTRAL_FILL = '#ffffff'
BAND_FILL = '#f5f5f5'
BLUE = '#1f77b4'
RED = '#d62728'
RED_FILL = '#fdf1ef'
GREEN = '#2ca02c'
CHIP_GRAY_FILL = '#e9e9e9'
BLUE_CHIP_FILL = '#eaf1f8'
BLUE_CHIP_TEXT = '#2a4d6e'
MATH = 7.6    # mathtext parent with scripts (scripts >= 5.13 pt)
BMATH = 7.4   # mathtext parent for the one diagram label with a subscript

PHAT = 'P\u0302'
BTIL = 'B\u0303'
RBAR = 'R\u0304'
BBAR = 'B\u0304'
ABAR = 'A\u0304'

fig = plt.figure(figsize=(W * MM, H * MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect('equal')
ax.axis('off')

FITS = []


def box(x0, y0, x1, y1, fc=NEUTRAL_FILL, ec=NEUTRAL_EDGE, lw=0.9, rounding=1.4, z=2):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle=f'round,pad=0,rounding_size={rounding}',
                       fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(x0, y0, x1, y1, color=GRAY, lw=1.1, ls='-', ms=7, z=3):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                        mutation_scale=ms, color=color, lw=lw, ls=ls,
                        shrinkA=0, shrinkB=0, zorder=z)
    ax.add_patch(a)


def line(x0, y0, x1, y1, color=GRAY, lw=1.0, ls='-', z=1):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle='round')


def text(x, y, s, name, host=None, **kw):
    t = ax.text(x, y, s, **kw)
    if host is not None:
        FITS.append((t, *host, name))
    return t


def check_fits():
    fig.canvas.draw()
    inv = ax.transData.inverted()
    bad = []
    for t, x0, x1, y0, y1, name in FITS:
        bb = t.get_window_extent()
        (dx0, dy0), (dx1, dy1) = inv.transform(bb)
        if dx0 < x0 or dx1 > x1:
            bad.append(f'{name}: x [{dx0:.1f},{dx1:.1f}] outside [{x0:.1f},{x1:.1f}]')
        if dy0 < y0 or dy1 > y1:
            bad.append(f'{name}: y [{dy0:.1f},{dy1:.1f}] outside [{y0:.1f},{y1:.1f}]')
    if bad:
        raise SystemExit('TEXT OVERFLOW:\n  ' + '\n  '.join(bad))
    print(f'fit-check OK for {len(FITS)} constrained texts')


# ------------------------------------------------------------ vector pictograms
def icon_server(cx, cy):
    w, h = 7.0, 5.6
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc='#ffffff',
                           ec=DARK, lw=0.8, zorder=4))
    for k in (1, 2):
        ax.plot([cx - w / 2 + 0.7, cx + w / 2 - 0.7],
                [cy - h / 2 + k * h / 3, cy - h / 2 + k * h / 3],
                color=DARK, lw=0.55, zorder=4)
    ax.add_patch(Circle((cx + w / 2 - 1.5, cy + h / 2 - 1.0), 0.38,
                        fc=GREEN, ec='none', zorder=5))
    ax.add_patch(Circle((cx - w / 2 + 1.5, cy + h / 2 - 1.0), 0.38,
                        fc='#bbbbbb', ec='none', zorder=5))


def icon_phone(cx, cy, s=4.6):
    w, h = 0.60 * s, s
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0,rounding_size=0.5',
                                fc='#ffffff', ec=DARK, lw=0.7, zorder=4))
    ax.add_patch(Rectangle((cx - w / 2 + 0.45, cy - h / 2 + 0.95),
                           w - 0.9, h - 1.9, fc=BLUE_CHIP_FILL,
                           ec='none', zorder=5))
    ax.add_patch(Circle((cx, cy - h / 2 + 0.55), 0.22, fc=DARK, ec='none', zorder=5))


def icon_drone(cx, cy, s=4.6):
    ax.plot([cx - 0.42 * s, cx + 0.42 * s], [cy - 0.24 * s, cy + 0.24 * s],
            color=DARK, lw=0.7, zorder=4)
    ax.plot([cx - 0.42 * s, cx + 0.42 * s], [cy + 0.24 * s, cy - 0.24 * s],
            color=DARK, lw=0.7, zorder=4)
    ax.add_patch(Rectangle((cx - 0.30 * s, cy - 0.17 * s), 0.60 * s, 0.34 * s,
                           fc='#ffffff', ec=DARK, lw=0.7, zorder=5))
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.add_patch(Circle((cx + sx * 0.44 * s, cy + sy * 0.26 * s),
                                0.13 * s, fc='none', ec=DARK, lw=0.6, zorder=4))


def icon_sat(cx, cy, s=4.6):
    ax.add_patch(Rectangle((cx - 0.17 * s, cy - 0.26 * s), 0.34 * s, 0.52 * s,
                           fc='#ffffff', ec=DARK, lw=0.7, zorder=5))
    for sx in (-1, 1):
        ax.add_patch(Rectangle((cx + sx * 0.25 * s - (0 if sx > 0 else 0.40 * s),
                                cy - 0.16 * s), 0.40 * s, 0.32 * s,
                               fc=BLUE_CHIP_FILL, ec=DARK, lw=0.6, zorder=4))
    ax.plot([cx, cx], [cy + 0.26 * s, cy + 0.44 * s], color=DARK, lw=0.7, zorder=4)
    ax.add_patch(Circle((cx, cy + 0.47 * s), 0.09 * s, fc=DARK, ec='none', zorder=5))


def icon_padlock(cx, cy, s=2.5, color=GRAY):
    ax.add_patch(Rectangle((cx - 0.38 * s, cy - 0.55 * s), 0.76 * s, 0.60 * s,
                           fc='#ffffff', ec=color, lw=0.6, zorder=5))
    th = np.linspace(0, np.pi, 14)
    ax.plot(cx + 0.26 * s * np.cos(th),
            cy - 0.55 * s + 0.50 * s + 0.30 * s * np.sin(th),
            color=color, lw=0.6, zorder=5)


def icon_clock(cx, cy, r=2.4):
    ax.add_patch(Circle((cx, cy), r, fc='#ffffff', ec=DARK, lw=0.8, zorder=5))
    for ang in (90, 0, 270, 180):
        a = np.deg2rad(ang)
        ax.plot([cx + 0.78 * r * np.cos(a), cx + 0.95 * r * np.cos(a)],
                [cy + 0.78 * r * np.sin(a), cy + 0.95 * r * np.sin(a)],
                color=DARK, lw=0.55, zorder=5)
    ax.plot([cx, cx], [cy, cy + 0.55 * r], color=DARK, lw=0.7, zorder=5)
    ax.plot([cx, cx + 0.42 * r], [cy, cy + 0.10 * r], color=RED, lw=0.7, zorder=5)
    ax.add_patch(Circle((cx, cy), 0.16, fc=DARK, ec='none', zorder=6))


def icon_tower(cx, ybase, h=8.0):
    """Base-station mast with emitter arcs."""
    ax.plot([cx - 1.7, cx, cx + 1.7], [ybase, ybase + h - 1.6, ybase],
            color=DARK, lw=0.8, zorder=4)
    ax.plot([cx - 1.05, cx + 1.05], [ybase + h * 0.36, ybase + h * 0.36],
            color=DARK, lw=0.6, zorder=4)
    ax.add_patch(Circle((cx, ybase + h - 1.2), 0.32, fc=BLUE, ec='none', zorder=5))
    for r in (1.5, 2.5):
        th = np.linspace(np.pi * 0.22, np.pi * 0.78, 14)
        ax.plot(cx + r * np.cos(th), ybase + h - 1.2 + r * np.sin(th),
                color=BLUE, lw=0.65, zorder=4)


def icon_person(cx, cy, r=0.85, fc=RED):
    ax.add_patch(Circle((cx, cy + 0.55 * r), r * 0.62, fc=fc, ec='none', zorder=5))
    ax.add_patch(FancyBboxPatch((cx - r, cy - r * 0.9), 2 * r, r * 0.95,
                                boxstyle=f'round,pad=0,rounding_size={r * 0.45}',
                                fc=fc, ec='none', zorder=5))


# ---------------------------------------------------------------- server band
box(3, 76, 187, 122, fc=BAND_FILL, ec=NEUTRAL_EDGE, lw=1.0, rounding=2.0)
icon_server(10.3, 119.1)
ax.text(15.5, 119.2, 'Edge aggregator — FedRAS server', fontsize=7.2,
        fontweight='bold', color=DARK, va='center')

BX = [(6, 42), (47, 97), (102, 142), (147, 184)]
BY0, BY1 = 78, 118
for x0, x1 in BX:
    box(x0, BY0, x1, BY1)
box(BX[2][0], BY0, BX[2][1], BY1, fc=RED_FILL, ec=RED, lw=1.4)

AX0, AX1 = BX[0]
BX0, BX1 = BX[1]
CX0, CX1 = BX[2]
DX0, DX1 = BX[3]
DIA_TOP = 96.5          # diagrams live between DIA_BOT and DIA_TOP
DIA_BOT = 79.5

# --- Box A: link states ------------------------------------------------------
text((AX0 + AX1) / 2, 114.8, 'Link states', 'A.title',
     (AX0 + 1, AX1 - 1, 113.0, 116.6), fontsize=7, fontweight='bold',
     ha='center', va='center')
text((AX0 + AX1) / 2, 108.3, f'uplink rate ${RBAR}_i$,\nmobility state $\\mu_i$',
     'A.body', (AX0 + 1, AX1 - 1, 103.8, 113.1), fontsize=MATH, ha='center',
     va='center', color=DARK, linespacing=1.5)
text((AX0 + AX1) / 2, 100.6, '(estimated each round)', 'A.note',
     (AX0 + 1, AX1 - 1, 99.2, 102.0), fontsize=6.0, ha='center', va='center',
     color='#666666')
# mini-diagram: tower + arcs + fluctuating rate sparkline
icon_tower(AX0 + 5.5, DIA_BOT + 0.5, h=9.0)
chx0, chx1 = AX0 + 12.5, AX1 - 1.5
chy0, chy1 = DIA_BOT + 3.5, DIA_TOP - 2.0
ax.plot([chx0, chx0], [chy0, chy1], color=GRAY, lw=0.6, zorder=4)
ax.plot([chx0, chx1], [chy0, chy0], color=GRAY, lw=0.6, zorder=4)
xs = np.linspace(chx0 + 1.0, chx1 - 1.0, 12)
pat = np.array([1, -0.7, 0.9, -1, 0.6, -0.85, 1, -0.5, 0.8, -0.95, 0.7, -0.75])
ys = chy0 + 1.4 + (chy1 - chy0 - 2.8) * (0.5 + 0.40 * pat)
ax.plot(xs, ys, color=BLUE, lw=0.9, zorder=5)
ax.plot(xs, ys, 'o', ms=1.2, color=BLUE, zorder=5)
text((chx0 + chx1) / 2, chy0 - 2.0, f'${RBAR}_i$ over rounds', 'A.dia',
     (chx0, chx1, 79.0, chy0 - 0.2), fontsize=BMATH, ha='center',
     va='center', color='#666666')

# --- Box B: per-client-rank prediction --------------------------------------
text((BX0 + BX1) / 2, 114.8, 'Per–client–rank prediction', 'B.title',
     (BX0 + 1, BX1 - 1, 113.0, 116.6), fontsize=7, fontweight='bold',
     ha='center', va='center')
text(BX0 + 2.2, 111.4, 'delivery hazard  (eq. 3)', 'B.lead1',
     (BX0 + 1, BX1 - 1, 110.1, 112.7), fontsize=6.0, ha='left', va='center',
     color='#666666')
text((BX0 + BX1) / 2, 107.9,
     f'${PHAT}_{{\\mathrm{{on}},i}}(r) = \\exp(-\\mu_i - 8B_i(r)/({RBAR}_i \\tau))$',
     'B.formula1', (BX0 + 1, BX1 - 1, 105.7, 110.1), fontsize=MATH,
     ha='center', va='center')
text(BX0 + 2.2, 103.5, 'saturating capacity  (eq. 5)', 'B.lead2',
     (BX0 + 1, BX1 - 1, 102.2, 104.8), fontsize=6.0, ha='left', va='center',
     color='#666666')
text((BX0 + BX1) / 2, 100.0,
     '$G_i(r) = \\pi_i\\sqrt{\\min(r,\\, r_{\\max})/r_{\\max}}$',
     'B.formula2', (BX0 + 1, BX1 - 1, 97.6, 102.4), fontsize=MATH,
     ha='center', va='center')
# mini-diagram: the trade-off — on-time decays with rank, capacity saturates
p1x0, p1x1 = BX0 + 2.5, BX0 + 22.5
p2x0, p2x1 = BX1 - 22.5, BX1 - 2.5
py0, py1 = DIA_BOT + 3.0, DIA_TOP - 0.5
for px0, px1 in ((p1x0, p1x1), (p2x0, p2x1)):
    ax.plot([px0, px0], [py0, py1], color=GRAY, lw=0.6, zorder=4)
    ax.plot([px0, px1], [py0, py0], color=GRAY, lw=0.6, zorder=4)
rr = np.linspace(0, 1, 40)
# panel 1: hazard exp(-c r)
y1 = py0 + 1.0 + (py1 - py0 - 2.0) * np.exp(-2.6 * rr)
ax.plot(p1x0 + 1.0 + rr * (p1x1 - p1x0 - 1.5), y1, color=RED, lw=1.0, zorder=5)
# panel 2: capacity sqrt saturating
y2 = py0 + 1.0 + (py1 - py0 - 2.0) * np.sqrt(rr)
ax.plot(p2x0 + 1.0 + rr * (p2x1 - p2x0 - 1.5), y2, color=GREEN, lw=1.0, zorder=5)
text((p1x0 + p1x1) / 2, DIA_BOT + 1.4, f'{PHAT} ↓ vs rank', 'B.dia1',
     (p1x0, p1x1, DIA_BOT + 0.2, py0 - 0.2), fontsize=5.7, ha='center',
     va='center', color='#666666')
text((p2x0 + p2x1) / 2, DIA_BOT + 1.4, 'G ↑ (saturating)', 'B.dia2',
     (p2x0, p2x1, DIA_BOT + 0.2, py0 - 0.2), fontsize=5.7, ha='center',
     va='center', color='#666666')

# --- Box C: deadline-aware scheduler (hero) ---------------------------------
text((CX0 + CX1) / 2, 114.8, 'Deadline-aware scheduler', 'C.title',
     (CX0 + 1, CX1 - 1, 113.0, 116.6), fontsize=6.6, fontweight='bold',
     ha='center', va='center', color=RED)
text((CX0 + CX1) / 2, 110.8,
     '$\\max\\ \\sum_i G_i(r_i)\\,%s_{\\mathrm{on},i}(r_i)$' % PHAT,
     'C.obj', (CX0 + 1, CX1 - 1, 107.4, 114.2), fontsize=MATH, ha='center',
     va='center')
text((CX0 + CX1) / 2, 104.0,
     f'$\\mathrm{{s.t.}}\\ \\sum_i {BTIL}_i(r_i) \\leq {BTIL}_{{\\max}}$',
     'C.s.t.', (CX0 + 1, CX1 - 1, 100.6, 107.4), fontsize=MATH, ha='center',
     va='center')
text((CX0 + CX1) / 2, 99.2, 'output: client i → rank r', 'C.out',
     (CX0 + 1, CX1 - 1, 97.7, 100.7), fontsize=7.0, ha='center', va='center',
     color=RED)
# mini-diagram: byte-budget bar packed with rank blocks + admission heads
bx0, bx1 = CX0 + 2.0, CX0 + 34.0
by0, by1 = 86.0, 89.0
edges = [bx0, bx0 + 13.5, bx0 + 20.3, bx0 + 23.7, bx0 + 27.1, bx1]
labels = ['16', '8', '4', '4']
fills = [RED, '#e06666', '#ea9999', '#ea9999']
for k in range(4):
    ax.add_patch(Rectangle((edges[k], by0), edges[k + 1] - edges[k], by1 - by0,
                           fc=fills[k], ec='#ffffff', lw=0.6, zorder=5))
    text((edges[k] + edges[k + 1]) / 2, (by0 + by1) / 2, labels[k],
         f'C.bar{k}', (edges[k] + 0.2, edges[k + 1] - 0.2, by0 + 0.1, by1 - 0.1),
         fontsize=5.5, ha='center', va='center', color='#ffffff')
ax.add_patch(Rectangle((edges[4], by0), bx1 - edges[4], by1 - by0,
                       fc='none', ec='#bbbbbb', lw=0.6, zorder=5))
for k in range(5):
    ax.plot([edges[k]] * 2, [by0 - 0.9, by1 + 0.9], color='#bbbbbb', lw=0.5,
            ls=(0, (1.5, 1.2)), zorder=5)
heads = [(edges[k] + edges[k + 1]) / 2 for k in range(4)]
for hx in heads:
    icon_person(hx, 92.3, r=0.85)
text(bx1 - 0.4, 92.3, '$\\leq K$', 'C.heads', (130.6, 137.7, 90.8, 93.8),
     fontsize=5.7, ha='right', va='center', color=DARK)
# B_max bracket under the bar
yb = 83.2
ax.plot([bx0, bx0], [yb, yb + 0.8], color=GRAY, lw=0.6, zorder=4)
ax.plot([bx1, bx1], [yb, yb + 0.8], color=GRAY, lw=0.6, zorder=4)
ax.plot([bx0, bx1], [yb, yb], color=GRAY, lw=0.6, zorder=4)
text((bx0 + bx1) / 2, yb - 2.5, f'$\\mathrm{{byte\\ budget}}\\ {BTIL}_{{\\max}}$',
     'C.bracket', (bx0, bx1, 78.3, 83.0), fontsize=BMATH, ha='center',
     va='center', color='#666666')

# --- Box D: rank-masked aggregation -----------------------------------------
text((DX0 + DX1) / 2, 113.7, 'Rank-masked\naggregation', 'D.title',
     (DX0 + 1, DX1 - 1, 110.3, 117.1), fontsize=6.5, fontweight='bold',
     ha='center', va='center', linespacing=1.35)
text((DX0 + DX1) / 2, 105.4, f'${BBAR}_{{:,j}}$ averaged\nover $r_i \\geq j$',
     'D.body', (DX0 + 1, DX1 - 1, 101.3, 109.5), fontsize=MATH, ha='center',
     va='center', color=DARK, linespacing=1.3)
text((DX0 + DX1) / 2, 99.6, f'${ABAR} = A$ frozen', 'D.frozen',
     (DX0 + 1, DX1 - 1, 97.5, 101.7), fontsize=MATH, ha='center', va='center',
     color=DARK)
# mini-diagram: per-client rank stacks -> averaged bar + frozen-A strip
sy0 = 81.0
for k, (sw, sh) in enumerate(((2.4, 12.0), (2.4, 7.0), (2.4, 4.0))):
    sx = DX0 + 2.5 + k * 3.6
    ax.add_patch(Rectangle((sx, sy0), sw, sh, fc=RED, ec='#ffffff', lw=0.4,
                           zorder=5))
    text(sx + sw / 2, sy0 - 1.4, ['16', '8', '4'][k], f'D.stack{k}',
         (sx - 0.4, sx + sw + 0.4, sy0 - 2.6, sy0 - 0.2), fontsize=5.5,
         ha='center', va='center', color='#666666')
arrow(DX0 + 14.6, 86.5, DX0 + 19.0, 86.5, color=GRAY, lw=0.9, ms=5)
ax.add_patch(Rectangle((DX0 + 20.5, sy0), 7.0, 7.5, fc='#ea9999',
                       ec='#ffffff', lw=0.4, zorder=5))
text(DX0 + 24.0, sy0 - 1.5, f'${BBAR}$', 'D.avg', (DX0 + 19.5, DX0 + 28.5,
     78.1, 80.9), fontsize=5.5, ha='center', va='center', color=DARK)
# frozen-A strip with padlock
fz0, fz1 = DX0 + 20.5, DX0 + 32.0
ax.add_patch(Rectangle((fz0, 91.3), fz1 - fz0, 3.2, fc='#c9d9ea',
                       ec=BLUE, lw=0.6, zorder=5))
text(fz0 + 3.4, 92.9, f'${ABAR}$', 'D.strip', (fz0 + 0.5, fz0 + 7.0, 91.5, 94.3),
     fontsize=6.0, ha='center', va='center', color=BLUE_CHIP_TEXT, zorder=6)
icon_padlock(fz1 - 1.8, 92.9, s=2.2, color=BLUE_CHIP_TEXT)

MIDY = 98
for (ax0, ax1), (bx0_, _) in zip(BX[:-1], BX[1:]):
    arrow(ax1 + 0.4, MIDY, bx0_ - 0.4, MIDY, color=GRAY, lw=1.2)

# ------------------------------------------------- orthogonal link plumbing
CARD_C = (35.0, 96.0, 157.0)
DROP_X = tuple(c for c in CARD_C)
RISE_X = tuple(c + 8 for c in CARD_C)
DL_BUS_Y, UL_BUS_Y = 59.0, 49.0
GATE_CX = 178.0

line(122.0, BY0, 122.0, DL_BUS_Y, color=BLUE, lw=1.0, ls=(0, (3, 2)))
line(DROP_X[0], DL_BUS_Y, DROP_X[-1], DL_BUS_Y, color=BLUE, lw=1.0, ls=(0, (3, 2)))
for x in DROP_X:
    arrow(x, DL_BUS_Y, x, 42.5, color=BLUE, lw=1.0, ls=(0, (3, 2)), ms=6)
text(65.0, 61.7, 'broadcast: backbone $w$, ranks $r_i$', 'dl.label',
     (36.9, 93.1, 59.4, 64.0), fontsize=MATH, ha='center', va='center', color=BLUE)

for x in RISE_X:
    line(x, 42.0, x, UL_BUS_Y, color=RED, lw=1.1)
line(RISE_X[0], UL_BUS_Y, GATE_CX, UL_BUS_Y, color=RED, lw=1.1)
text(73.5, 51.8, 'uplink: $B_i(r_i)$ only, arriving by $\\tau$', 'ul.label',
     (46.4, 100.6, 49.4, 54.2), fontsize=MATH, ha='center', va='center', color=RED)

GW0, GW1 = 172.0, 184.0
box(GW0, 52.5, GW1, 68.0, fc='#ffffff', ec=GRAY, lw=1.2, rounding=1.0)
arrow(GATE_CX, UL_BUS_Y, GATE_CX, 52.5, color=RED, lw=1.1, ms=6)
icon_clock(GATE_CX, 65.2, r=2.4)
text(GATE_CX, 58.6, 'on-time\ngate ($\\tau$)', 'gate.label',
     (GW0 + 0.6, GW1 - 0.6, 55.2, 62.0), fontsize=6.0, ha='center', va='center',
     color=DARK, linespacing=1.3)
arrow(GATE_CX, 68.0, GATE_CX, 78.0, color=GREEN, lw=1.2, ms=6)
line(182.0, 68.0, 182.0, 71.5, color='#777777', lw=0.9, ls=(0, (4, 2.5)), z=3)
line(182.0, 71.5, 22, 71.5, color='#777777', lw=0.9, ls=(0, (4, 2.5)))
arrow(22, 71.5, 22, 76.0, color='#777777', lw=0.9, ls=(0, (4, 2.5)), ms=6)
ax.plot([109.4, 111.0], [44.3, 45.9], color='#888888', lw=0.8, zorder=4)
ax.plot([109.4, 111.0], [45.9, 44.3], color='#888888', lw=0.8, zorder=4)
text(118.6, 45.1, 'late/lost\ndiscarded', 'gate.late', (112.6, 124.0, 42.5, 47.7),
     fontsize=5.7, ha='center', va='center', color='#888888', linespacing=1.3)
text(65, 73.6, 'observed outcomes update the link-state history',
     'fb.label', (30.0, 116.0, 72.0, 75.2), fontsize=6.0, ha='center', va='center',
     color='#666666')

# ---------------------------------------------------------------- client band
CARDS = [
    (10, 60, 'Ground client — stable', icon_phone, 16),
    (71, 121, 'Aerial client — handover', icon_drone, 8),
    (132, 182, 'Satellite client — outage', icon_sat, 4),
]
RMAX = 16
CY0, CY1 = 6, 42
for cx0, cx1, header, icon, r in CARDS:
    box(cx0, CY0, cx1, CY1)
    cxc = (cx0 + cx1) / 2
    text(cxc, 39.2, header, f'card{r}.header', (cx0 + 1, cx1 - 1, 37.6, 40.8),
         fontsize=6.4, ha='center', va='center', fontweight='bold', color=DARK)
    icon(cxc, 34.9, s=4.6)
    box(cx0 + 5, 27.6, cx1 - 5, 32.0, fc=CHIP_GRAY_FILL, ec='#bbbbbb', lw=0.7,
        rounding=0.8)
    text(cxc - 1.2, 29.8, 'backbone $w$', f'card{r}.chip1',
         (cx0 + 6.5, cx1 - 6.5, 27.7, 31.9), fontsize=6.2, ha='center', va='center',
         color=DARK)
    icon_padlock(cx1 - 8.0, 29.8, s=2.6)
    box(cx0 + 5, 22.6, cx1 - 5, 27.0, fc=BLUE_CHIP_FILL, ec=BLUE, lw=0.7,
        rounding=0.8)
    text(cxc - 1.2, 24.8, 'shared $A$ — not sent', f'card{r}.chip2',
         (cx0 + 6.5, cx1 - 6.5, 22.7, 26.9), fontsize=6.2, ha='center', va='center',
         color=BLUE_CHIP_TEXT)
    icon_padlock(cx1 - 8.0, 24.8, s=2.6, color=BLUE_CHIP_TEXT)
    n = RMAX
    cw, gap = 1.15, 0.62
    total = n * cw + (n - 1) * gap
    x_start = cxc - total / 2
    top, bot = 21.8, 16.8
    for j in range(n):
        xj = x_start + j * (cw + gap)
        shaded = j < r
        ax.add_patch(Rectangle((xj, bot), cw, top - bot,
                               fc=(RED if shaded else '#ffffff'),
                               ec=(RED if shaded else '#aaaaaa'), lw=0.5, zorder=2))
    text(cxc, 14.3, 'trainable $B$, sliced to $r_i$', f'card{r}.cap',
         (cx0 + 2, cx1 - 2, 12.0, 16.6), fontsize=MATH, ha='center', va='center',
         color=DARK)
    box(cxc - 15, 7.4, cxc + 15, 11.8, fc='#fbe9e7', ec=RED, lw=0.8, rounding=1.5)
    text(cxc, 9.6, 'assigned $r_i = %d$' % r, f'card{r}.pill',
         (cxc - 14, cxc + 14, 7.2, 12.0), fontsize=MATH, ha='center', va='center',
         color=RED)

text(95, 3.2, 'Uplink = trainable B only: 1,408 params per rank column (5.63 KB); '
              'rank 16: 0.090 MB vs 5.29 MB full model (58.7×).',
     'payload', (8, 182, 1.6, 4.8), fontsize=6.0, ha='center', va='center',
     color='#555555')

check_fits()

# --------------------------------------------------------------------- export
require_matplotlib_panel_alignment(
    fig,
    json_out=f'{NAME}.alignment.json',
    overlay_svg=f'{NAME}.alignment.svg',
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    strict=True,
)
fig.savefig(f'{NAME}.svg', bbox_inches='tight')
fig.savefig(f'{NAME}.pdf', bbox_inches='tight')
fig.savefig(f'{NAME}.png', dpi=600, bbox_inches='tight')
print('exported', NAME)
