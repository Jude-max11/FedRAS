"""FedRAS system-architecture figure, v5 strict-grid redesign (fig:architecture).

Backend: Python (matplotlib), per the nature-figure backend gate (user-selected).
Single-panel schematic-led composite, 190 mm Elsevier double-column width.

v5 design contract (author request: horizontal-vertical neatness + icons):
- strict modular grid: four equal server boxes on one row (equal width, height,
  gaps, shared top/bottom edges), three equal client cards, one horizontal
  arrow mid-line, all connectors orthogonal (Manhattan routing);
- pictographic icons drawn with vector primitives: server rack (band label),
  smartphone / drone / satellite (client cards), padlocks on frozen chips,
  clock inside the on-time gate; no bitmap, no external assets;
- notation and every number match CN_reviewed.tex Section 3: hazard (eq. 3),
  capacity (eq. 5), D = 1,408 params/rank column = 5.63 KB, r=16 -> 0.090 MB,
  full model 5.29 MB (58.7x), ranks {2,4,8,16}; Propositions 1 and 3 cited in
  the caption.
Glyph floor: DejaVu mathtext shrinks first-level scripts to 0.675x, so
mathtext with sub/superscripts uses a parent of at least 7.6 pt; accents use
precomposed Unicode and the hazard exponent is merged into one exp() so no
script-of-script level exists.
Layout safety: every text artist is measured against its host region after the
final draw (check_fits) and the script fails loudly on overflow.

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
W, H = 190.0, 106.0

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
MATH = 7.6   # parent size for mathtext with sub/superscripts (scripts >= 5.13 pt)

PHAT = 'P\u0302'    # P-hat
BTIL = 'B\u0303'    # B-tilde
RBAR = 'R\u0304'    # R-bar
BBAR = 'B\u0304'    # B-bar
ABAR = 'A\u0304'    # A-bar

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
    """Rack-mounted server: chassis with three slots and status LEDs."""
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
    """Ground client: smartphone."""
    w, h = 0.60 * s, s
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0,rounding_size=0.5',
                                fc='#ffffff', ec=DARK, lw=0.7, zorder=4))
    ax.add_patch(Rectangle((cx - w / 2 + 0.45, cy - h / 2 + 0.95),
                           w - 0.9, h - 1.9, fc=BLUE_CHIP_FILL,
                           ec='none', zorder=5))
    ax.add_patch(Circle((cx, cy - h / 2 + 0.55), 0.22, fc=DARK, ec='none', zorder=5))


def icon_drone(cx, cy, s=4.6):
    """Aerial client: quad-rotor UAV."""
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
    """Satellite client: bus with two solar panels and antenna."""
    ax.add_patch(Rectangle((cx - 0.17 * s, cy - 0.26 * s), 0.34 * s, 0.52 * s,
                           fc='#ffffff', ec=DARK, lw=0.7, zorder=5))
    for sx in (-1, 1):
        ax.add_patch(Rectangle((cx + sx * 0.25 * s - (0 if sx > 0 else 0.40 * s),
                                cy - 0.16 * s), 0.40 * s, 0.32 * s,
                               fc=BLUE_CHIP_FILL, ec=DARK, lw=0.6, zorder=4))
    ax.plot([cx, cx], [cy + 0.26 * s, cy + 0.44 * s], color=DARK, lw=0.7, zorder=4)
    ax.add_patch(Circle((cx, cy + 0.47 * s), 0.09 * s, fc=DARK, ec='none', zorder=5))


def icon_padlock(cx, cy, s=2.5, color=GRAY):
    """Frozen-factor padlock."""
    ax.add_patch(Rectangle((cx - 0.38 * s, cy - 0.55 * s), 0.76 * s, 0.60 * s,
                           fc='#ffffff', ec=color, lw=0.6, zorder=5))
    th = np.linspace(0, np.pi, 14)
    ax.plot(cx + 0.26 * s * np.cos(th), cy - 0.55 * s + 0.50 * s
            + 0.34 * s * np.sin(th) * 0 + 0.30 * s * np.sin(th),
            color=color, lw=0.6, zorder=5)


def icon_clock(cx, cy, r=2.6):
    """Deadline clock for the on-time gate."""
    ax.add_patch(Circle((cx, cy), r, fc='#ffffff', ec=DARK, lw=0.8, zorder=5))
    for ang in (90, 0, 270, 180):
        a = np.deg2rad(ang)
        ax.plot([cx + 0.78 * r * np.cos(a), cx + 0.95 * r * np.cos(a)],
                [cy + 0.78 * r * np.sin(a), cy + 0.95 * r * np.sin(a)],
                color=DARK, lw=0.55, zorder=5)
    ax.plot([cx, cx], [cy, cy + 0.55 * r], color=DARK, lw=0.7, zorder=5)
    ax.plot([cx, cx + 0.42 * r], [cy, cy + 0.10 * r], color=RED, lw=0.7, zorder=5)
    ax.add_patch(Circle((cx, cy), 0.16, fc=DARK, ec='none', zorder=6))


# ---------------------------------------------------------------- server band
box(3, 68.5, 187, 104, fc=BAND_FILL, ec=NEUTRAL_EDGE, lw=1.0, rounding=2.0)
icon_server(10.3, 101.1)
ax.text(15.5, 101.2, 'Edge aggregator — FedRAS server', fontsize=7.2,
        fontweight='bold', color=DARK, va='center')

BX = [(6, 42), (47, 97), (102, 142), (147, 184)]   # equal boxes, equal gaps
BY0, BY1 = 70, 96
for x0, x1 in BX:
    box(x0, BY0, x1, BY1)
box(BX[2][0], BY0, BX[2][1], BY1, fc=RED_FILL, ec=RED, lw=1.4)  # hero

AX0, AX1 = BX[0]
BX0, BX1 = BX[1]
CX0, CX1 = BX[2]
DX0, DX1 = BX[3]

text((AX0 + AX1) / 2, 93.9, 'Link states', 'A.title', (AX0 + 1, AX1 - 1, 92.2, 95.6),
     fontsize=7, fontweight='bold', ha='center', va='center')
text((AX0 + AX1) / 2, 84.5, f'uplink rate ${RBAR}_i$,\nmobility state $\\mu_i$',
     'A.body', (AX0 + 1, AX1 - 1, 79.0, 90.0), fontsize=MATH, ha='center',
     va='center', color=DARK, linespacing=1.5)
text((AX0 + AX1) / 2, 74.5, '(estimated each round)', 'A.note',
     (AX0 + 1, AX1 - 1, 72.8, 76.2), fontsize=6.0, ha='center', va='center',
     color='#666666')

text((BX0 + BX1) / 2, 93.9, 'Per–client–rank prediction', 'B.title',
     (BX0 + 1, BX1 - 1, 92.2, 95.6), fontsize=7, fontweight='bold',
     ha='center', va='center')
text(BX0 + 2.2, 90.2, 'delivery hazard  (eq. 3)', 'B.lead1',
     (BX0 + 1, BX1 - 1, 88.6, 91.8), fontsize=6.0, ha='left', va='center',
     color='#666666')
text((BX0 + BX1) / 2, 86.8,
     f'${PHAT}_{{\\mathrm{{on}},i}}(r) = \\exp(-\\mu_i - 8B_i(r)/({RBAR}_i \\tau))$',
     'B.formula1', (BX0 + 1, BX1 - 1, 84.0, 89.6), fontsize=MATH,
     ha='center', va='center')
text(BX0 + 2.2, 81.7, 'saturating capacity  (eq. 5)', 'B.lead2',
     (BX0 + 1, BX1 - 1, 80.1, 83.3), fontsize=6.0, ha='left', va='center',
     color='#666666')
text((BX0 + BX1) / 2, 78.1,
     '$G_i(r) = \\pi_i\\sqrt{\\min(r,\\, r_{\\max})/r_{\\max}}$',
     'B.formula2', (BX0 + 1, BX1 - 1, 75.1, 81.1), fontsize=MATH,
     ha='center', va='center')

text((CX0 + CX1) / 2, 93.9, 'Deadline-aware scheduler', 'C.title',
     (CX0 + 1, CX1 - 1, 92.2, 95.6), fontsize=6.6, fontweight='bold',
     ha='center', va='center', color=RED)
text((CX0 + CX1) / 2, 87.2,
     '$\\max\\ \\sum_i G_i(r_i)\\,%s_{\\mathrm{on},i}(r_i)$' % PHAT,
     'C.obj', (CX0 + 1, CX1 - 1, 82.9, 91.5), fontsize=MATH, ha='center', va='center')
text((CX0 + CX1) / 2, 78.5,
     f'$\\mathrm{{s.t.}}\\ \\sum_i {BTIL}_i(r_i) \\leq {BTIL}_{{\\max}}$',
     'C.s.t.', (CX0 + 1, CX1 - 1, 74.2, 82.8), fontsize=MATH, ha='center', va='center')
text((CX0 + CX1) / 2, 72.0, 'output: client i → rank r', 'C.out',
     (CX0 + 1, CX1 - 1, 70.2, 73.8), fontsize=7.0, ha='center', va='center', color=RED)

text((DX0 + DX1) / 2, 92.2, 'Rank-masked\naggregation', 'D.title',
     (DX0 + 1, DX1 - 1, 88.7, 95.7), fontsize=6.5, fontweight='bold',
     ha='center', va='center', linespacing=1.35)
text((DX0 + DX1) / 2, 82.0, f'${BBAR}_{{:,j}}$ averaged\nover $r_i \\geq j$',
     'D.body', (DX0 + 1, DX1 - 1, 76.7, 87.3), fontsize=MATH, ha='center',
     va='center', color=DARK, linespacing=1.3)
text((DX0 + DX1) / 2, 73.5, f'${ABAR} = A$ frozen', 'D.frozen',
     (DX0 + 1, DX1 - 1, 71.4, 75.6), fontsize=MATH, ha='center', va='center',
     color=DARK)

MIDY = 83
for (ax0, ax1), (bx0, _) in zip(BX[:-1], BX[1:]):
    arrow(ax1 + 0.4, MIDY, bx0 - 0.4, MIDY, color=GRAY, lw=1.2)

# ------------------------------------------------- orthogonal link plumbing
CARD_C = (35.0, 96.0, 157.0)
DROP_X = tuple(c for c in CARD_C)            # blue broadcast drops (card centers)
RISE_X = tuple(c + 8 for c in CARD_C)        # red uplink risers (center + 8)
DL_BUS_Y, UL_BUS_Y = 59.0, 49.0
GATE_CX = 178.0

# blue broadcast: scheduler -> bus -> three drops
line(122.0, BY0, 122.0, DL_BUS_Y, color=BLUE, lw=1.0, ls=(0, (3, 2)))
line(DROP_X[0], DL_BUS_Y, DROP_X[-1], DL_BUS_Y, color=BLUE, lw=1.0, ls=(0, (3, 2)))
for x in DROP_X:
    arrow(x, DL_BUS_Y, x, 42.5, color=BLUE, lw=1.0, ls=(0, (3, 2)), ms=6)
text(65.0, 61.7, 'broadcast: backbone $w$, ranks $r_i$', 'dl.label',
     (36.9, 93.1, 59.4, 64.0), fontsize=MATH, ha='center', va='center', color=BLUE)

# red uplink: card risers -> bus -> on-time gate
for x in RISE_X:
    line(x, 42.0, x, UL_BUS_Y, color=RED, lw=1.1)
line(RISE_X[0], UL_BUS_Y, GATE_CX, UL_BUS_Y, color=RED, lw=1.1)
text(73.5, 51.8, 'uplink: $B_i(r_i)$ only, arriving by $\\tau$', 'ul.label',
     (46.4, 100.6, 49.4, 54.2), fontsize=MATH, ha='center', va='center', color=RED)

# on-time gate: slot below the aggregation box, clock + label inside
GW0, GW1 = 172.0, 184.0
box(GW0, 52.5, GW1, 68.0, fc='#ffffff', ec=GRAY, lw=1.2, rounding=1.0)
arrow(GATE_CX, UL_BUS_Y, GATE_CX, 52.5, color=RED, lw=1.1, ms=6)
icon_clock(GATE_CX, 65.2, r=2.4)
text(GATE_CX, 58.6, 'on-time\ngate ($\\tau$)', 'gate.label',
     (GW0 + 0.6, GW1 - 0.6, 55.2, 62.0), fontsize=6.0, ha='center', va='center',
     color=DARK, linespacing=1.3)
arrow(GATE_CX, 68.0, GATE_CX, 70.0, color=GREEN, lw=1.2, ms=6)
text(118.0, 45.1, 'late/lost\ndiscarded', 'gate.late', (112.0, 124.0, 42.5, 47.7),
     fontsize=5.7, ha='center', va='center', color='#888888', linespacing=1.3)

# feedback: observed outcomes -> link-state history
line(GW0, 64.4, 22, 64.4, color='#777777', lw=0.9, ls=(0, (4, 2.5)))
arrow(22, 64.4, 22, 67.8, color='#777777', lw=0.9, ls=(0, (4, 2.5)), ms=6)
text(65, 66.2, 'observed outcomes update the link-state history',
     'fb.label', (30.0, 116.0, 64.6, 67.8), fontsize=6.0, ha='center', va='center',
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
    # frozen backbone chip (padlock = frozen)
    box(cx0 + 5, 27.6, cx1 - 5, 32.0, fc=CHIP_GRAY_FILL, ec='#bbbbbb', lw=0.7,
        rounding=0.8)
    text(cxc - 1.2, 29.8, 'backbone $w$', f'card{r}.chip1',
         (cx0 + 6.5, cx1 - 6.5, 27.7, 31.9), fontsize=6.2, ha='center', va='center',
         color=DARK)
    icon_padlock(cx1 - 8.0, 29.8, s=2.6)
    # frozen shared A chip (padlock = frozen)
    box(cx0 + 5, 22.6, cx1 - 5, 27.0, fc=BLUE_CHIP_FILL, ec=BLUE, lw=0.7,
        rounding=0.8)
    text(cxc - 1.2, 24.8, 'shared $A$ — not sent', f'card{r}.chip2',
         (cx0 + 6.5, cx1 - 6.5, 22.7, 26.9), fontsize=6.2, ha='center', va='center',
         color=BLUE_CHIP_TEXT)
    icon_padlock(cx1 - 8.0, 24.8, s=2.6, color=BLUE_CHIP_TEXT)
    # trainable B, sliced to r: RMAX mini-columns, first r shaded
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
    # assigned-rank pill
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
