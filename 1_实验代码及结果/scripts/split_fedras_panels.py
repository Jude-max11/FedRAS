#!/usr/bin/env python3
"""Split the FedRAS combined experiment figures into independent,
zero-margin per-panel PDFs (plus shared-legend strips where present),
so panel subcaptions live in the LaTeX source (subfloat), not baked
into the artwork. Single-panel figures are just tight-cropped.

Vector surgery on the matplotlib output in recolored_figures/: panel
anchors come from the bottom-axis drawing segments, boundaries from the
word block hugging each next left spine. Regression gates: every source
word must land in some piece (minus intentionally stripped title bands),
no boundary may slice a word, every piece must be ink-tight.
"""
import os
import sys
import pymupdf

ZOOM = 4.0
INK = 250
GAP_ROW_PT = 9.0
TITLE_MAX_H_PT = 15.0
LEGEND_MAX_H_PT = 30.0
TALL_PT = 20.0

# name -> (n_panels, legend_position, stacked)   legend: None|'top'
FIGS = {
    'fig_sim_accuracy':   (2, None,  False),
    'fig_sim_regimes':    (2, 'top', False),
    'fig_dp_study':       (2, 'top', False),
    'fig_sensitivity':    (3, None,  False),
    'fig_calibration':    (2, None,  False),
    'fig_weakbb':         (2, None,  False),
}
# single-panel figures: tight-crop only, output keeps the same stem
CROPS = ['fig_trainer_nominal', 'fig_trainer_stress', 'fig_rank_dynamics',
         'fig_cross_dataset', 'fig_normlaw']


def runs(mask, min_gap):
    out, start, gap = [], None, 0
    for i, on in enumerate(mask):
        if on:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                out.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        out.append((start, len(mask)))
    return out


def ink_projection(pm, x0, y0, x1, y1):
    w, n = pm.width, pm.n
    s = pm.samples
    rows = [False] * (y1 - y0)
    cols = [False] * (x1 - x0)
    step = w * n
    for y in range(y0, y1):
        base = y * step
        row_on = False
        for x in range(x0, x1):
            o = base + x * n
            if s[o] < INK or s[o + 1] < INK or s[o + 2] < INK:
                row_on = True
                cols[x - x0] = True
        rows[y - y0] = row_on
    return rows, cols


def tight_bbox(pm, x0, y0, x1, y1):
    rows, cols = ink_projection(pm, x0, y0, x1, y1)
    ys = [i for i, on in enumerate(rows) if on]
    xs = [i for i, on in enumerate(cols) if on]
    if not ys or not xs:
        raise RuntimeError('empty region')
    return (x0 + xs[0], y0 + ys[0], x0 + xs[-1] + 1, y0 + ys[-1] + 1)


def export_piece(src, clip_px, out_path):
    sx = 1 / ZOOM
    clip = pymupdf.Rect(clip_px[0] * sx, clip_px[1] * sx,
                        clip_px[2] * sx, clip_px[3] * sx)
    out = pymupdf.open()
    page = out.new_page(width=clip.width, height=clip.height)
    page.show_pdf_page(page.rect, src, 0, clip=clip)
    out.save(out_path, garbage=3, deflate=True)
    out.close()
    return clip.width, clip.height, clip


def check_margin(path):
    doc = pymupdf.open(path)
    pm = doc[0].get_pixmap(matrix=pymupdf.Matrix(600 / 72, 600 / 72))
    doc.close()
    b = tight_bbox(pm, 0, 0, pm.width, pm.height)
    return min(b[0], b[1], pm.width - b[2], pm.height - b[3]) * 72 / 600


def strip_title(pm, px_rect):
    x0, y0, x1, y1 = px_rect
    rows, _ = ink_projection(pm, x0, y0, x1, y1)
    bands = runs(rows, 2)
    if not bands:
        raise RuntimeError('no ink in panel')
    first = bands[0]
    h_pt = (first[1] - first[0]) / ZOOM
    if h_pt > TITLE_MAX_H_PT or len(bands) == 1:
        return px_rect, None
    cut = (x0, y0 + first[0], x1, y0 + first[1])
    return tight_bbox(pm, x0, y0 + first[1], x1, y1), cut


def axes_spans(page, pm, band, n_panels):
    ry0, ry1 = band
    groups = []
    for d in page.get_drawings():
        for item in d['items']:
            if item[0] == 'l':
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 0.7 and abs(p1.x - p2.x) > 40:
                    y, a, b = p1.y, min(p1.x, p2.x), max(p1.x, p2.x)
                else:
                    continue
            elif item[0] == 're':
                r = item[1]
                if r.height < 1.5 and r.width > 40:
                    y, a, b = r.y0, r.x0, r.x1
                else:
                    continue
            else:
                continue
            for g in groups:
                if abs(g[0] - y) <= 2.0:
                    g[1].append((a, b))
                    break
            else:
                groups.append([y, [(a, b)]])
    cand = []
    for y, segs in groups:
        uniq = []
        for s in sorted(segs):
            if not uniq or abs(s[0] - uniq[-1][0]) > 1 or abs(s[1] - uniq[-1][1]) > 1:
                uniq.append(s)
        if len(uniq) == n_panels:
            cand.append((y, uniq))
    if not cand:
        raise RuntimeError(f'no drawing row with {n_panels} axis segments')
    _, axes = max(cand, key=lambda t: t[0])
    words = [w for w in page.get_text('words')
             if ry0 / ZOOM <= (w[1] + w[3]) / 2 <= ry1 / ZOOM]
    _, cols = ink_projection(pm, 0, ry0, pm.width, ry1)
    col_runs = runs(cols, 8)
    xs = [i for i, on in enumerate(cols) if on]
    ink0, ink1 = xs[0], xs[-1] + 1
    bounds_px = []
    for i in range(1, n_panels):
        s0, prev_e = axes[i][0], axes[i - 1][1]
        block = [w for w in words
                 if s0 - 45.0 <= w[2] <= s0 + 2.0 and w[0] >= prev_e]
        if block:
            grown = True
            while grown:
                grown = False
                cur = min(w[0] for w in block)
                for w in words:
                    if w in block or w[0] < prev_e or w[2] > s0 + 2.0:
                        continue
                    if not (w[0] < cur <= w[2] + 12.0):
                        continue
                    if any(not (w[3] <= b[1] or w[1] >= b[3]) for b in block):
                        block.append(w)
                        grown = True
            bounds_px.append(int((min(w[0] for w in block) - 0.5) * ZOOM))
        else:
            # shared-y panel: no outside labels at all, cut in the middle
            # of the whitespace gutter between the two panels' ink blocks
            def run_containing(x):
                for r in col_runs:
                    if r[0] <= x <= r[1]:
                        return r
                return None
            ra = run_containing(prev_e * ZOOM)
            rb = run_containing(s0 * ZOOM)
            if ra is None or rb is None or ra is rb:
                raise RuntimeError(f'degenerate gutter at spine {s0:.1f}pt')
            bounds_px.append((ra[1] + rb[0]) // 2)
    for bd in bounds_px:
        for w in words:
            if w[0] * ZOOM + 0.5 < bd < w[2] * ZOOM - 0.5:
                raise RuntimeError(f'boundary {bd} slices word {w[4]!r}')
    edges = [ink0] + bounds_px + [ink1]
    return [(edges[i], edges[i + 1]) for i in range(n_panels)]


def gutter_spans(pm, band, n_panels):
    ry0, ry1 = band
    W = pm.width
    _, cols = ink_projection(pm, 0, ry0, W, ry1)
    col_runs = runs(cols, 8)
    stacks = []
    for r in col_runs:
        if stacks and r[0] - stacks[-1][1] <= int(8 * ZOOM):
            stacks[-1][1] = max(stacks[-1][1], r[1])
        else:
            stacks.append([r[0], r[1]])
    if len(stacks) < n_panels:
        raise RuntimeError(f'only {len(stacks)} column stacks '
                           f'for {n_panels} panels: {stacks}')
    return sorted(sorted(stacks, key=lambda r: r[1] - r[0])[-n_panels:],
                  key=lambda r: r[0])


def check_word_coverage(src, band_px, pieces, cuts, name):
    ry0, ry1 = band_px
    bad = []
    for w in src[0].get_text('words'):
        yc = (w[1] + w[3]) / 2 * ZOOM
        if not (ry0 <= yc <= ry1):
            continue
        r = pymupdf.Rect(w[:4])
        c = (r.tl + r.br) / 2
        if any(c in cut for cut in cuts):
            continue
        if not any(c in pr for pr in pieces):
            bad.append(w[4])
    if bad:
        raise RuntimeError(f'{name}: words lost outside pieces: {bad}')


def split(name, src_dir, out_dir):
    n_panels, legend_pos, stacked = FIGS[name]
    src = pymupdf.open(os.path.join(src_dir, name + '.pdf'))
    page = src[0]
    pm = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), colorspace=pymupdf.csRGB)
    W, H = pm.width, pm.height
    rows, _ = ink_projection(pm, 0, 0, W, H)
    bands = runs(rows, int(GAP_ROW_PT * ZOOM))

    legend_band = None
    if legend_pos == 'top':
        cand = bands[0]
        if (cand[1] - cand[0]) / ZOOM > LEGEND_MAX_H_PT:
            raise RuntimeError(f'{name}: top band is no legend')
        legend_band = cand
        bands = bands[1:]
    band = bands[0]
    band_px = band

    cuts, pieces, report = [], [], []
    letters = 'abcdefghijklmnopqrstuvwxyz'
    pi = 0

    def emit(px_rect):
        nonlocal pi
        rect, cut = strip_title(pm, px_rect)
        if cut:
            cuts.append(pymupdf.Rect(cut[0] / ZOOM, cut[1] / ZOOM,
                                     cut[2] / ZOOM, cut[3] / ZOOM))
        slug = f'{name}_{letters[pi]}'
        w, h, clip = export_piece(src, rect, os.path.join(out_dir, slug + '.pdf'))
        pieces.append(clip)
        mg = check_margin(os.path.join(out_dir, slug + '.pdf'))
        report.append((slug, w, h, mg))
        pi += 1

    try:
        spans = axes_spans(page, pm, band, n_panels)
    except RuntimeError:
        spans = gutter_spans(pm, band, n_panels)
    for sp in spans:
        emit(tight_bbox(pm, sp[0], band[0], sp[1], band[1]))

    if legend_band is not None:
        rect = tight_bbox(pm, 0, legend_band[0], W, legend_band[1])
        w, h, clip = export_piece(src, rect,
                                  os.path.join(out_dir, name + '_legend.pdf'))
        pieces.append(clip)
        mg = check_margin(os.path.join(out_dir, name + '_legend.pdf'))
        report.append((name + '_legend', w, h, mg))

    check_word_coverage(src, band_px, pieces, cuts, name)
    src.close()
    return report


def crop(name, src_dir, out_dir):
    src = pymupdf.open(os.path.join(src_dir, name + '.pdf'))
    pm = src[0].get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), colorspace=pymupdf.csRGB)
    rect = tight_bbox(pm, 0, 0, pm.width, pm.height)
    w, h, _ = export_piece(src, rect, os.path.join(out_dir, name + '.pdf'))
    mg = check_margin(os.path.join(out_dir, name + '.pdf'))
    src.close()
    return [(name, w, h, mg)]


def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else src_dir
    worst = 0.0
    for name in FIGS:
        for slug, w, h, mg in split(name, src_dir, out_dir):
            worst = max(worst, mg)
            print(f'{slug:28s} {w:7.2f} x {h:6.2f} pt   margin {mg:.2f} pt')
    for name in CROPS:
        for slug, w, h, mg in crop(name, src_dir, out_dir):
            worst = max(worst, mg)
            print(f'{slug:28s} {w:7.2f} x {h:6.2f} pt   margin {mg:.2f} pt')
    print(f'worst margin: {worst:.2f} pt')
    assert worst < 0.35, 'zero-margin requirement violated'
    print('word-coverage regression: all pieces complete')


if __name__ == '__main__':
    main()
