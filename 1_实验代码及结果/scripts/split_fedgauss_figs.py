#!/usr/bin/env python3
"""Split the FedGauss submission's combined experiment figures into
independent, zero-margin per-panel PDFs (plus one shared-legend strip
where present), so that panel titles and captions live in the LaTeX
source (subfloat), not baked into the artwork.

Pure PDF surgery on the existing vector figures: no data re-runs, no
rasterisation. Each output page is exactly the tight ink bbox of its
piece (white margin <= ~0.3 pt from bbox rounding).

Regression gate: every text word of the source must land in some piece,
except words inside intentionally stripped title bands.
"""
import os
import sys
import pymupdf

ZOOM = 4.0                       # 288 dpi analysis raster (vector export untouched)
INK = 250                        # gray < INK counts as ink
GAP_ROW_PT = 9.0                 # horizontal white gap that separates coarse bands
GUTTER_PT = 8.0                  # vertical white run this wide can separate panels
TITLE_MAX_H_PT = 15.0            # a top ink band this short is a baked title
LEGEND_MAX_H_PT = 46.0           # bottom band must be at most this tall
TALL_PT = 20.0                   # axes block height for the stacked walker

# name -> (n_panels, has_legend, stacked)
FIGS = {
    'fig_main_results':   (3, True,  False),
    'fig_attr':           (3, False, False),
    'fig_matrix_heatmap': (2, False, False),
    'fig_ladder':         (1, False, False),
    'fig_mechanism':      (2, False, True),
    'fig_sensitivity':    (3, True,  False),
    'fig_comm':           (2, False, False),
    'fig_runtime_memory': (2, True,  False),
    'fig_scalability':    (2, True,  False),
    'fig_robust':         (2, True,  False),
    'fig_paper_ablation': (2, False, False),
}

# text surgery on the source artwork before splitting (pre-existing defects):
# literal "\%" typo in fig_main_results panel (c), and the "-0.025" tick label
# that collides with both neighbours because three labels share ~24 pt.
FIX_BACKSLASH = {'fig_main_results'}
DROP_WORDS = {'fig_main_results': ['\u22120.025']}


def runs(mask, min_gap):
    """Ink runs along one axis: mask[i] True = ink present.
    Returns list of (start, end_exclusive), splitting on white gaps >= min_gap."""
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
    """Row- and column-ink boolean arrays for the pixmap region."""
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
    """Exact ink bbox inside region, in px (top-left origin)."""
    rows, cols = ink_projection(pm, x0, y0, x1, y1)
    ys = [i for i, on in enumerate(rows) if on]
    xs = [i for i, on in enumerate(cols) if on]
    if not ys or not xs:
        raise RuntimeError('empty region')
    return (x0 + xs[0], y0 + ys[0], x0 + xs[-1] + 1, y0 + ys[-1] + 1)


def fix_source_text(doc, name):
    """Surgical text fixes on the source page; returns list of fix notes."""
    page = doc[0]
    notes = []
    if name in FIX_BACKSLASH:
        raw = page.get_text('rawdict')
        for block in raw['blocks']:
            for line in block.get('lines', []):
                for span in line.get('spans', []):
                    for ch in span.get('chars', []):
                        if ch['c'] == '\\':
                            r = pymupdf.Rect(ch['bbox'])
                            page.add_redact_annot(r + (-0.3, -0.3, 0.3, 0.3))
                            notes.append('dropped literal backslash')
    for target in DROP_WORDS.get(name, []):
        hit = page.search_for(target)
        if hit:
            r = hit[0]
            page.add_redact_annot(r + (1.5, -0.5, -1.5, 0.5))
            notes.append(f'dropped colliding tick label {target!r}')
    if notes:
        page.apply_redactions()
    return notes


def strip_title(pm, px_rect):
    """Cut the baked title band off a panel; returns (rect, cut_rect|None)."""
    x0, y0, x1, y1 = px_rect
    rows, _ = ink_projection(pm, x0, y0, x1, y1)
    bands = runs(rows, 2)  # 2 px ~ 0.5 pt granularity
    if not bands:
        raise RuntimeError('no ink in panel')
    first = bands[0]
    h_pt = (first[1] - first[0]) / ZOOM
    if h_pt > TITLE_MAX_H_PT or len(bands) == 1:
        return px_rect, None                   # no strippable title
    cut = (x0, y0 + first[0], x1, y0 + first[1])
    return tight_bbox(pm, x0, y0 + first[1], x1, y1), cut


def row_longest_run(pm, y0, y1):
    """Longest continuous ink run per row: axis lines / box edges are long
    solid runs, text rows shatter into short fragments."""
    w, n = pm.width, pm.n
    s = pm.samples
    step = w * n
    out = []
    for y in range(y0, y1):
        base = y * step
        best = cur = 0
        for x in range(w):
            o = base + x * n
            if s[o] < INK or s[o + 1] < INK or s[o + 2] < INK:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        out.append(best)
    return out


def axes_spans(page, pm, band, n_panels):
    """Panel x-spans from the PDF's own vector drawing commands: each panel
    draws a bottom axis line, so the bottom-most drawing row holding exactly
    n_panels long horizontal segments gives the exact axes extents. The
    boundary between neighbouring panels sits just left of the word block
    that hugs the next panel's left spine (its tick labels / axis title),
    so outside labels always stay with the panel they belong to."""
    ry0, ry1 = band
    groups = []                                   # [y, [(x0, x1), ...]]
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
        uniq = []                     # an axis line can be drawn twice
        for s in sorted(segs):
            if not uniq or abs(s[0] - uniq[-1][0]) > 1 or abs(s[1] - uniq[-1][1]) > 1:
                uniq.append(s)
        if len(uniq) == n_panels:
            cand.append((y, uniq))
    if not cand:
        raise RuntimeError(f'no drawing row with {n_panels} axis segments')
    _, axes = max(cand, key=lambda t: t[0])       # bottom-most = the axis row
    words = [w for w in page.get_text('words')
             if ry0 / ZOOM <= (w[1] + w[3]) / 2 <= ry1 / ZOOM]
    bounds = []
    for i in range(1, n_panels):
        s0, prev_e = axes[i][0], axes[i - 1][1]
        block = [w for w in words
                 if s0 - 45.0 <= w[2] <= s0 + 2.0 and w[0] >= prev_e]
        if not block:
            raise RuntimeError(f'no label block left of spine at {s0:.1f}pt')
        # grow the block along shared text rows so a long label is never
        # sliced mid-word: absorb gap words on the same y-row, stepping
        # leftward at most 12 pt at a time
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
        bounds.append(min(w[0] for w in block) - 0.5)
    for bd in bounds:                     # a boundary may never slice a word
        for w in words:
            if w[0] + 0.5 < bd < w[2] - 0.5:
                raise RuntimeError(f'boundary {bd:.1f} slices word {w[4]!r}')
    _, cols = ink_projection(pm, 0, ry0, pm.width, ry1)
    xs = [i for i, on in enumerate(cols) if on]
    ink0, ink1 = xs[0], xs[-1] + 1
    edges = [ink0] + [int(b * ZOOM) for b in bounds] + [ink1]
    return [(edges[i], edges[i + 1]) for i in range(n_panels)]


def gutter_spans(pm, band, n_panels):
    """Fallback when panels draw no axis line (e.g. filled-rect heatmaps):
    chain column ink into stacks at small gaps (a panel's labels attach to
    its body), then split at the widest stack-to-stack gutters."""
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


def split_stacked(pm, W):
    """Vertically stacked panels: panel boundaries sit where a title-height
    ink band is closely followed by a tall axes band (gap sizes there are
    smaller than every other gap, so pure thresholding cannot work)."""
    rows, _ = ink_projection(pm, 0, 0, W, pm.height)
    fine = runs(rows, 2)
    tall = int(TALL_PT * ZOOM)
    title = int(9 * ZOOM)
    close_gap = int(6 * ZOOM)
    panels, cur, seen_tall = [], 0, False
    for i, (a, b) in enumerate(fine):
        if seen_tall and (b - a) <= title and i + 1 < len(fine):
            nb = fine[i + 1]
            if (nb[0] - b) <= close_gap and (nb[1] - nb[0]) >= tall:
                panels.append((cur, a))
                cur = a
        if (b - a) >= tall:
            seen_tall = True
    panels.append((cur, pm.height))
    return panels


def export_piece(src, clip_px, out_path):
    """Clip rect in px (top-left origin) -> standalone vector PDF."""
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
    """Render the piece at 600 dpi; report worst white margin in pt."""
    doc = pymupdf.open(path)
    pm = doc[0].get_pixmap(matrix=pymupdf.Matrix(600 / 72, 600 / 72))
    doc.close()
    b = tight_bbox(pm, 0, 0, pm.width, pm.height)
    return min(b[0], b[1], pm.width - b[2], pm.height - b[3]) * 72 / 600


def check_word_coverage(src, piece_rects, cut_rects, name):
    """Every source word must fall in a piece, or in a stripped title band."""
    bad = []
    for w in src[0].get_text('words'):
        r = pymupdf.Rect(w[:4])
        c = (r.tl + r.br) / 2
        if any(c in cut for cut in cut_rects):
            continue
        if not any(c in pr for pr in piece_rects):
            bad.append(w[4])
    if bad:
        raise RuntimeError(f'{name}: words lost outside pieces: {bad}')


def split(name, src_dir, out_dir):
    n_panels, has_legend, stacked = FIGS[name]
    src_path = os.path.join(src_dir, 'figures_combined_original', name + '.pdf')
    if not os.path.exists(src_path):
        src_path = os.path.join(src_dir, name + '.pdf')
    src = pymupdf.open(src_path)
    notes = fix_source_text(src, name)
    page = src[0]
    pm = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), colorspace=pymupdf.csRGB)
    W, H = pm.width, pm.height

    rows, _ = ink_projection(pm, 0, 0, W, H)
    bands = runs(rows, int(GAP_ROW_PT * ZOOM))

    legend_band = None
    if has_legend:
        cand = bands[-1]
        if (cand[1] - cand[0]) / ZOOM > LEGEND_MAX_H_PT:
            raise RuntimeError(f'{name}: bottom band is no legend')
        legend_band = cand
        bands = bands[:-1]

    cuts, pieces = [], []
    report = []
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

    if stacked:
        spans = split_stacked(pm, W)
        if len(spans) != n_panels:
            raise RuntimeError(f'{name}: stacked walk found {len(spans)} panels')
        for sp in spans:
            emit(tight_bbox(pm, 0, sp[0], W, sp[1]))
    else:
        band = bands[0]
        if n_panels == 1:
            emit(tight_bbox(pm, 0, band[0], W, band[1]))
        else:
            try:
                spans = axes_spans(page, pm, band, n_panels)
            except RuntimeError:
                spans = gutter_spans(pm, band, n_panels)
            for sp in spans:
                emit(tight_bbox(pm, sp[0], band[0], sp[1], band[1]))

    if legend_band is not None:
        rect = tight_bbox(pm, 0, legend_band[0], W, legend_band[1])
        w, h, clip = export_piece(src, rect, os.path.join(out_dir, name + '_legend.pdf'))
        pieces.append(clip)
        mg = check_margin(os.path.join(out_dir, name + '_legend.pdf'))
        report.append((name + '_legend', w, h, mg))

    check_word_coverage(src, pieces, cuts, name)
    src.close()
    return report, notes


def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    out_dir = sys.argv[2] if len(sys.argv) > 2 else src_dir
    worst = 0.0
    for name in FIGS:
        report, notes = split(name, src_dir, out_dir)
        for note in notes:
            print(f'[{name}] fix: {note}')
        for slug, w, h, mg in report:
            worst = max(worst, mg)
            print(f'{slug:28s} {w:7.2f} x {h:6.2f} pt   margin {mg:.2f} pt')
    print(f'worst margin: {worst:.2f} pt')
    assert worst < 0.35, 'zero-margin requirement violated'
    print('word-coverage regression: all pieces complete')


if __name__ == '__main__':
    main()
