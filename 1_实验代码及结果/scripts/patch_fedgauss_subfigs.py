#!/usr/bin/env python3
"""Rewrite the FedGauss main.tex figures into subfloat layout per the
els-cas figure template: independent per-panel PDFs, subcaptions and
captions in the LaTeX source, shared legends as separate strips, row
widths filled to 98% of \\linewidth (uniform scale per row, capped at
1.25x). Also audits table font sizes (<= 9 pt)."""
import re
import sys

PATH = sys.argv[1]
tex = open(PATH, encoding='utf-8').read()

# ---- preamble: subfig machinery, exactly as the journal's figure example
PREAMBLE = """\\usepackage[caption=false,font=footnotesize]{subfig}
\\captionsetup[subfloat]{subrefformat=simple,labelformat=simple}
\\renewcommand{\\thesubfigure}{(\\alph{subfigure})}
"""
anchor = '\\usepackage[nameinlink]{cleveref}'
assert tex.count(anchor) == 1
tex = tex.replace(anchor, PREAMBLE + anchor)

# ---- panel references -> subfigure \cref
REFS = [
    (r'(\\cref\{tab:cis\}; )\\cref\{fig:basis\} \(b\), \(c\)', r'\1\\cref{fig:basisb,fig:basisc}'),
    (r'\\cref\{fig:mainresults\} \(c\)', r'\\cref{fig:mainresultsc}'),
    (r'\\cref\{fig:basis\} \(a\)', r'\\cref{fig:basisa}'),
    (r'\\cref\{fig:scalability\} \(a\)', r'\\cref{fig:scalabilitya}'),
    (r'\\cref\{fig:robust\} \(a\)', r'\\cref{fig:robusta}'),
    (r'\\cref\{fig:robust\} \(b\)', r'\\cref{fig:robustb}'),
]
for pat, rep in REFS:
    n = len(re.findall(pat, tex))
    assert n == 1, (pat, n)
    tex = re.sub(pat, rep, tex)

# ---- figure environments
# name -> (env, subcaps, fractions, legend fraction, row target \linewidth)
FIGS = {
    'fig_main_results.pdf': ('figure*',
        ['Test Macro-F1', 'False-alarm rate', 'Membership-inference audit'],
        [.3191, .3196, .3414], .5549),
    'fig_attr.pdf': ('figure*',
        ['Basis provenance (paired)', 'UNSW-NB15', 'CICIDS2017'],
        [.2624, .3658, .3517], None),
    'fig_matrix_heatmap.pdf': ('figure*',
        ['UNSW-NB15', 'CICIDS2017'], [.4900, .4900], None),
    'fig_mechanism.pdf': ('figure',
        ['Accounting + clipping recovery ($\\beta=0.9$)',
         'Grid differences vs. validation-selected point'],
        [.8696, .9800], None),
    'fig_sensitivity.pdf': ('figure*',
        ['Privacy budget (UNSW-NB15)', 'Data heterogeneity (CICIDS2017)',
         'LoRA rank (CICIDS2017)'],
        [.3381, .3250, .3170], .4521),
    'fig_comm.pdf': ('figure*',
        ['Per-round payload', 'Cumulative volume ($M=10$)'],
        [.4838, .4962], None),
    'fig_runtime_memory.pdf': ('figure*',
        ['Wall-clock time', 'Peak memory (CICIDS2017)'],
        [.4937, .4863], .2781),
    'fig_scalability.pdf': ('figure*',
        ['Protocol length (CICIDS2017)', 'Client population (CICIDS2017)'],
        [.4966, .4834], .2891),
    'fig_robust.pdf': ('figure*',
        ['Overall ratio effect', 'Per-attack type: 10\\% malicious (UNSW-NB15)'],
        [.4898, .4902], .5715),
    'fig_paper_ablation.pdf': ('figure*',
        ['CICIDS2017', 'UNSW-NB15'], [.5789, .4011], None),
}

def base(name):        # fig_main_results.pdf -> fig:mainresults label stem
    return name[:-4]

def letters(n):
    return 'abcdefghijklmnopqrstuvwxyz'[n]

for name, (env, subcaps, fracs, legfrac) in FIGS.items():
    E = re.escape(env)
    pat = (r'\\begin\{' + E + r'\}\[pos=tbp\]\s*\n\\centering\s*\n'
           r'\\includegraphics\[[^\]]*\]\{' + re.escape(name) + r'\}\s*\n'
           r'(\\caption\{.*?\})\s*\n\s*(\\label\{[^}]*\})\s*\n\\end\{' + E + r'\}')
    m = re.search(pat, tex, re.S)
    assert m, name
    caption, label = m.group(1), m.group(2)
    lab = label[len('\\label{'):-1]
    assert re.fullmatch(r'fig:[a-z0-9]+', lab), lab
    n = len(subcaps)
    assert n == len(fracs)
    rows = []
    for i, (sc, fr) in enumerate(zip(subcaps, fracs)):
        eol = '\\hfill%' if i < n - 1 else ''
        rows.append(f'\\subfloat[{sc}]{{\\includegraphics'
                    f'[width={fr:.4f}\\linewidth]{{{base(name)}_{letters(i)}.pdf}}'
                    f'\\label{{{lab}{letters(i)}}}}}{eol}')
    body = '\n'.join(rows)
    if legfrac is not None:
        body += f'\n\\\\[4pt]\n\\includegraphics[width={legfrac:.4f}\\linewidth]' \
                f'{{{base(name)}_legend.pdf}}'
    new = (f'\\begin{{{env}}}[pos=tbp]\n\\centering\n'
           + body + f'\n{caption}\n{label}\n\\end{{{env}}}')
    tex = tex[:m.start()] + new + tex[m.end():]

# ---- table font audit: every table env must set \small/\footnotesize/\scriptsize
bad = []
for m in re.finditer(r'\\begin\{(table\*?)\}.*?\\end\{\1\}', tex, re.S):
    body = m.group(0)
    if not re.search(r'\\(small|footnotesize|scriptsize|tiny)\b', body):
        line = tex[:m.start()].count('\n') + 1
        bad.append(line)
print('tables without <=9pt size command at lines:', bad or 'none')

open(PATH, 'w', encoding='utf-8').write(tex)
print('patched', PATH)
