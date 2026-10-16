#!/usr/bin/env python3
"""Six-stage restructuring of the experimental section (reviewer round 2).
Moves content blocks by unique anchors; verifies every anchor is found."""
import re, sys

P = '/home/jovyan/FedLoRA_remote_package/kbs_20261008/main.tex'
t = open(P).read()
orig = t

def cut(a, b):
    """return (block, rest) where block spans from anchor a (inclusive) to anchor b (exclusive)."""
    global t
    ia = t.index(a); ib = t.index(b)
    assert ia < ib, (a, b)
    block, t = t[ia:ib], t[:ia] + t[ib:]
    return block

S6 = '\\section{Experimental Evaluation}'
SETUP = '\\subsection{Experimental Setup}'
B1h = '\\subsection{Comparison With Baselines}'
B2h = '\\subsubsection{Accuracy Parity and Deadline Stress}'
B3h = '\\subsubsection{External Baselines}'
B4h = '\\subsubsection{Scheduling Variants}'
B5h = '\\subsubsection{Comparison at 500 Clients}'
B5split = '\\textbf{Heuristic gaps and runtime.}'
B6h = '\\subsection{Main Results}'
B7h = '\\subsection{Ablation Study}'
B8h = '\\subsection{Robustness and Sensitivity}'
DISC = '\\section{Discussion and Limitations}'

head = t[:t.index(S6)]
tail = t[t.index(DISC):]
sec = t[t.index(S6):t.index(DISC)]

t = sec
intro = cut(S6, SETUP)
setup = cut(SETUP, B1h)
b1 = cut(B1h, B2h)
b2 = cut(B2h, B3h)
b3 = cut(B3h, B4h)
b4 = cut(B4h, B5h)
b5 = cut(B5h, B6h)
b6 = cut(B6h, B7h)
b7 = cut(B7h, B8h)
b8 = t  # remainder of the section

# split b5 into the 500-client comparison and the solver-study paragraphs
i = b5.index(B5split)
b5a, b5b = b5[:i], b5[i:]

# ---- transforms ----
intro = intro.replace(
    'The evaluation compares FedRAS against baselines in the real trainer and at 500-client scale, '
    'states the main results, ablates the components, and probes robustness and sensitivity.',
    'The evaluation proceeds in five stages: the setup, the main experiment in the real trainer '
    '(the deadline ladder and the adaptive task), the comparison against baselines, the ablation '
    'and sensitivity analysis, and the efficiency and robustness studies at 500-client scale.')

b1 = ('\\subsection{Main Experiment}\n\\label{sec:cmp}\n'
      'The main experiment answers two questions in the real trainer: how the scheduler behaves as '
      'the deadline tightens, and whether the communication it schedules buys measurable learning '
      'on tasks whose backbone genuinely needs adaptation.\n\\par\n')

b2 = b2.replace('\\subsubsection{Accuracy Parity and Deadline Stress}',
                '\\subsubsection{Deadline Ladder: Parity, Downshifting, and What It Buys}')

ADAPTIVE_STUB = ('% ==== R2 PENDING DATA: adaptive-task subsubsection here ====\n'
                 '% (shift task CNN + ViT: zero-comm control, accuracy gains, bytes/time-to-target)\n\n')

b6 = b6.replace('\\subsection{Main Results}', '\\subsubsection{Capacity Payoff under a Weak Backbone}')

CMP3 = ('\\subsection{Comparison With Baselines}\n\\label{sec:cmp2}\n'
        'The comparison places FedRAS against fixed-rank and full-model policies, external schedules '
        'in both stylized and faithful variants, simple network-aware rules, and the heuristics at '
        '500-client scale, under identical partitions, link traces, and seed protocol.\n\\par\n')

ABL4 = ('\\subsection{Ablation and Sensitivity}\n\\label{sec:abl}\n'
        'The ablation first removes each mechanism of the architecture and then varies the solver '
        'regime, separating the value of candidate-pool widening from the value of exact solving '
        'under a binding byte budget.\n\\par\n')
b7 = b7.replace('\\subsection{Ablation Study}', '\\subsubsection{Mechanism Ablations}')
b4 = b4.replace('\\subsubsection{Scheduling Variants}',
                '\\subsubsection{Scheduling Arms under Slack and Binding Budgets}')
BIND_STUB = ('% ==== R2 PENDING DATA: binding-budget 0.4 MB arms (DP vs value/vpb greedy) ====\n\n')

EFF5 = ('\\subsection{Efficiency and Robustness}\n\\label{sec:eff}\n'
        'The final stage asks whether the architecture is deployable: what its scheduling and local '
        'computation cost, how it behaves under heterogeneous link classes and correlated data skew, '
        'under a structurally mismatched channel, under predictor error, and across datasets and '
        'stress axes.\n\\par\n')
b5b = ('\\subsubsection{Solver Study and Scheduling Cost}\n' + b5b)
b8 = b8.replace('\\subsection{Robustness and Sensitivity}', '', 1)
b8 = b8.replace('\\subsubsection{Extended Stress Matrix}',
                '\\subsubsection{Extended Stress Matrix and Dataset Robustness}')
b8 = b8.replace('\\textbf{SAGIN link profile.}',
                '\\subsubsection{Link-Class Heterogeneity and Selection Bias}\n'
                '\\textbf{SAGIN link profile.}')
BIAS_STUB = ('% ==== R2 PENDING DATA: correlated-skew bias results paragraph here ====\n\n')

new = (head + S6 + intro.replace(S6, S6 + '\n', 1) if False else None)
# assemble explicitly
new = (head
       + S6 + '\n' + intro[len(S6):].lstrip('\n')
       + setup
       + b1 + b2 + ADAPTIVE_STUB + b6
       + CMP3 + b3 + b5a
       + ABL4 + b7 + BIND_STUB + b4
       + EFF5 + b5b + BIAS_STUB + b8
       + tail)

open(P, 'w').write(new)
print('restructure OK; size', len(orig), '->', len(new))
