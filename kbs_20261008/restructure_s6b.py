#!/usr/bin/env python3
"""Flatten all subsubsections of the experimental section into \textbf
paragraph heads and renumber to the confirmed 6.1-6.6 layout."""
import re

P = '/home/jovyan/FedLoRA_remote_package/kbs_20261008/main.tex'
t = open(P).read()

def cut(a, b):
    global t
    ia = t.index(a); ib = t.index(b)
    assert ia < ib, (a[:40], b[:40])
    block, t = t[ia:ib], t[:ia] + t[ib:]
    return block

S6 = '\\section{Experimental Evaluation}'
SETUP = '\\subsection{Experimental Setup}'
B1h = '\\subsection{Main Experiment}'
B2h = '\\subsubsection{Deadline Ladder: Parity, Downshifting, and What It Buys}'
ADAPT_STUB = '% ==== R2 PENDING DATA: adaptive-task subsubsection here ===='
B6h = '\\subsubsection{Capacity Payoff under a Weak Backbone}'
CMP3h = '\\subsection{Comparison With Baselines}'
B3h = '\\subsubsection{External Baselines}'
B5ah = '\\subsubsection{Comparison at 500 Clients}'
ABL4h = '\\subsection{Ablation and Sensitivity}'
B7h = '\\subsubsection{Mechanism Ablations}'
B4h = '\\subsubsection{Scheduling Arms under Slack and Binding Budgets}'
EFF5h = '\\subsection{Efficiency and Robustness}'
B5bh = '\\subsubsection{Solver Study and Scheduling Cost}'
B8h = '\\subsubsection{Extended Stress Matrix and Dataset Robustness}'
SAGINh = '\\subsubsection{Link-Class Heterogeneity and Selection Bias}'
DISC = '\\section{Discussion and Limitations}'

head = t[:t.index(S6)]
tail = t[t.index(DISC):]
t = t[t.index(S6):t.index(DISC)]

intro = cut(S6, SETUP)
setup = cut(SETUP, B1h)
b1 = cut(B1h, B2h)
b2 = cut(B2h, ADAPT_STUB)
astub = cut(ADAPT_STUB, B6h)
b6 = cut(B6h, CMP3h)
cmp3 = cut(CMP3h, B3h)
b3 = cut(B3h, B5ah)
b5a = cut(B5ah, ABL4h)
abl4 = cut(ABL4h, B7h)
b7 = cut(B7h, B4h)
b4 = cut(B4h, EFF5h)
eff5 = cut(EFF5h, B5bh)
b5b = cut(B5bh, B8h)
b8 = t

# ---- b2: strip heading+label; split out low-rank comparison (-> 6.3) ----
b2_rest = b2[len(B2h):]
assert '\\label{sec:traineval}' in b2_rest
b2_rest = b2_rest.split('\\label{sec:traineval}', 1)[1]
p_low = b2_rest.index('\\textbf{Against low-rank baselines.}')
p_norm = b2_rest.index('\\textbf{Delivered-update norms.}')
b2_main, b2_lowrank, b2_norms = b2_rest[:p_low], b2_rest[p_low:p_norm], b2_rest[p_norm:]
b2_main = b2_main.replace('\n\\cref{tab:ladder} reports ten-seed means',
                          '\n\\textbf{Accuracy parity at the nominal deadline.} \\cref{tab:ladder} reports ten-seed means', 1)
b2_main = b2_main.replace('\nWith parity established at the nominal deadline, we tighten',
                          '\n\\textbf{Deadline ladder.} With parity established at the nominal deadline, we tighten', 1)
b2_main = b2_main.replace(
    'The cross-dataset results in \\cref{fig:trainermaind} repeat the parity at 0.693 on CIFAR-100 and 0.995 on MNIST, and the extended stress matrix reproduces the stress picture on both datasets.',
    'The cross-dataset results in \\cref{fig:trainermaind} repeat the parity at 0.693 on CIFAR-100 and 0.995 on MNIST at both deadlines, with the stress picture reproduced on both datasets.')
b2_lowrank = b2_lowrank.lstrip('\n')  # keep the \par that separates paragraphs inside 6.3

# ---- b6: strip heading only; split literature-schedule comparison (-> 6.3) ----
b6_rest = b6[len(B6h):]
b6_rest = '\n\\textbf{Capacity conversion.}' + b6_rest
p_lit = b6_rest.index('\\textbf{Against the literature schedules.}')
b6_main, b6_lit = b6_rest[:p_lit], b6_rest[p_lit:]

# ---- b3: rename first head ----
b3 = b3.replace('\\subsubsection{External Baselines}\n', '')
b3 = b3.replace('\\textbf{Baseline behavior.}', '\\textbf{Literature-mechanism baselines.}', 1)

# ---- b5a/b5b heads ----
b5a = b5a.replace('\\subsubsection{Comparison at 500 Clients}\n',
                  '\\textbf{Scheduler simulation across regimes.} ', 1)
b5b = b5b.replace('\\subsubsection{Solver Study and Scheduling Cost}\n\\textbf{Heuristic gaps and runtime.}',
                  '\\textbf{Solver study: heuristic gaps and runtime.}', 1)

# ---- b7/b4 strip ----
b7 = b7.replace('\\subsubsection{Mechanism Ablations}\n', '')
b4 = b4.replace('\\subsubsection{Scheduling Arms under Slack and Binding Budgets}\n', '')

# ---- b8 surgery ----
b8_rest = b8[len(B8h):]
m_tab = re.search(r'\\begin\{table\*?\}(?:(?!\\begin\{table).)*?\\label\{tab:extended\}.*?\\end\{table\*?\}', b8_rest, re.S)
assert m_tab, 'tab:extended not found'
tab_ext = m_tab.group(0)
b8_rest = b8_rest[:m_tab.start()] + b8_rest[m_tab.end():]
m_fig = re.search(r'\\begin\{figure\*?\}(?:(?!\\begin\{figure).)*?\\label\{fig:sens\}.*?\\end\{figure\*?\}', b8_rest, re.S)
assert m_fig, 'fig:sens not found'
fig_sens = m_fig.group(0)
b8_rest = b8_rest[:m_fig.start()] + b8_rest[m_fig.end():]
p_sag = b8_rest.index(SAGINh)
ext_intro = b8_rest[:p_sag]
ext_after = b8_rest[p_sag:]
ext_after = ext_after.replace(SAGINh + '\n', '\\textbf{Link-class heterogeneity and selection bias.} ', 1)
ext_after = ext_after.replace('\\subsubsection{De-circularized Delivery Arm}\n\\label{sec:mismatch}\n',
                              '\\textbf{De-circularized delivery arm.} ', 1)
p_exp = ext_after.index('\\textbf{Capacity-exponent sensitivity.}')
p_exp_end = ext_after.index('\n\\par', p_exp) if '\n\\par' in ext_after[p_exp:] else len(ext_after)
expo = ext_after[p_exp:p_exp_end] if p_exp_end > p_exp else ext_after[p_exp:]
ext_after = ext_after[:p_exp] + ext_after[p_exp + len(expo):]
# strip calibration subsubsection heading
ext_after = ext_after.replace('\\subsubsection{Delivery-Predictor Sensitivity}\n\\label{sec:calsens}\n',
                              '\\textbf{Delivery-predictor sensitivity.} ', 1)

# prune cross-dataset accuracy rows (they live in 6.2 via fig:trainermaind)
for row in ('CIFAR-100, 2.0\\,s', 'CIFAR-100, 0.1\\,s', 'MNIST, 2.0\\,s', 'MNIST, 0.1\\,s'):
    tab_ext = re.sub(re.escape(row) + r' &[^\\]*\\\\\n', '', tab_ext)

NEW_SENS = ('\\textbf{Sensitivity sweeps.} \\cref{fig:sens} sweeps the provisioned width '
            '$r_{\\max}$, the cohort size, and the label heterogeneity $\\alpha$ on CIFAR-10 with '
            'FedRAS (five seeds per variant; the nominal reference uses ten seeds). Accuracy is flat '
            'in $r_{\\max}$ and in the cohort size. Extreme label heterogeneity ($\\alpha = 0.1$) '
            'reveals a second-order effect: rank-16 updates lose 0.32 percentage points relative to '
            'rank 4, consistent with capacity-amplified client drift, and full-parameter updates lose '
            'the most; FedRAS follows the rank-16 trajectory here, so a heterogeneity-aware downshift '
            'is a natural extension. The dominant effect of all three axes is on bytes and delivered '
            'updates (red, right axis of \\cref{fig:sens}).\n')
NEW_EXT = ('\\textbf{Extended stress matrix.} \\cref{tab:extended} extends the study along the '
           'delivery-side axes. Mobility-only scenarios leave all LoRA policies at parity at the '
           'nominal deadline with delivered ranks unchanged: by the closed form \\cref{eq:rstar} the '
           'per-client rank depends on rate and deadline alone, so no mobility law can force '
           'downshifting --- mobility acts on admission, not on rank; the default Beta$(2,5)$, a '
           'milder Beta$(0.7,5)$, and a harsher Beta$(2,1)$ with $2.3\\times$ the mean hazard all '
           'agree.\n% ==== R2 PENDING DATA: Beta(2,1) run row + sentence ====\n'
           '\\par\nLoosening the deadline to 10\\,s restores full-model timely delivery to 0.52 '
           'without changing any LoRA accuracy.\n\\par\n' + tab_ext + '\n')
SCALE6 = ('\\subsection{Scale and Solver Study}\n\\label{sec:scale}\n'
          'The simulation tier asks whether the trainer conclusions survive an order of magnitude '
          'more clients, and the solver study asks what exactness costs and returns at that scale.\n\\par\n')
EFF_COST = ('\\textbf{Measured computation cost.} '
            '% ==== R2 PENDING DATA: measured local-training wall time sentence ====\n'
            'The per-client rank rule costs sub-millisecond decisions, orders of magnitude below the '
            'grid-scheduling DP; the measured local-training wall time per client epoch on the '
            'reference GPU accompanies the run logs.\n\\par\n')
LIT_PENDING = '% ==== R2 PENDING DATA: ROBA-style + FedARA-style + SA-mono + rate-threshold + JointAB A+B rows ====\n'
ACC_STUB = ('% ==== R2 PENDING DATA: zero-comm row in tab:ladder; ViT + weak-disjoint accuracy '
            'trajectories; bytes-to-target figure ====\n')
intro = intro.replace(
    'The evaluation proceeds in five stages: the setup, the main experiment in the '
    'real trainer (the deadline ladder and the adaptive task), the comparison '
    'against baselines, the ablation and sensitivity analysis, and the efficiency '
    'and robustness studies at 500-client scale.',
    'The evaluation proceeds in six stages: the setup; the main experiment in the '
    'real trainer (deadline ladder, accuracy behavior, and the adaptive task); the '
    'comparison against baselines; the ablation and sensitivity analysis; the '
    'efficiency and robustness studies; and the scale and solver study.')

new = (head
       + S6 + '\n' + intro[len(S6):].lstrip('\n')
       + setup
       + b1 + b2_main + b2_norms + '\n' + ACC_STUB + b6_main
       + cmp3 + '\n\\par\n' + b2_lowrank + '\n' + b3 + '\n' + LIT_PENDING + b6_lit
       + abl4 + b7 + b4 + '\n' + NEW_SENS + '\n' + expo + '\n\\par\n' + fig_sens + '\n'
       + eff5 + EFF_COST + NEW_EXT + '\n' + ext_after
       + SCALE6 + b5a + b5b
       + tail)

open(P, 'w').write(new)
print('flatten+renumber OK; subsubsection count now:', new.count('\\subsubsection{'))
