"""Delivered-capacity accounting: the single-currency metric from trainer logs.

dc_meas(round) = aggregated x update_norm  -- exact identity (mean of delivered
                 increments x number delivered = total delivered update mass);
dc_law(round)  = aggregated x sqrt(mean_rank / rmax) -- the paper's capacity
                 model in normalized units, as a model-side cross-check.
Both accumulate across rounds; dc per attempted MB; terminal window is rounds
30-39 as everywhere in the R2 protocol. Only complete runs (round 39 present
for the method) enter the terminal summary, so in-flight families are skipped
safely.

Writes outputs/journal_stats/delivered_capacity.csv (per round) and
delivered_capacity_terminal.csv (per family x method with paired Wilcoxon on
terminal dc_meas for FedRAS vs fixed ranks). Writes nothing else.
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, 'outputs', 'real_fedlora_v4')
OUT = os.path.join(ROOT, 'outputs', 'journal_stats')
TERMINAL_LO, TERMINAL_HI = 30, 39
PAIRS = [('Adaptive_rank', 'FedLoRA_r16'), ('Adaptive_rank', 'FedLoRA_r4'),
         ('FedLoRA_r16', 'FedLoRA_r4')]


def scan_runs():
    """Yield (run, family, seed, config) for every run dir with metrics.csv."""
    for d in sorted(glob.glob(f'{RUNS}/*_seed*')):
        if not os.path.isfile(os.path.join(d, 'metrics.csv')):
            continue
        m = re.match(r'^(.*?)_seed(\d+)$', os.path.basename(d))
        if not m or os.path.basename(d).startswith('warm_'):
            continue
        try:
            cfg = json.load(open(os.path.join(d, 'config.json')))
        except Exception:
            continue
        yield d, m.group(1), int(m.group(2)), cfg


def run_frame(d, family, seed, cfg):
    try:
        df = pd.read_csv(os.path.join(d, 'metrics.csv'))
    except pd.errors.EmptyDataError:  # in-flight run, header not flushed yet
        return None
    if df.empty:
        return None
    rmax = float(cfg.get('rmax', 16))
    df['run'] = os.path.basename(d)
    df['family'] = family
    df['deadline'] = cfg.get('deadline')
    df['pretrain_epochs'] = cfg.get('pretrain_epochs')
    df['backbone_acc'] = cfg.get('backbone_acc')
    df['rmax'] = rmax
    df['dc_meas'] = df['aggregated'] * df['update_norm']
    df['dc_law'] = df['aggregated'] * np.sqrt(df['mean_rank'] / rmax)
    df['dc_per_mb'] = df['dc_meas'] / df['bytes_mb']
    df = df.sort_values(['method', 'round'])
    df['cum_dc_meas'] = df.groupby('method')['dc_meas'].cumsum()
    return df


def terminal_summary(all_df):
    done = all_df[all_df['round'] == TERMINAL_HI]['run'].unique()
    d = all_df[all_df['run'].isin(done)]
    t = d[(d['round'] >= TERMINAL_LO) & (d['round'] <= TERMINAL_HI)]
    g = (t.groupby(['family', 'method', 'seed'])
           .agg(dc_meas=('dc_meas', 'mean'), dc_law=('dc_law', 'mean'),
                timely=('timely_ratio', 'mean'), bytes=('bytes_mb', 'mean'),
                rank=('mean_rank', 'mean'), acc=('accuracy', 'mean'),
                backbone_acc=('backbone_acc', 'mean'))
           .reset_index())
    rows, tests = [], []
    for fam, gg in g.groupby('family'):  # backbone_acc varies per seed (independent pretrains); report its mean
        for m, mm in gg.groupby('method'):
            rows.append(dict(family=fam, backbone_acc=mm['backbone_acc'].mean(), method=m, n=len(mm),
                             dc_meas=mm.dc_meas.mean(), dc_meas_sd=mm.dc_meas.std(),
                             dc_law=mm.dc_law.mean(), timely=mm['timely'].mean(),
                             bytes=mm['bytes'].mean(), rank=mm['rank'].mean(),
                             acc=mm['acc'].mean()))
        wide = gg.pivot(index='seed', columns='method', values='dc_meas')
        for a, b in PAIRS:
            if a in wide and b in wide:
                pair = wide[[a, b]].dropna()
                if len(pair) >= 5:
                    try:
                        p = wilcoxon(pair[a], pair[b]).pvalue
                    except Exception:
                        p = np.nan
                    tests.append(dict(family=fam, pair=f'{a} vs {b}', n=len(pair),
                                      dc_a=pair[a].mean(), dc_b=pair[b].mean(),
                                      diff_pct=100 * (pair[a] / pair[b] - 1).mean(),
                                      p_wilcoxon=p))
    test_cols = ['family', 'pair', 'n', 'dc_a', 'dc_b', 'diff_pct', 'p_wilcoxon']
    return pd.DataFrame(rows), pd.DataFrame(tests, columns=test_cols)


def main():
    frames = [run_frame(*args) for args in scan_runs()]
    frames = [f for f in frames if f is not None and not f.empty]
    all_df = pd.concat(frames, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    all_df.to_csv(os.path.join(OUT, 'delivered_capacity.csv'), index=False)

    summ, tests = terminal_summary(all_df)
    summ.to_csv(os.path.join(OUT, 'delivered_capacity_terminal.csv'), index=False)
    tests.to_csv(os.path.join(OUT, 'delivered_capacity_tests.csv'), index=False)

    show = summ[summ.family.isin(['cifar10', 'cifar10_dl0.1', 'weak', 'weak_dl0.1',
                                  'weak5', 'weak5_dl0.1', 'weakdisj_dl0.1'])]
    cols = ['family', 'method', 'n', 'dc_meas', 'dc_law', 'timely', 'bytes', 'rank', 'acc']
    print('== terminal-window (rounds 30-39) delivered capacity, key families ==')
    print(show[cols].round(4).to_string(index=False))
    print('\n== paired Wilcoxon on terminal dc_meas (key families) ==')
    tk = tests[tests.family.isin(show.family)]
    print(tk.round(4).to_string(index=False) if not tk.empty else '(none yet)')
    print(f'\nwrote {OUT}/delivered_capacity.csv ({len(all_df)} rows), '
          f'delivered_capacity_terminal.csv, delivered_capacity_tests.csv')


if __name__ == '__main__':
    main()
