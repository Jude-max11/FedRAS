"""Aggregate real-trainer FedLoRA runs (v2) into summary tables and figures.

Scans <root>/*/metrics.csv (one directory per seed/dataset run), merges them
into all_metrics.csv, writes summary.csv (per dataset x method mean/std over
seeds of the final and best rounds) and renders accuracy/efficiency figures.
"""
import argparse, glob, json, os, re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='outputs/real_fedlora_v2')
    p.add_argument('--compare', default='outputs/all_real_metrics.csv',
                   help='optional v1 metrics csv for before/after comparison')
    a = p.parse_args()
    frames = []
    for f in sorted(glob.glob(os.path.join(a.root, '*', 'metrics.csv'))):
        x = pd.read_csv(f)
        cfg = json.load(open(os.path.join(os.path.dirname(f), 'config.json')))
        run = os.path.basename(os.path.dirname(f))
        ds = cfg.get('dataset', '?')
        mv = re.match(rf"{ds}(.*)_seed\d+$", run)  # variant suffix: _dl0.25, _mob0.7, _ext, _rmax8, ...
        if mv and mv.group(1):
            ds = ds + mv.group(1)
        x['dataset'] = ds; x['run'] = run
        frames.append(x)
    if not frames:
        raise SystemExit(f'no metrics.csv found under {a.root}')
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(os.path.join(a.root, 'all_metrics.csv'), index=False)

    last = d.sort_values('round').groupby(['dataset', 'seed', 'method']).tail(1)
    best = d.loc[d.groupby(['dataset', 'seed', 'method'])['accuracy'].idxmax()]
    cols = ['accuracy', 'timely_ratio', 'bytes_norm', 'mean_rank', 'update_norm']
    fin = last.groupby(['dataset', 'method'])[cols].agg(['mean', 'std']).round(4)
    fin.columns = ['final_' + '_'.join(c) for c in fin.columns]
    bst = best.groupby(['dataset', 'method'])['accuracy'].agg(['mean', 'std']).round(4)
    bst.columns = ['best_accuracy_mean', 'best_accuracy_std']
    s = fin.join(bst).reset_index()
    s.to_csv(os.path.join(a.root, 'summary.csv'), index=False)
    print(s.to_string(index=False))

    figs = os.path.join(a.root, 'figures'); os.makedirs(figs, exist_ok=True)
    for ds, g in d.groupby('dataset'):
        plt.figure(figsize=(7, 4.5))
        LABEL = {'FedAvg_full': 'FedAvg (full)', 'FedLoRA_r4': 'FedLoRA r=4',
                 'FedLoRA_r16': 'FedLoRA r=16', 'Random_rank': 'Random rank',
                 'Adaptive_rank': 'FedRAS (ours)', 'Reliability_only': 'Reliability-only'}
        for m, h in g.groupby('method'):
            z = h.groupby('round')['accuracy'].agg(['mean', 'std'])
            plt.plot(z.index, z['mean'], label=LABEL.get(m, m))
            plt.fill_between(z.index, z['mean'] - z['std'], z['mean'] + z['std'], alpha=.15)
        plt.xlabel('Round'); plt.ylabel('Test accuracy'); plt.title(f'{ds}: accuracy vs round (mean±std over seeds)')
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(figs, f'{ds}_accuracy.png'), dpi=180); plt.close()

        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        z = g.groupby('method').agg(acc=('accuracy', 'max'), byt=('bytes_norm', 'mean')).reindex(
            ['FedLoRA_r4', 'FedLoRA_r16', 'Random_rank', 'Adaptive_rank', 'FedAvg_full']).dropna()
        ax1.bar(range(len(z)), z['acc'], color=['#7f8c8d', '#2980b9', '#8e44ad', '#27ae60', '#c0392b'][:len(z)])
        ax1.set_xticks(range(len(z))); ax1.set_xticklabels(z.index, rotation=20, ha='right')
        ax1.set_ylabel('Best test accuracy')
        ax2 = ax1.twinx(); ax2.plot(range(len(z)), z['byt'], 'ko--', label='bytes/round')
        ax2.set_ylabel('Normalized bytes per round')
        ax1.set_title(f'{ds}: accuracy vs communication')
        fig.tight_layout(); fig.savefig(os.path.join(figs, f'{ds}_accuracy_bytes.png'), dpi=180); plt.close(fig)

    if a.compare and os.path.exists(a.compare):
        v1 = pd.read_csv(a.compare)
        f1 = v1.sort_values('round').groupby(['dataset', 'method'])['accuracy'].tail(1)
        v1f = v1.sort_values('round').groupby(['dataset', 'method']).tail(1).groupby(['dataset', 'method'])['accuracy'].mean()
        v2f = last.groupby(['dataset', 'method'])['accuracy'].mean()
        rows = []
        for (ds, m), new in v2f.items():
            old = v1f.get((ds, m))
            if old is not None:
                rows.append({'dataset': ds, 'method': m, 'v1_final_accuracy': round(old, 4),
                             'v2_final_accuracy': round(new, 4), 'gain': round(new - old, 4)})
        if rows:
            cmp_df = pd.DataFrame(rows)
            cmp_df.to_csv(os.path.join(a.root, 'v1_v2_comparison.csv'), index=False)
            print('\n=== v1 (MLP backbone) vs v2 (CNN+recipe) ===')
            print(cmp_df.to_string(index=False))

if __name__ == '__main__':
    main()
