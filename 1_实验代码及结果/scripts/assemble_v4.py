"""Assemble v4 real-byte trainer runs into all_metrics.csv.

Scans <root>/*/metrics.csv, tags each row with the dataset/variant from the
directory name and config, and writes <root>/all_metrics.csv. The v4 metrics
schema is: seed, round, method, accuracy, timely_ratio, bytes_mb, mean_rank
(delivered-update mean; empty on zero-delivery rounds), selected, aggregated
(delivered count = correct pooling weight), update_norm.
"""
import argparse, glob, json, os, re
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='outputs/real_fedlora_v4')
    a = p.parse_args()
    frames = []
    for f in sorted(glob.glob(os.path.join(a.root, '*', 'metrics.csv'))):
        x = pd.read_csv(f)
        cfg = json.load(open(os.path.join(os.path.dirname(f), 'config.json')))
        run = os.path.basename(os.path.dirname(f))
        # Tag by the run-directory name minus the seed suffix. Directory names
        # carry the full variant identity (cifar10_dl0.1, dp, dpsame_dl0.1,
        # cifar10_ext, ...); the old dataset-prefix regex silently merged the
        # dp/dpsame arms into the cifar10 tag.
        ds = re.sub(r'_seed\d+$', '', run)
        x['dataset'] = ds
        x['run'] = run
        frames.append(x)
    if not frames:
        raise SystemExit('no metrics found under ' + a.root)
    d = pd.concat(frames, ignore_index=True)
    d['mean_rank'] = pd.to_numeric(d['mean_rank'], errors='coerce')  # '' -> NaN
    d.to_csv(os.path.join(a.root, 'all_metrics.csv'), index=False)
    print(f'{len(frames)} runs, {len(d)} rows -> {os.path.join(a.root, "all_metrics.csv")}')
    print(d.groupby('dataset').agg(rows=('round', 'size'), seeds=('seed', 'nunique'),
                                   methods=('method', 'nunique')).to_string())


if __name__ == '__main__':
    main()
