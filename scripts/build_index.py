#!/usr/bin/env python3
"""Build a queryable index over outputs/real_v6.

Writes
  data/runs.csv      one row per run with its configuration and terminal metrics
  data/fedras.db     SQLite database with tables `runs` and `results`

Terminal metrics follow the paper: the mean over the last ten rounds of a run.
Delivered capacity uses kappa = timely_ratio * sqrt(rank / rmax) with
G(r) = sqrt(min(r, rmax) / rmax), the capacity model of Eq. (4).

Usage
  python scripts/build_index.py [--root outputs/real_v6] [--window 10]
"""
import argparse, csv, glob, json, math, os, sqlite3, statistics as st

CONF_KEYS = ['dataset', 'deadline', 'arch', 'link_profile', 'rmax', 'dp_budget',
             'mobility_shape', 'pretrain_epochs', 'pretrain_frac', 'alpha',
             'shift_severity', 'rate_est_bias', 'mu_est_scale', 'seed',
             'clients', 'per_round', 'rounds', 'local_epochs', 'batch_size', 'lr']

def cap_gain(r, rmax):
    if rmax <= 0: return 0.0
    return math.sqrt(min(r, rmax) / rmax)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs/real_v6')
    ap.add_argument('--window', type=int, default=10)
    ap.add_argument('--out', default='data')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    runs, results = [], []
    for d in sorted(glob.glob(os.path.join(a.root, '*_seed*'))):
        cp, mp = os.path.join(d, 'config.json'), os.path.join(d, 'metrics.csv')
        if not (os.path.exists(cp) and os.path.exists(mp)): continue
        cfg = json.load(open(cp, encoding='utf-8'))
        rows = list(csv.DictReader(open(mp, encoding='utf-8')))
        if not rows: continue
        rounds = max(int(float(r['round'])) for r in rows) + 1
        lo = max(0, rounds - a.window)
        term = [r for r in rows if int(float(r['round'])) >= lo]

        run = os.path.basename(d)
        family = run.rsplit('_seed', 1)[0]
        rec = {'run': run, 'family': family}
        for k in CONF_KEYS: rec[k] = cfg.get(k)
        rec['rounds'] = rounds
        runs.append(rec)

        by = {}
        for r in rows: by.setdefault(r['method'], []).append(r)
        for m, rs in by.items():
            okset = [r for r in rs if r.get('accuracy') not in (None, '')]
            if len(okset) < 20: continue
            rds = sorted(int(float(r['round'])) for r in okset)
            last10 = set(rds[-10:])
            def nums(key, pool):
                return [float(r[key]) for r in pool if r.get(key) not in (None, '')]
            acc_v = nums('accuracy', [r for r in okset if int(float(r['round'])) in last10])
            net = [r for r in rs if int(float(r['round'])) >= 30] or rs
            tim_v = nums('timely_ratio', net); mb_v = nums('bytes_mb', net)
            pairs = [(float(r['mean_rank']), float(r['aggregated'])) for r in net
                     if r.get('mean_rank') not in (None, '') and r.get('aggregated') not in (None, '')]
            wsum = sum(w for _, w in pairs)
            rank = sum(rk * w for rk, w in pairs) / wsum if wsum > 0 else None
            timely = st.mean(tim_v) if tim_v else None
            kappa = timely * cap_gain(rank, float(cfg.get('rmax') or 16)) if (timely is not None and rank is not None) else None
            agg_v = nums('aggregated', net)
            results.append({
                'run': run, 'family': family, 'method': m,
                'seed': cfg.get('seed'), 'dataset': cfg.get('dataset'),
                'deadline': cfg.get('deadline'), 'arch': cfg.get('arch'),
                'link_profile': cfg.get('link_profile'), 'rmax': cfg.get('rmax'),
                'dp_budget': cfg.get('dp_budget'), 'mobility_shape': cfg.get('mobility_shape'),
                'accuracy': st.mean(acc_v) if acc_v else None,
                'timely': timely, 'bytes_mb': st.mean(mb_v) if mb_v else None,
                'rank': rank, 'kappa': kappa,
                'aggregated': st.mean(agg_v) if agg_v else None,
                'selected': st.mean(nums('selected', net)) if net else None,
            })

    with open(os.path.join(a.out, 'runs.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(runs[0].keys())); w.writeheader(); w.writerows(runs)

    db = os.path.join(a.out, 'fedras.db')
    if os.path.exists(db): os.remove(db)
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''CREATE TABLE runs(run TEXT PRIMARY KEY, family TEXT, dataset TEXT, deadline REAL,
        arch TEXT, link_profile TEXT, rmax INTEGER, dp_budget REAL, mobility_shape REAL,
        pretrain_epochs INTEGER, pretrain_frac REAL, alpha REAL, shift_severity REAL,
        rate_est_bias REAL, mu_est_scale REAL, seed INTEGER, clients INTEGER,
        per_round INTEGER, rounds INTEGER, local_epochs INTEGER, batch_size INTEGER, lr REAL)''')
    cur.execute('''CREATE TABLE results(id INTEGER PRIMARY KEY AUTOINCREMENT, run TEXT, family TEXT,
        method TEXT, seed INTEGER, dataset TEXT, deadline REAL, arch TEXT, link_profile TEXT,
        rmax INTEGER, dp_budget REAL, mobility_shape REAL, accuracy REAL, timely REAL,
        bytes_mb REAL, rank REAL, kappa REAL, aggregated REAL, selected REAL)''')
    for r in runs:
        cols = [k for k in r if k != 'run']
        cur.execute(f"INSERT INTO runs(run,{','.join(cols)}) VALUES(?{',?'*len(cols)})",
                    [r['run']] + [r[c] for c in cols])
    for r in results:
        cols = list(r.keys())
        cur.execute(f"INSERT INTO results({','.join(cols)}) VALUES({','.join('?'*len(cols))})",
                    [r[c] for c in cols])
    cur.execute('CREATE VIEW seed_level AS SELECT * FROM results')
    con.commit(); con.close()

    print(f"runs: {len(runs)}   results rows: {len(results)}")
    print(f"wrote {a.out}/runs.csv and {a.out}/fedras.db")

if __name__ == '__main__':
    main()
