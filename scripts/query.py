#!/usr/bin/env python3
"""Query the FedRAS result index.

The index is built by scripts/build_index.py into data/fedras.db.

Examples
  python scripts/query.py --list-datasets
  python scripts/query.py --list-methods
  python scripts/query.py --dataset cifar10 --deadline 0.1
  python scripts/query.py --dataset cifar10 --deadline 0.1 --method FedRAS
  python scripts/query.py --family p3_fig5 --metric accuracy
  python scripts/query.py --sql "SELECT dataset, deadline, method, AVG(accuracy) a FROM results GROUP BY 1,2,3 LIMIT 10"
"""
import argparse, os, sqlite3, statistics as st, sys

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fedras.db')
ALIAS = {'fedras': 'Adaptive_rank', 'adaptive_rank': 'Adaptive_rank',
         'fedlora-4': 'FedLoRA_r4', 'fedlora-16': 'FedLoRA_r16',
         'random': 'Random_rank', 'sa-fedlora': 'SA_mono', 'oort': 'Oort_utility',
         'roba': 'ROBA_style'}
METRICS = ['accuracy', 'timely', 'bytes_mb', 'rank', 'kappa', 'aggregated', 'selected']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=DB)
    ap.add_argument('--dataset'); ap.add_argument('--method'); ap.add_argument('--family')
    ap.add_argument('--deadline', type=float); ap.add_argument('--arch'); ap.add_argument('--link-profile')
    ap.add_argument('--metric', default='accuracy', choices=METRICS)
    ap.add_argument('--by', default='method', help='group by: method, dataset, deadline, family')
    ap.add_argument('--sql')
    ap.add_argument('--list-datasets', action='store_true'); ap.add_argument('--list-methods', action='store_true')
    ap.add_argument('--schema', action='store_true')
    a = ap.parse_args()

    if not os.path.exists(a.db):
        sys.exit(f"index not found: {a.db}\nrun: python scripts/build_index.py")
    con = sqlite3.connect(a.db); con.row_factory = sqlite3.Row; cur = con.cursor()

    if a.schema:
        for r in cur.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view')"):
            print(f"{r['type']:5s} {r['name']}")
        print("\nresults columns:", [d[1] for d in cur.execute('PRAGMA table_info(results)')])
        return
    if a.list_datasets:
        for r in cur.execute("SELECT dataset, COUNT(DISTINCT run) n FROM results GROUP BY dataset ORDER BY 1"):
            print(f"  {r['dataset']:10s} {r['n']} runs")
        return
    if a.list_methods:
        for r in cur.execute("SELECT method, COUNT(*) n FROM results GROUP BY method ORDER BY 2 DESC"):
            print(f"  {r['method']:16s} {r['n']} rows")
        return
    if a.sql:
        rows = list(cur.execute(a.sql))
        if not rows: print("no rows"); return
        cols = rows[0].keys()
        print(' | '.join(f"{c:>14s}" for c in cols))
        for r in rows: print(' | '.join(f"{r[c]!s:>14.14s}" for c in cols))
        return

    where, args = [], []
    if a.dataset: where.append('dataset = ?'); args.append(a.dataset)
    if a.deadline is not None: where.append('ABS(deadline - ?) < 1e-9'); args.append(a.deadline)
    if a.arch: where.append('arch = ?'); args.append(a.arch)
    if a.link_profile: where.append('link_profile = ?'); args.append(a.link_profile)
    if a.family: where.append('family LIKE ?'); args.append(a.family + '%')
    if a.method:
        m = ALIAS.get(a.method.lower(), a.method)
        where.append('method = ?'); args.append(m)
    q = f"SELECT {a.by} g, {a.metric} v FROM results"
    if where: q += ' WHERE ' + ' AND '.join(where)
    rows = [r for r in cur.execute(q, args) if r['v'] is not None]
    if not rows: sys.exit('no rows match')

    agg = {}
    for r in rows: agg.setdefault(r['g'], []).append(float(r['v']))
    print(f"{a.by:14s} {'n':>4s} {a.metric:>10s} {'sd':>9s}")
    for g in sorted(agg, key=lambda k: -st.mean(agg[k])):
        vs = agg[g]
        sd = st.stdev(vs) if len(vs) > 1 else 0.0
        print(f"{str(g):14.14s} {len(vs):4d} {st.mean(vs):10.4f} {sd:9.4f}")

if __name__ == '__main__':
    main()
