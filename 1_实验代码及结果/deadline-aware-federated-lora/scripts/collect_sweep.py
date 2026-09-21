import argparse,glob,os
import pandas as pd
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',required=True); a=p.parse_args(); files=glob.glob(os.path.join(a.root,'**','round_metrics_seed*.csv'),recursive=True)
 if not files: raise SystemExit('No round metrics found')
 frames=[]
 for f in files:
  x=pd.read_csv(f); x.insert(0,'scenario',os.path.relpath(os.path.dirname(f),a.root)); frames.append(x)
 d=pd.concat(frames,ignore_index=True); d.to_csv(os.path.join(a.root,'all_round_metrics.csv'),index=False)
 last=d.sort_values('round').groupby(['scenario','seed','method']).tail(1)
 s=last.groupby(['scenario','method'])[['accuracy','timely_ratio','bytes_norm','mean_rank','useful_update_value']].agg(['mean','std']).reset_index(); s.to_csv(os.path.join(a.root,'sweep_summary.csv'),index=False); print(s.to_string(index=False))
if __name__=='__main__': main()
