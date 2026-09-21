import argparse,glob,os
import pandas as pd
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',required=True); a=p.parse_args(); fs=glob.glob(os.path.join(a.root,'round_metrics_seed*.csv')); df=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True); last=df.sort_values('round').groupby(['seed','method']).tail(1); s=last.groupby('method')[['accuracy','timely_ratio','bytes_norm','mean_rank','useful_update_value']].agg(['mean','std']).reset_index(); s.to_csv(os.path.join(a.root,'summary.csv'),index=False); print(s.to_string(index=False))
if __name__=='__main__': main()
