"""Solver-only optimality-gap and scalability study (no FL training required)."""
from pathlib import Path
import numpy as np
import pandas as pd
from mprbps.model import ActionEstimate, Policy
from mprbps.solvers import solve_gcdp, solve_greedy_repair

rng=np.random.default_rng(2026); rows=[]
npol=Policy("n",0,32,secagg=False); spol=Policy("s",.5,8,secagg=True)
for n in [50,100,200,500]:
  for k in [5,10,15,25]:
    if k>n: continue
    m=min(5,k)
    for seed in range(10):
      grid={}
      for h in [0]+list(range(m,k+1)):
        grid[h]={}
        for cid in range(n):
          base=rng.uniform(-.1,.4); acts=[]
          for p in [npol,spol]:
            u=base+rng.normal(0,.04)-(.0015*h*h if p.secagg else 0)
            acts.append(ActionEstimate(cid,p,h,float(u),.8,1,.2,.5,.3,.1,100,not p.secagg or h>=m))
          grid[h][cid]=acts
      exact=solve_gcdp(grid,k,m); greedy=solve_greedy_repair(grid,k,m)
      rows.append({"clients":n,"cap":k,"trial":seed,"gcdp_ms":exact.runtime_ms,"greedy_ms":greedy.runtime_ms,"gcdp_objective":exact.objective,"greedy_objective":greedy.objective,"relative_gap":(exact.objective-greedy.objective)/max(abs(exact.objective),1e-12)})
out=Path("outputs/scalability"); out.mkdir(parents=True,exist_ok=True); x=pd.DataFrame(rows); x.to_csv(out/"scalability.csv",index=False)
x.groupby(["clients","cap"]).agg(gap=("relative_gap","mean"),gcdp_ms=("gcdp_ms","mean"),greedy_ms=("greedy_ms","mean")).to_csv(out/"summary.csv")
print(x.groupby(["clients","cap"])[["relative_gap","gcdp_ms","greedy_ms"]].mean().to_string())
