from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS={"gcdp":"#0072B2","greedy_repair":"#D55E00","two_stage":"#CC79A7","reactive":"#009E73","fixed_secagg":"#E69F00"}
ap=argparse.ArgumentParser(); ap.add_argument("--root", required=True); a=ap.parse_args(); root=Path(a.root)
x=pd.read_csv(root/"all_results.csv")
plt.style.use("seaborn-v0_8-whitegrid")
for variable, g in x.groupby("variable"):
    fig,axes=plt.subplots(1,3,figsize=(9,3.0))
    for method,m in g.groupby("method"):
        z=m.groupby("value").agg(obj=("mean_objective","mean"),obj_s=("mean_objective","std"),acc=("final_accuracy","mean"),acc_s=("final_accuracy","std"),budget=("mean_budget","mean"),budget_s=("mean_budget","std")).reset_index()
        for ax,col,err,ylabel in [(axes[0],"obj","obj_s","Modeled utility"),(axes[1],"acc","acc_s","Final accuracy"),(axes[2],"budget","budget_s","Final budget")]:
            ax.errorbar(z.value,z[col],yerr=z[err].fillna(0),label=method.replace('_',' '),color=COLORS.get(method),marker='o',lw=1.6,capsize=2)
            ax.set_xlabel(variable.replace('_',' ')); ax.set_ylabel(ylabel); ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(fontsize=7,frameon=False); fig.tight_layout(); fig.savefig(root/f"{variable}_tradeoff.png",dpi=300,bbox_inches="tight"); plt.close(fig)
