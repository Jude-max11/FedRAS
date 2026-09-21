import argparse, os, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='Run the complete FedLoRA paper experiment matrix')
    p.add_argument('--root',default='outputs/paper_sweep'); p.add_argument('--rounds',type=int,default=150)
    p.add_argument('--clients',type=int,default=500); p.add_argument('--seeds',default='0,1,2,3,4')
    a=p.parse_args(); root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
    # Budgets chosen to actually bind: cap=25 clients x max payload (~1.8MB
    # for FedAvg rank-64) tops out near 45MB, so the old 150/300/600 settings
    # never constrained selection.
    jobs=[('main_cifar10',a.clients,300,25,1.0),('cifar100_proxy',a.clients,300,25,1.0),('mnist_scalability',1000,60,50,1.0)]
    for name,clients,budget,cap,deadline in jobs:
        out=root/name; cmd=[sys.executable,'scripts/run_fedlora.py','--out',str(out),'--seeds',a.seeds,'--rounds',str(a.rounds),'--clients',str(clients),'--budget',str(budget),'--cap',str(cap),'--deadline',str(deadline)]
        print('RUN',name); subprocess.run(cmd,check=True)
    settings=[('deadline_025',0.25,300,2.0),('deadline_050',0.5,300,2.0),('deadline_200',2.0,300,2.0),('budget_low',1.0,12,2.0),('budget_high',1.0,40,2.0),('mobility_stress',1.0,300,0.7)]
    for name,deadline,budget,mshape in settings:
        out=root/'stress'/name; cmd=[sys.executable,'scripts/run_fedlora.py','--out',str(out),'--seeds',a.seeds,'--rounds',str(a.rounds),'--clients',str(a.clients),'--budget',str(budget),'--cap','25','--deadline',str(deadline),'--mobility-shape',str(mshape)]
        print('RUN',name); subprocess.run(cmd,check=True)
    print('Completed sweep. Run scripts/collect_sweep.py --root',root)
if __name__=='__main__': main()
