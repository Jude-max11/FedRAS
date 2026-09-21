import argparse, csv, os
import numpy as np

METHODS=['FedAvg','FedLoRA_r4','FedLoRA_r16','Random_rank','Reliability_only','Adaptive_rank','SA_rank']
RANKS=[2,4,8,16]; RMAX=16

def cap_gain(rr):
    # Saturating capacity of a rank-r LoRA update, consistent with the real
    # trainer (rmax=16): ranks above rmax only add payload, not capacity.
    return np.sqrt(np.minimum(rr,RMAX)/RMAX)

def size_of(rr): return 0.02*rr+0.5  # MB

def pon_of(mob, rate, rr, deadline):
    # Delivery hazard: overload = 8*size[MB]/(rate[Mbps]*deadline[s]) —
    # the MB->Mbit factor makes large ranks genuinely deadline-critical.
    return np.exp(-mob-8.0*size_of(rr)/(rate*deadline))

def run(seed, rounds, clients, ranks, budget, cap, deadline, out, mobility_shape=2.0):
    rng=np.random.default_rng(seed); rows=[]; acc={m:0.10 for m in METHODS}
    # Common random numbers: every method sees identical link states, client
    # cohorts and budgets in each round.  The server first samples participants
    # uniformly (standard cross-device FL), then admits under the byte budget —
    # so per-client rank adaptation affects what actually gets through.
    traces=[(np.clip(rng.lognormal(2.5,0.7,clients),1,40),
             rng.beta(mobility_shape,5,clients), rng.uniform(0.15,0.45,clients),
             rng.choice(clients,min(cap,clients),replace=False)) for _ in range(rounds)]
    rank_arr=np.asarray(ranks)
    for t in range(rounds):
        rate,mobility,base,cohort=traces[t]
        # delivery probability matrix: clients x ranks
        p_mat=pon_of(mobility[:,None],rate[:,None],rank_arr[None,:],deadline)
        for m in METHODS:
            if m=='FedAvg': rr=np.full(clients,64)
            elif m=='FedLoRA_r4': rr=np.full(clients,4)
            elif m=='FedLoRA_r16': rr=np.full(clients,16)
            elif m=='Random_rank': rr=rng.choice(ranks,clients)
            elif m=='Reliability_only': rr=np.full(clients,2)  # pure delivery focus
            elif m=='SA_rank':
                # SA-FedLoRA-style annealing: heat then cool, link-state oblivious
                pr=t/max(1,rounds-1); r_=2+(RMAX-2)*(2*pr) if pr<.5 else RMAX-(RMAX-2)*(2*pr-1)
                rr=np.full(clients,rank_arr[int(np.argmin(np.abs(rank_arr-r_)))])
            else:
                # Adaptive: maximize expected useful delivered capacity
                # cap_gain(r)*P_deliver(r); downshifts under mobility/deadline
                # pressure instead of collapsing to the minimum rank (the old
                # delivery-only score was monotone decreasing in r).
                rr=rank_arr[np.argmax(cap_gain(rank_arr)[None,:]*p_mat,axis=1)]
            size=size_of(rr); p=pon_of(mobility,rate,rr,deadline); utility=base*cap_gain(rr)*p
            # Budget admission inside the sampled cohort: knapsack-greedy by
            # delivered value per byte.
            order=cohort[np.argsort(-utility[cohort]/size[cohort])]; sel=[]; used=0
            for i in order:
                if len(sel)>=cap or used+size[i]>budget: continue
                sel.append(i); used+=size[i]
            timely=float(np.mean(p[sel])) if sel else 0.
            # Total delivered value normalized by the cap: delivering more
            # useful updates within the budget is more learning signal.
            gain=float(np.sum(utility[sel])/cap) if sel else 0.
            noise=np.random.default_rng(seed*100000+t*100+METHODS.index(m)).normal(0,0.0005)
            # Saturating accuracy proxy: delivered useful value drives acc toward
            # its equilibrium instead of the old unbounded linear recursion.
            acc[m]=float(np.clip(acc[m]+0.08*gain*(1-acc[m])-0.0005*acc[m]+noise,0,1))
            rows.append([seed,t,m,acc[m],timely,used,len(sel),float(np.mean(rr[sel])) if sel else 0.,gain])
    os.makedirs(out,exist_ok=True)
    with open(os.path.join(out,f'round_metrics_seed{seed}.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(['seed','round','method','accuracy','timely_ratio','bytes_norm','selected','mean_rank','useful_update_value']); w.writerows(rows)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='outputs/fedlora'); p.add_argument('--seeds',default='0,1,2,3,4'); p.add_argument('--rounds',type=int,default=150); p.add_argument('--clients',type=int,default=500); p.add_argument('--budget',type=float,default=300); p.add_argument('--cap',type=int,default=25); p.add_argument('--deadline',type=float,default=1.0); p.add_argument('--mobility-shape',type=float,default=2.0); a=p.parse_args();
    for s in map(int,a.seeds.split(',')): run(s,a.rounds,a.clients,[2,4,8,16],a.budget,a.cap,a.deadline,a.out,a.mobility_shape)
if __name__=='__main__': main()
