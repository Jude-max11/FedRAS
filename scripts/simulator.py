"""v4 scheduler simulator (real-byte era).

Changes vs v3 (kept in backups/scripts_backup_20260917_pre_realbytes.zip):
- Payloads are the real serialized B-only uploads (size(r)=r*5632/1e6 MB);
  FedAvg carries its true full-update payload (5.286 MB), not a rank-64
  stand-in. --payload-model stylized restores the old s(r)=0.02r+0.5 MB
  protocol with rank-64 FedAvg for the normalized-cost sensitivity cell only.
- Budget admission uses the SAME conservative ceil discretization on a
  0.01 MB grid as the trainer and the solver study; budgets are real MB.
- The 'accuracy' column remains a bounded scheduling proxy derived from the
  delivered useful value (not dataset accuracy); with --payload-model real
  and saturating G(r), the full-update baseline receives the same capacity
  credit as rank 16 at 59x the payload - a disclosed modeling assumption.
"""
import argparse, csv, math, os
import numpy as np

METHODS=['FedAvg','FedLoRA_r4','FedLoRA_r16','Random_rank','Reliability_only','Adaptive_rank','SA_mono','ROBA_style']
RANKS=[2,4,8,16]; RMAX=16
BYTES_PER_RANK=5632          # B-only, FP32 (see journal_model_audit.py)
FULL_MB=5.286                # real FedAvg_full transmitted update (FP32)
GRID=0.01                    # MB per budget unit (ceil everywhere)

def size_of(rr, model='real'):
    rr = np.asarray(rr, dtype=float)
    if model == 'real':
        return np.where(rr < 64, rr*BYTES_PER_RANK/1e6, FULL_MB)
    return 0.02*rr+0.5      # stylized protocol (normalized-cost sensitivity only)

def units_of(mb):            # conservative ceil grid units
    return max(1, int(math.ceil(mb/GRID - 1e-9)))

def cap_gain(rr):
    # Saturating capacity of a rank-r LoRA update, consistent with the real
    # trainer (rmax=16): ranks above rmax only add payload, not capacity.
    return np.sqrt(np.minimum(rr,RMAX)/RMAX)

def pon_of(mob, rate, rr, deadline, model='real'):
    # Delivery hazard: overload = 8*size[MB]/(rate[Mbps]*deadline[s]).
    return np.exp(-mob-8.0*size_of(rr,model)/(rate*deadline))

def run(seed, rounds, clients, ranks, budget, cap, deadline, out, mobility_shape=2.0, payload_model='real'):
    rng=np.random.default_rng(seed); rows=[]; acc={m:0.10 for m in METHODS}
    rank_arr=np.asarray(ranks)
    sizes_real=rank_arr*BYTES_PER_RANK/1e6
    costs=np.maximum(1,np.ceil(sizes_real/GRID-1e-9)).astype(int)   # grid units per rank
    budget_u=units_of(budget)
    # Common random numbers: every method sees identical link states, client
    # cohorts and budgets in each round.  The server first samples participants
    # uniformly (standard cross-device FL), then admits under the byte budget —
    # so per-client rank adaptation affects what actually gets through.
    traces=[(np.clip(rng.lognormal(2.5,0.7,clients),1,40),
             rng.beta(mobility_shape,5,clients), rng.uniform(0.15,0.45,clients),
             rng.choice(clients,min(cap,clients),replace=False)) for _ in range(rounds)]
    for t in range(rounds):
        rate,mobility,base,cohort=traces[t]
        # delivery probability matrix: clients x ranks
        p_mat=pon_of(mobility[:,None],rate[:,None],rank_arr[None,:],deadline,payload_model)
        fed_avg_size=FULL_MB if payload_model=='real' else size_of(64,'stylized')
        fed_avg_pon=np.exp(-mobility-8.0*fed_avg_size/(rate*deadline))
        for m in METHODS:
            if m=='FedAvg': rr=np.full(clients,64)
            elif m=='FedLoRA_r4': rr=np.full(clients,4)
            elif m=='FedLoRA_r16': rr=np.full(clients,16)
            elif m=='Random_rank': rr=rng.choice(ranks,clients)
            elif m=='Reliability_only': rr=np.full(clients,2)  # pure delivery focus
            elif m=='SA_mono':
                # SA-FedLoRA (faithful): hold the full width for the first 20% of
                # rounds, then shrink linearly to the narrowest rank; oblivious to
                # the link state, exactly as in the real trainer.
                pr=t/max(1,rounds-1); warm=.2
                r_=RMAX if pr<warm else max(rank_arr[0], RMAX-(RMAX-rank_arr[0])*(pr-warm)/max(1e-9,1-warm))
                rr=np.full(clients,rank_arr[int(np.argmin(np.abs(rank_arr-r_)))])
            elif m=='ROBA_style':
                # Expected-latency rule (EL, after ROBA/TSFA): the largest rank
                # whose EXPECTED transmit time fits the deadline at the estimated
                # rate; no loss term and no joint admission, exactly as in the
                # real trainer.
                fits=8.0*size_of(rank_arr,payload_model)[None,:]/(rate[:,None]*deadline)<=1.0
                rr=rank_arr[np.max(np.where(fits, np.arange(len(rank_arr))[None,:], 0), axis=1)]
            else:
                # Adaptive: maximize expected useful delivered capacity
                # cap_gain(r)*P_deliver(r); downshifts under mobility/deadline
                # pressure instead of collapsing to the minimum rank.
                rr=rank_arr[np.argmax(cap_gain(rank_arr)[None,:]*p_mat,axis=1)]
            if m=='FedAvg':
                p=fed_avg_pon; utility=base*cap_gain(np.minimum(rr,RMAX))*p
                cost_i=np.full(clients,units_of(fed_avg_size))
            else:
                p=pon_of(mobility,rate,rr,deadline,payload_model); utility=base*cap_gain(rr)*p
                cost_map={int(x):int(c) for x,c in zip(rank_arr,costs)}
                cost_i=np.array([cost_map[int(x)] for x in rr])
            # Budget admission inside the sampled cohort, ceil grid units:
            # knapsack-greedy by delivered value per unit cost.
            order=cohort[np.argsort(-utility[cohort]/cost_i[cohort])]; sel=[]; used_u=0
            for i in order:
                if len(sel)>=cap or used_u+cost_i[i]>budget_u: continue
                sel.append(i); used_u+=int(cost_i[i])
            used=float(used_u*GRID)
            timely=float(np.mean(p[sel])) if sel else 0.
            # Total delivered value normalized by the cap: delivering more
            # useful updates within the budget is more learning signal.
            gain=float(np.sum(utility[sel])/cap) if sel else 0.
            noise=np.random.default_rng(seed*100000+t*100+METHODS.index(m)).normal(0,0.0005)
            # Saturating accuracy proxy: delivered useful value drives acc toward
            # its equilibrium instead of the old unbounded linear recursion.
            acc[m]=float(np.clip(acc[m]+0.08*gain*(1-acc[m])-0.0005*acc[m]+noise,0,1))
            delivered=rr[sel] if sel else np.array([])
            rows.append([seed,t,m,acc[m],timely,used,len(sel),float(np.mean(delivered)) if len(delivered) else '',gain])
    os.makedirs(out,exist_ok=True)
    with open(os.path.join(out,f'round_metrics_seed{seed}.csv'),'w',newline='') as f:
        w=csv.writer(f); w.writerow(['seed','round','method','accuracy','timely_ratio','bytes_mb','selected','mean_rank','useful_update_value']); w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='outputs/fedlora_sim_v4'); p.add_argument('--seeds',default='0,1,2,3,4')
    p.add_argument('--rounds',type=int,default=150); p.add_argument('--clients',type=int,default=500)
    p.add_argument('--budget',type=float,default=30.0,help='byte budget in real MB (ceil grid 0.01 MB)')
    p.add_argument('--cap',type=int,default=25); p.add_argument('--deadline',type=float,default=2.0)
    p.add_argument('--mobility-shape',type=float,default=2.0)
    p.add_argument('--payload-model',choices=['real','stylized'],default='real')
    a=p.parse_args();
    for s in map(int,a.seeds.split(',')): run(s,a.rounds,a.clients,[2,4,8,16],a.budget,a.cap,a.deadline,a.out,a.mobility_shape,a.payload_model)
if __name__=='__main__': main()
