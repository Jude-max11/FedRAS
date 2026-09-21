"""WCNC scheduler robustness checks (reviewer R1-(2)/R2-(8), round 2):

(a) P_on estimation error: schedule with a NOISY rate estimate (multiplicative
    lognormal error on the estimated rate, i.e. a biased/noisy predictor),
    evaluate the achieved objective under the TRUE link model. Quantifies the
    scheduling loss of an imperfect predictor vs the oracle DP.
(b) G(r) shape sensitivity: sqrt vs linear vs log saturation; reports the
    objective under each utility and how much the adaptive objective drops
    when evaluated under a different utility shape (rank-selection stability).

Reuses the identical instance model as dp_scheduler_study.py.
ISOLATION: writes ONLY to outputs/wcnc_extra/scheduler_robustness.csv.
"""
import csv, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dp_scheduler_study import instance, RANKS, RMAX, SIZE

SIGMA = 0.2          # predictor rate-estimation error (lognormal sigma)
CELLS = [(1.0, 300.0), (0.25, 300.0), (0.25, 12.0)]   # slack / tight-deadline / binding-budget

def values(rate, mob, base, deadline):
    P = np.exp(-mob[:, None] - 8.0 * SIZE[None, :] / (rate[:, None] * deadline))
    G = base[:, None] * CAP[None, :]
    return G * P

def cap(shape):
    if shape == 'sqrt':
        return np.sqrt(np.minimum(RANKS, RMAX) / RMAX)
    if shape == 'linear':
        return np.minimum(RANKS, RMAX) / RMAX
    if shape == 'log':
        return np.log1p(np.minimum(RANKS, RMAX)) / np.log1p(RMAX)
    raise ValueError(shape)

CAP = cap('sqrt')

def greedy_admit(U, cohort, K, budget):
    """Fair-cohort protocol: per-client argmax rank, greedy admit by value/byte."""
    used, tot = 0.0, 0.0
    ridx = np.argmax(U, axis=1)
    key = U[np.arange(len(U)), ridx] / SIZE[None, :][0]
    for i in cohort[np.argsort(-key[cohort])]:
        if used + SIZE[ridx[i]] > budget or used >= K and False:
            continue
        tot += U[i, ridx[i]]; used += SIZE[ridx[i]]
    return tot

def dp_decision_assign(rate_est, mob, base, deadline, K, budget, step=0.05, gshape='sqrt'):
    """DP under the ESTIMATED model; returns the per-client chosen rank (0=skip)."""
    global CAP
    saved = CAP; CAP = cap(gshape)
    U = values(rate_est, mob, base, deadline)
    CAP = saved
    n = U.shape[0]
    B = int(round(budget / step))
    sizes = np.maximum(1, np.round(SIZE / step).astype(int))
    NEG = -1e18
    V = np.full((K + 1, B + 1), NEG); V[0, 0] = 0.0
    back = np.zeros((n, K + 1, B + 1), dtype=np.int8)
    for i in range(n):
        W = np.full_like(V, NEG)
        for j, r in enumerate(RANKS):
            s = sizes[j]
            if s > B:
                continue
            cand = V[:-1, :B - s + 1] + U[i, j]
            tgt = W[1:, s:]
            m = cand > tgt
            tgt[m] = cand[m]
            back[i, 1:, s:][m] = j + 1
        take = W > V
        Vn = np.where(take, W, V)
        back[i][~take] = 0
        V = Vn
    # backtrack
    b = int(np.argmax(V.max(axis=0))) if False else int(np.unravel_index(np.argmax(V), V.shape)[1])
    k = int(np.unravel_index(np.argmax(V), V.shape)[0])
    assign = np.zeros(n, dtype=int)
    for i in range(n - 1, -1, -1):
        j = back[i, k, b]
        if j > 0:
            assign[i] = RANKS[j - 1]
            b -= int(sizes[j - 1]); k -= 1
    return assign

def achieved(assign, rate, mob, base, deadline, gshape='sqrt'):
    global CAP
    saved = CAP; CAP = cap(gshape)
    U = values(rate, mob, base, deadline)
    CAP = saved
    ridx = [int(np.where(RANKS == a)[0][0]) for a in assign if a > 0]
    return float(sum(U[i, j] for i, j in zip(*np.where(assign[:, None] == RANKS[None, :]))))

def main():
    rng = np.random.default_rng(2026)
    out_rows = []
    for deadline, budget in CELLS:
        loss_noisy, loss_g, agree = [], {g: [] for g in ('linear', 'log')}, {g: [] for g in ('linear', 'log')}
        for _ in range(30):
            rate, mob, base = instance(500, 2.0, rng)
            U_true = values(rate, mob, base, deadline)
            # (a) noisy predictor: estimated rate is true rate * lognormal(sigma)
            rate_est = rate * rng.lognormal(0.0, SIGMA, len(rate))
            assign_noisy = dp_decision_assign(rate_est, mob, base, deadline, 25, budget)
            v_noisy = achieved(assign_noisy, rate, mob, base, deadline)
            assign_oracle = dp_decision_assign(rate, mob, base, deadline, 25, budget)
            v_oracle = achieved(assign_oracle, rate, mob, base, deadline)
            loss_noisy.append((v_oracle - v_noisy) / max(v_oracle, 1e-9))
            # (b) utility-shape sensitivity: decisions under linear/log utility,
            #     evaluated under the true (sqrt) utility
            for g in ('linear', 'log'):
                assign_g = dp_decision_assign(rate, mob, base, deadline, 25, budget, gshape=g)
                v_g = achieved(assign_g, rate, mob, base, deadline, gshape='sqrt')
                loss_g[g].append((v_oracle - v_g) / max(v_oracle, 1e-9))
                agree[g].append(float(np.mean((assign_g > 0) == (assign_oracle > 0))))
        row = {'deadline': deadline, 'budget': budget,
               'obj_loss_pred_sigma0.2_mean': round(float(np.mean(loss_noisy)), 4),
               'obj_loss_pred_sigma0.2_max': round(float(np.max(loss_noisy)), 4),
               'obj_loss_Glinear_mean': round(float(np.mean(loss_g['linear'])), 4),
               'obj_loss_Glog_mean': round(float(np.mean(loss_g['log'])), 4),
               'admit_agreement_Glinear': round(float(np.mean(agree['linear'])), 3),
               'admit_agreement_Glog': round(float(np.mean(agree['log'])), 3)}
        out_rows.append(row)
        print(row, flush=True)
    os.makedirs('outputs/wcnc_extra', exist_ok=True)
    out = 'outputs/wcnc_extra/scheduler_robustness.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print('written', out)


if __name__ == '__main__':
    main()
