"""Deadline-aware FedRAS trainer (v4, real-byte era).

Changes vs v3 (kept in backups/scripts_backup_20260917_pre_realbytes.zip):
1. REAL payload accounting. Frozen-A clients upload only the trainable B factor
   (4 bytes x sum of B output dims = 5,632 B per rank column, i.e. 0.090 MB at
   rank 16); FedAvg_full transmits its full state (backbone trainables + head
   adapters, ~5.3 MB). The stylized s(r)=0.02r+0.5 MB model is removed from the
   trainer: delivery probabilities and byte metrics use real serialized bytes.
2. Rank-r local training. Each LoRA module slices to the assigned rank for
   local forward/backward; gradients reach only the first r coordinates, so
   the rank genuinely bounds local adaptation capacity.
3. Truly frozen A. Server state of frozen-A methods holds only .B tensors; A
   stays at its shared random initialization and never enters aggregation or
   server momentum (Proposition 2 of the paper). JointAB keeps joint A/B
   averaging as the ablation competitor.
4. Per-method evaluation. A dedicated eval model is loaded with each method's
   own post-round state (the aggregated state if any update arrived, else the
   method's previous state). Zero-delivery rounds no longer inherit another
   method's model state.
5. Uniform CONSERVATIVE budget discretization: ceil onto a 0.01 MB grid for
   the DP arms and the byte budget, matching the solver study and simulator.
6. New DP_same arm: exact DP restricted to the sampled cohort; comparing
   DP_cohort vs DP_same isolates candidate-pool widening from in-cohort
   scheduling (DP_same vs fair-cohort Adaptive_rank).
7. Optional SAGIN link profile (--link-profile sagin): ground/aerial/satellite
   client classes with distinct rate and mobility laws; per-client outcomes
   are logged to link_outcomes.csv when enabled.
8. FedAvg_offline control: FedAvg_full without delivery modeling (accuracy
   upper bound separating learning capacity from delivery feasibility).
9. metrics.csv: bytes_mb (real serialized MB); mean_rank is the delivered-
   update mean and is left empty on zero-delivery rounds (`aggregated` is the
   per-round delivered count and the correct weight for pooling).

Reviewer-round-2 additions (v4.1):
10. Mean-stationary AR(1) in the mismatch arm: the log-rate process now reverts
    to the headline law's mean (2.5) instead of decaying to zero.
11. Distribution-shift task (--shift-severity SEV): deterministic covariate
    shift (noise + brightness/contrast) applied to client training data and the
    test set while the backbone pretrains on clean data; --pretrain-frac F
    further disjointifies the pretrain corpus from the client corpus.
12. Tiny-ViT backbone (--arch vit): LoRA on qkv/proj/fc1/fc2 of every block;
    payload constants are audited from the instantiated model as for the CNN.
13. New baselines: SA_mono (faithful-style monotone-shrink annealing),
    Rate_thresh (rate threshold at fixed ranks), Greedy_value / Greedy_vpb
    (budget-aware value / value-per-byte greedy over the full client pool).
14. Aggregation edge cases: --empty-cols {zero,hold} fixes the W_j=0 rule;
    empty rounds hold state and momentum (documented).
15. Budget cap uses floor(B_max/q) while item costs stay ceil (conservative
    grid; identical when budgets are grid multiples, as in all reported runs).
16. --corr-skew: label skew correlated with SAGIN link classes; per-class-half
    accuracy is logged to bias_eval.csv. --jointab-ab bills JointAB at A+B.
    Local training wall time is accumulated into config.json (train_ms_total).
"""

import argparse, csv, hashlib, json, math, os, random, shutil, time
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

RANKS = [2, 4, 8, 16]
RMAX = 16
BUDGET_GRID = 0.01  # MB per budget unit; ceil discretization everywhere

class LoRALinear(nn.Module):
    def __init__(self, base, rmax=RMAX):
        super().__init__()
        self.base, self.rmax, self.eff_r = base, rmax, rmax
        for p in base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(rmax, base.in_features) * .01)
        self.B = nn.Parameter(torch.zeros(base.out_features, rmax))
    def forward(self, x):
        r = self.eff_r
        return self.base(x) + (x @ self.A[:r].t()) @ self.B[:, :r].t()
    def extra_repr(self):
        return f'rmax={self.rmax} eff_r={self.eff_r}'

class LoRAConv2d(nn.Module):
    """Low-rank conv delta via 1x1 factorization: base(x) + B(A(x))."""
    def __init__(self, base, rmax=RMAX):
        super().__init__()
        self.base, self.rmax, self.eff_r = base, rmax, rmax
        for p in base.parameters():
            p.requires_grad = False
        k = base.kernel_size
        assert k == (3, 3), 'LoRA delta uses 1x1 factors; base stays 3x3'
        self.A = nn.Parameter(torch.randn(rmax, base.in_channels, 1, 1) * .01)
        self.B = nn.Parameter(torch.zeros(base.out_channels, rmax, 1, 1))
    def forward(self, x):
        r = self.eff_r
        return self.base(x) + nn.functional.conv2d(
            nn.functional.conv2d(x, self.A[:r]), self.B[:, :r],
            stride=self.base.stride, padding=0)
    def extra_repr(self):
        return f'rmax={self.rmax} eff_r={self.eff_r}'

def set_eff_r(m, r):
    for mod in m.modules():
        if hasattr(mod, 'eff_r'):
            mod.eff_r = min(r, mod.rmax)

def conv_block(cin, cout, rmax):
    return nn.Sequential(
        LoRAConv2d(nn.Conv2d(cin, cout, 3, padding=1, bias=False), rmax), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        LoRAConv2d(nn.Conv2d(cout, cout, 3, padding=1, bias=False), rmax), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))

class Net(nn.Module):
    """Frozen CNN feature extractor + federated LoRA adapters on every block."""
    def __init__(self, c, in_ch, width=256, rmax=RMAX, base_ch=64):
        super().__init__()
        self.feat = nn.Sequential(conv_block(in_ch, base_ch, rmax), conv_block(base_ch, 2 * base_ch, rmax),
                                  conv_block(2 * base_ch, 4 * base_ch, rmax),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten())
        fdim = 4 * base_ch
        self.fc1 = LoRALinear(nn.Linear(fdim, width), rmax)
        self.fc2 = LoRALinear(nn.Linear(width, width), rmax)
        self.out = nn.Linear(width, c)
    def lora_params(self):
        return [p for n, p in self.named_parameters() if n.endswith('.A') or n.endswith('.B')]
    def forward(self, x):
        h = self.feat(x)
        return self.out(torch.relu(self.fc2(torch.relu(self.fc1(h)))))

class ViTBlock(nn.Module):
    def __init__(self, dim, heads, rmax=RMAX):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.qkv = LoRALinear(nn.Linear(dim, 3 * dim), rmax)
        self.proj = LoRALinear(nn.Linear(dim, dim), rmax)
        self.fc1 = LoRALinear(nn.Linear(dim, 4 * dim), rmax)
        self.fc2 = LoRALinear(nn.Linear(4 * dim, dim), rmax)
        self.h = heads
    def forward(self, x):
        b, n, dd = x.shape; hd = dd // self.h
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.h, hd).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        o = (torch.softmax(att, -1) @ v).transpose(1, 2).reshape(b, n, dd)
        x = x + self.proj(o)
        return x + self.fc2(nn.functional.gelu(self.fc1(self.ln2(x))))

class ViTNet(nn.Module):
    """Frozen tiny-ViT backbone + federated LoRA on qkv/proj/fc1/fc2 per block."""
    def __init__(self, c, in_ch, img=32, patch=4, dim=192, depth=6, heads=3, rmax=RMAX):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, dim, patch, patch)
        self.pos = nn.Parameter(torch.randn(1, (img // patch) ** 2 + 1, dim) * .02)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList([ViTBlock(dim, heads, rmax=rmax) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, c)
    def lora_params(self):
        return [p for n, p in self.named_parameters() if n.endswith('.A') or n.endswith('.B')]
    def forward(self, x):
        h = self.patch(x).flatten(2).transpose(1, 2)
        h = torch.cat([self.cls.expand(h.size(0), -1, -1), h], 1) + self.pos
        for blk in self.blocks:
            h = blk(h)
        return self.out(self.ln(h)[:, 0])

class ShiftedDS(torch.utils.data.Dataset):
    """Deterministic covariate shift of an underlying dataset, fixed per index:
    additive Gaussian noise plus a brightness/contrast offset. The corruption
    is a property of the deployment data, so it is identical across epochs."""
    def __init__(self, ds, sev):
        self.ds, self.sev = ds, float(sev)
    def __len__(self):
        return len(self.ds)
    @property
    def targets(self):
        return self.ds.targets
    def __getitem__(self, i):
        x, y = self.ds[i]
        g = np.random.default_rng(i)
        x = x + self.sev * torch.from_numpy(g.standard_normal(tuple(x.shape))).float()
        x = x * (1 + .25 * self.sev) + .1 * self.sev
        return x, y

STATS = {'mnist': ((.1307,), (.3081,)),
         'cifar10': ((.4914, .4822, .4465), (.2470, .2435, .2616)),
         'cifar100': ((.5071, .4865, .4409), (.2673, .2564, .2762))}

def make_transforms(name, augment):
    mean, std = STATS[name]
    tf = [transforms.ToTensor()]
    if augment and name != 'mnist':
        tf = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
              transforms.Normalize(mean, std), transforms.RandomErasing(p=.25, value=0)]
        return transforms.Compose(tf)
    return transforms.Compose(tf + [transforms.Normalize(mean, std)])

def pretrain_backbone(model, ds, device, epochs, lr, batch):
    # Train base weights only: adapters stay at their zero-delta init so the
    # backbone is a pure base model and all adapter learning is federated.
    model.train()
    for n, p in model.named_parameters():
        p.requires_grad = not (n.endswith('.A') or n.endswith('.B'))
    base_params = [p for n, p in model.named_parameters() if not (n.endswith('.A') or n.endswith('.B'))]
    opt = torch.optim.SGD(base_params, lr=lr, momentum=.9, nesterov=True, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    amp = device.type == 'cuda'
    for _ in range(epochs):
        for x, y in DataLoader(ds, batch, shuffle=True, num_workers=0):
            opt.zero_grad()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
        sch.step()

def ls(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if k.endswith('.A') or k.endswith('.B')}
def ls_b(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if k.endswith('.B')}
def load(m, s):
    z = m.state_dict()
    for k, v in s.items():
        if k in z: z[k].copy_(v)
    m.load_state_dict(z)
def load_base(m, s):
    z = m.state_dict()
    for k, v in s.items():
        if k in z: z[k].copy_(v)
    m.load_state_dict(z)

def avg_b(ss, ww, rr, use_mask=True, prev=None):
    # Aggregate B-only states in the shared rmax coordinate system; each
    # coordinate is normalized by the clients that actually activate it
    # (rank-masked averaging over the frozen A, Proposition 3 of the paper).
    # Columns with no delivered coverage (W_j = 0) follow the configured rule:
    # 'zero' pulls them to zero via the momentum target, 'hold' keeps the
    # previous server state (prev) at those columns.
    o = {}
    for k in ss[0]:
        z = torch.zeros_like(ss[0][k]); den = torch.zeros_like(ss[0][k])
        for s, w, r in zip(ss, ww, rr):
            v = s[k].clone(); v[:, r:] = 0
            mask = torch.ones_like(v); mask[:, r:] = 0
            z += w * v; den += w * mask
        if not use_mask:  # ablation: plain weighted averaging, no rank masking
            den = torch.full_like(den, sum(ww))
        fallback = (prev or {}).get(k)
        o[k] = torch.where(den > 0, z / den, fallback if fallback is not None else torch.zeros_like(z))
    return o

def avg_ab(ss, ww, use_mask=True):
    # Joint A/B averaging (the JointAB ablation competitor). JointAB rounds are
    # homogeneous (every client trains at rmax), so masking is a no-op here.
    o = {}
    for k in ss[0]:
        z = torch.zeros_like(ss[0][k]); den = 0.0
        for s, w in zip(ss, ww):
            z += w * s[k]; den += w
        o[k] = z / max(den, 1e-12)
    return o

def avg_full(states, weights):
    # Plain FedAvg over full trainable head state dicts (FedAvg_full baseline).
    o = {}
    for k in states[0]:
        z = torch.zeros_like(states[0][k]); den = 0.0
        for s, w in zip(states, weights):
            z += w * s[k]; den += w
        o[k] = z / max(den, 1e-12)
    return o

def full_state(m, keys):
    return {k: m.state_dict()[k].detach().cpu().clone() for k in keys}

def parts(y, n, seed, a, min_per=8):
    # Per-class Dirichlet split with a guaranteed minimum shard size.
    g = np.random.default_rng(seed); out = [[] for _ in range(n)]
    for c in np.unique(y):
        ids = np.where(np.asarray(y) == c)[0]; g.shuffle(ids)
        cuts = (np.cumsum(g.dirichlet(np.full(n, a))) * len(ids)).astype(int)[:-1]
        for j, x in enumerate(np.split(ids, cuts)): out[j] += x.tolist()
    out = [np.asarray(sorted(x)) for x in out]
    for j in range(n):  # move excess from the largest shard into small ones
        while len(out[j]) < min_per:
            src = int(np.argmax([len(x) for x in out]))
            if len(out[src]) <= min_per + 2: break
            out[j] = np.sort(np.append(out[j], out[src][-1:])); out[src] = out[src][:-1]
    return out

def parts_corr(y, n, seed, alpha, ncls, min_per=8):
    # Label skew correlated with SAGIN link classes: classes below ncls/2 live
    # only on weak-link (aerial/satellite) clients, the rest on ground clients.
    g = np.random.default_rng(seed); out = [[] for _ in range(n)]
    ground = [i for i in range(n) if group_of(i, n) == 0]
    weak = [i for i in range(n) if group_of(i, n) >= 1]
    half = ncls // 2
    for cls in np.unique(np.asarray(y)):
        ids = np.where(np.asarray(y) == cls)[0]; g.shuffle(ids)
        elig = weak if cls < half else ground
        wts = g.dirichlet(np.full(len(elig), alpha))
        cuts = (np.cumsum(wts) * len(ids)).astype(int)[:-1]
        for j, x in enumerate(np.split(ids, cuts)):
            out[elig[j]] += x.tolist()
    out = [np.asarray(sorted(x)) if len(x) else np.array([], dtype=int) for x in out]
    for j in range(n):
        while len(out[j]) < min_per:
            src = int(np.argmax([len(x) for x in out]))
            if len(out[src]) <= min_per + 2: break
            out[j] = np.sort(np.append(out[j], out[src][-1:])); out[src] = out[src][:-1]
    return out

def ev(m, ds, d, batch=512):
    m.eval(); n = o = 0
    amp = d.type == 'cuda'
    with torch.no_grad():
        for x, y in DataLoader(ds, batch):
            with torch.autocast(d.type, dtype=torch.bfloat16, enabled=amp):
                out = m(x.to(d))
            o += (out.argmax(1).cpu() == y).sum().item(); n += len(y)
    return o / max(1, n)

SAGIN_GROUPS = ('ground', 'aerial', 'satellite')

def group_of(i, n):
    # 60% ground / 25% aerial / 15% satellite (12/5/3 at n=20, 300/125/75 at n=500)
    ng = int(round(.6 * n)); na = int(round(.25 * n))
    return 0 if i < ng else (1 if i < ng + na else 2)

def draw_links(profile, g, n, mobility_shape):
    # Draw order is fixed so every method sees identical traces (CRN).
    if profile == 'uniform':
        return (np.clip(g.lognormal(2.5, .7, n), 1, 40), g.beta(*mobility_shape, size=n))
    rate = np.empty(n); mob = np.empty(n)
    idx = [np.array([i for i in range(n) if group_of(i, n) == k]) for k in range(3)]
    # Mobility intensity: ground lowest (stable links), aerial mid (handover),
    # satellite highest (outages); rate laws do the rest of the separation.
    laws = [((2.8, .5, 5, 40), (2, 8)),    # ground: stable and fast
            ((2.3, .7, 2, 40), (2, 5)),    # aerial: handover-prone
            ((1.6, .5, .5, 15), (2, 3))]   # satellite: scarce, outage-prone
    for k, ids in enumerate(idx):
        (mu, sg, lo, hi), (a, b) = laws[k]
        rate[ids] = np.clip(g.lognormal(mu, sg, len(ids)), lo, hi)
        mob[ids] = g.beta(a, b, len(ids))
    return rate, mob

def size_mb(r, bytes_per_rank):
    # Real serialized upload of a rank-r frozen-A update: B only, FP32.
    return r * bytes_per_rank / 1e6

def budget_units(mb, step=BUDGET_GRID):
    # Conservative ceil: the budget always covers the true byte bill.
    return max(1, int(math.ceil(mb / step - 1e-9)))

def budget_cap_units(mb, step=BUDGET_GRID):
    # Floor on the budget cap: a grid-feasible plan is then feasible in true
    # bytes (ceil'd costs >= true costs, floor'd cap <= true budget).
    return max(0, int(math.floor(mb / step + 1e-9)))

def rank_for(method, i, mob, rate, deadline, g, ranks, rmax, rnd=0, rounds=1, no_capacity=False,
             bytes_per_rank=5632, full_mb=5.3):
    def pon(r):
        return math.exp(-mob[i]) * math.exp(-8.0 * size_mb(r, bytes_per_rank) / (rate[i] * deadline))
    cap = lambda r: math.sqrt(min(r, rmax) / rmax)
    if method.startswith('FedLoRA_r'): return int(method.split('_r')[1])
    if method == 'Random_rank': return int(g.choice(ranks))
    if method.startswith('FedAvg'): return 64  # sentinel: no LoRA rank (full update)
    if method == 'JointAB': return rmax  # joint A/B training at the full provisioned rank
    if method == 'Oort_utility': return ranks[len(ranks) // 2]
    if method == 'SA_rank':
        # Simulated-annealing schedule (SA-FedLoRA style): heat then cool,
        # shared by all clients, independent of link state.
        pr = rnd / max(1, rounds - 1)
        target = 2 + (rmax - 2) * (2 * pr) if pr < .5 else rmax - (rmax - 2) * (2 * pr - 1)
        return ranks[int(np.argmin(np.abs(np.asarray(ranks) - target)))]
    if method == 'Adaptive_rank':  # maximize expected useful delivered capacity
        if no_capacity:  # ablation: delivery probability only
            return ranks[int(np.argmax([pon(r) for r in ranks]))]
        return ranks[int(np.argmax([cap(r) * pon(r) for r in ranks]))]
    if method == 'SA_mono':
        # Faithful-style monotone schedule (after SA-FedLoRA): hold the full
        # width through a short initialization phase, then shrink monotonically
        # to the minimum rank; link-oblivious like SA_rank.
        pr = rnd / max(1, rounds - 1)
        warm = .2
        target = rmax if pr < warm else \
            max(min(ranks), rmax - (rmax - min(ranks)) * (pr - warm) / max(1e-9, 1 - warm))
        return ranks[int(np.argmin(np.abs(np.asarray(ranks) - target)))]
    if method == 'Rate_thresh':
        # Simple network-aware rule: full rank on links whose estimated rate
        # clears a fixed threshold, minimum rank otherwise.
        return rmax if rate[i] >= RATE_THRESH_MBPS else min(ranks)
    if method == 'ROBA_style':
        # Wireless-aware, delivery-assured rule (after ROBA/TSFA): the largest
        # rank whose EXPECTED transmission time fits the deadline at the
        # estimated rate; no loss term, no joint admission -- the expectation
        # vs per-payload-probability contrast of the comparison study.
        best = min(ranks)
        for r in ranks:
            if 8.0 * size_mb(r, bytes_per_rank) / max(rate[i], 1e-6) <= deadline:
                best = r
        return best
    if method == 'FedARA_svd':
        # Training-side adaptive rank (after FedARA): next round's global rank
        # is the SVD energy-capture rank of the aggregated update delta.
        return FEDARA['rank']
    raise ValueError(method)

RATE_THRESH_MBPS = 8.0
FEDARA = {'rank': 16, 'energy': 0.9}


def svd_energy_rank(cand, old, ranks):
    """Smallest candidate rank capturing FEDARA['energy'] of the aggregated
    delta's squared singular values, per adapted layer; the median across
    layers becomes the next round's global rank."""
    ks = []
    for k, v in cand.items():
        d = (v - old[k]).float().reshape(v.shape[0], -1)
        if d.abs().max().item() < 1e-12:
            continue
        s = torch.linalg.svdvals(d)
        e = s ** 2
        c = torch.cumsum(e, 0) / e.sum().clamp_min(1e-20)
        ks.append(int(torch.searchsorted(c, torch.tensor(FEDARA['energy'])).item()) + 1)
    if not ks:
        return FEDARA['rank']
    k = int(np.median(ks))
    return int(ranks[int(np.argmin(np.abs(np.asarray(ranks) - k)))])

def dp_assign(mob, rate, deadline, ranks, rmax, client_ids, cap, budget, bytes_per_rank,
              step=BUDGET_GRID, cap_exp=0.5):
    """Exact client-rank assignment over `client_ids`: maximize
    sum_i G(r_i)*P_on(i,r_i) s.t. sum ceil-grid cost <= budget and at most `cap`
    admissions (multiple-choice knapsack with cardinality constraint).
    Conservative ceil discretization; returns {client_id: r}."""
    Bu = budget_cap_units(budget, step)
    NEG = -1e9
    V = np.full((cap + 1, Bu + 1), NEG); V[0, 0] = 0.0
    back = []
    for i in client_ids:
        best = V.copy(); act = np.full((cap + 1, Bu + 1), -1, dtype=np.int64)
        fail = math.exp(-mob[i])
        for r in ranks:
            c = budget_units(size_mb(r, bytes_per_rank), step)
            if c > Bu: continue
            val = fail * math.exp(-8.0 * size_mb(r, bytes_per_rank) / (rate[i] * deadline)) * \
                (min(r, rmax) / rmax) ** cap_exp
            shifted = np.full_like(V, NEG)
            shifted[1:, c:] = V[:-1, :Bu + 1 - c] + val
            upd = shifted > best
            best = np.where(upd, shifted, best)
            act[upd] = r
        back.append(act); V = best
    k, b = np.unravel_index(np.argmax(V), V.shape)
    assign = {}
    for j in reversed(range(len(client_ids))):
        r = int(back[j][k, b])
        if r > 0:
            assign[client_ids[j]] = r
            b -= budget_units(size_mb(r, bytes_per_rank), step); k -= 1
    return assign

def greedy_assign(mob, rate, deadline, ranks, rmax, client_ids, cap, budget, bytes_per_rank,
                  per_byte=False, step=BUDGET_GRID, cap_exp=0.5):
    """Budget-aware greedy over the full pool: sort (client, rank) pairs by
    utility (or utility per grid cost) and admit while cap and floor'd budget
    hold. The strong heuristic comparators of the binding-budget study."""
    Bu = budget_cap_units(budget, step)
    pairs = []
    for i in client_ids:
        fail = math.exp(-mob[i])
        for r in ranks:
            c = budget_units(size_mb(r, bytes_per_rank), step)
            if c > Bu:
                continue
            u = fail * math.exp(-8.0 * size_mb(r, bytes_per_rank) / (rate[i] * deadline)) * \
                (min(r, rmax) / rmax) ** cap_exp
            pairs.append((u / c if per_byte else u, i, r, c))
    pairs.sort(key=lambda t: -t[0])
    used = 0; assign = {}
    for _, i, r, c in pairs:
        if i in assign or len(assign) >= cap or used + c > Bu:
            continue
        assign[i] = r; used += c
    return assign

def frob(sd):
    return float(sum(torch.linalg.vector_norm(v).item() ** 2 for v in sd.values()) ** .5)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--dataset', choices=['cifar10', 'cifar100', 'mnist'], default='cifar10')
    p.add_argument('--out', default='outputs/real_fedlora')
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--clients', type=int, default=20)
    p.add_argument('--per-round', type=int, default=10)
    p.add_argument('--local-epochs', type=int, default=1)
    p.add_argument('--pretrain-epochs', type=int, default=6)
    p.add_argument('--pretrain-lr', type=float, default=.05)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=.005)
    p.add_argument('--lr-final-frac', type=float, default=.05)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--server-momentum', type=float, default=.3)
    p.add_argument('--deadline', type=float, default=2.0,
                   help='round deadline in seconds; nominal is 2.0 under real-byte payloads')
    p.add_argument('--mobility-shape', default='2.0',
                   help='Beta shape of per-round client mobility: "a" (b=5) or "a,b"; lower mean = more mobile')
    p.add_argument('--link-profile', choices=['uniform', 'sagin'], default='uniform')
    p.add_argument('--arch', choices=['cnn', 'vit'], default='cnn',
                   help='backbone: frozen CNN (default) or tiny ViT with LoRA on attention/MLP projections')
    p.add_argument('--shift-severity', type=float, default=0.0,
                   help='distribution-shift task: deterministic noise+brightness corruption of client train data and the test set (backbone pretrains clean)')
    p.add_argument('--pretrain-frac', type=float, default=1.0,
                   help='fraction of the training set used for centralized pretraining; clients hold the disjoint complement')
    p.add_argument('--corr-skew', action='store_true',
                   help='label skew correlated with SAGIN link classes: weak-link classes live on aerial/satellite clients; logs bias_eval.csv')
    p.add_argument('--empty-cols', choices=['zero', 'hold'], default='zero',
                   help='W_j=0 aggregation rule: pull uncovered columns to zero or hold the previous server state')
    p.add_argument('--jointab-ab', action='store_true',
                   help='bill the JointAB baseline at its true A+B payload instead of B-only')
    p.add_argument('--pretrain-only', action='store_true',
                   help='pretrain the backbone, save it to the cache, and exit')
    p.add_argument('--backbone-cache-dir', default='',
                   help='shared backbone cache directory (keyed by config fingerprint); avoids re-pretraining per run')
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--base-channels', type=int, default=64, help='width of the first conv block; blocks are ch/2ch/4ch')
    p.add_argument('--lora-init-std', type=float, default=.1, help='std of the A init; larger values break the B=0 cold start faster')
    p.add_argument('--full-lr-scale', type=float, default=.4, help='lr scale for the FedAvg_full head baseline')
    p.add_argument('--prox', type=float, default=0., help='FedProx coefficient anchoring local params to the server state')
    p.add_argument('--no-freeze-a', action='store_true', help='train both A and B (JointAB competitor)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--no-reset-lora', action='store_true', help='start federated LoRA from pretrained A/B instead of fresh init')
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--no-full-baseline', action='store_true', help='skip the FedAvg_full head fine-tuning baseline')
    p.add_argument('--rmax', type=int, default=16, help='provisioned adapter factor width')
    p.add_argument('--methods', default='', help='comma-separated subset of methods to run (default: standard five)')
    p.add_argument('--no-rank-mask', action='store_true', help='ablation: plain weighted averaging without rank masking')
    p.add_argument('--fp16-upload', action='store_true',
                   help='compression baseline: trainable B is uploaded at 2 bytes/param with a real fp16 round-trip (quantize on delivery); delivery and byte accounting use the halved payload')
    p.add_argument('--no-capacity-term', action='store_true', help='ablation: adaptive rule uses delivery probability only (no capacity gain)')
    p.add_argument('--rate-est-bias', type=float, default=0.0,
                   help='scheduler sees rate*(1+bias); delivery keeps the true rate')
    p.add_argument('--mu-est-scale', type=float, default=1.0,
                   help='scheduler sees mobility*scale; delivery keeps the true value')
    p.add_argument('--delivery-mismatch', action='store_true', help='de-circularized arm: AR(1) realized rates, stale EMA estimates for scheduling, and structural delivery (propagation/handover delay + independent loss) instead of the hazard law')
    p.add_argument('--dp-budget', type=float, default=1.0,
                   help='byte budget in REAL MB for the DP_cohort/DP_same arms; 1.0 MB = 10x the ceil-grid cost of rank 16')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--a-init-from', default='',
                   help='checkpoint.pt of an earlier run whose frozen LoRA A factors seed this run, '
                        'so arms trained in a separate process share one adapter initialisation')
    p.add_argument('--device', default='auto')
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    d = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available() else a.device if a.device != 'auto' else 'cpu')
    os.makedirs(a.out, exist_ok=True)
    mn = a.dataset == 'mnist'; c = 100 if a.dataset == 'cifar100' else 10; in_ch = 1 if mn else 3
    augment = not a.no_augment
    DS = datasets.MNIST if mn else (datasets.CIFAR100 if a.dataset == 'cifar100' else datasets.CIFAR10)
    tr = DS(a.data, train=True, download=False, transform=make_transforms(a.dataset, augment))
    te = DS(a.data, train=False, download=False, transform=make_transforms(a.dataset, False))
    # Distribution-shift task: clients and the deployment test set see a
    # deterministically corrupted view; the backbone pretrains on clean data.
    # --pretrain-frac disjointifies the pretrain corpus from the client corpus.
    te_eval = ShiftedDS(te, a.shift_severity) if a.shift_severity > 0 else te
    tr_client = ShiftedDS(tr, a.shift_severity) if a.shift_severity > 0 else tr
    tr_targets = np.asarray(tr.targets)
    if a.pretrain_frac < 1.0:
        rg = np.random.default_rng(1000 + a.seed)
        perm = rg.permutation(len(tr_targets))
        na = int(a.pretrain_frac * len(perm))
        pre_ds = Subset(tr, perm[:na].tolist()); cli_idx = perm[na:]
    else:
        pre_ds = tr; cli_idx = np.arange(len(tr_targets))
    if a.corr_skew:
        assert a.link_profile == 'sagin', '--corr-skew requires --link-profile sagin'
        pp = [cli_idx[x] for x in parts_corr(tr_targets[cli_idx], a.clients, a.seed, a.alpha, c)]
    else:
        pp = parts(tr_targets[cli_idx], a.clients, a.seed, a.alpha)
        if a.pretrain_frac < 1.0:
            pp = [cli_idx[x] for x in pp]

    ranks = [r for r in (2, 4, 8, 16, 32) if r <= a.rmax]

    def new_model():
        return (ViTNet(c, in_ch, rmax=a.rmax) if a.arch == 'vit' else
                Net(c, in_ch, a.width, rmax=a.rmax, base_ch=a.base_channels)).to(d)
    m = new_model()
    # Backbone cache keyed by config fingerprint (never silently reuse stale weights).
    fp = {'arch': a.arch + '_lora', 'dataset': a.dataset, 'width': a.width, 'base_channels': a.base_channels,
          'vit': {'patch': 4, 'dim': 192, 'depth': 6, 'heads': 3} if a.arch == 'vit' else None,
          'pretrain_epochs': a.pretrain_epochs, 'pretrain_lr': a.pretrain_lr, 'batch_size': a.batch_size,
          'augment': augment, 'seed': a.seed, 'pretrain_frac': a.pretrain_frac}
    cache_path = ''
    if a.backbone_cache_dir:
        os.makedirs(a.backbone_cache_dir, exist_ok=True)
        cache_path = os.path.join(a.backbone_cache_dir,
                                  hashlib.md5(json.dumps(fp, sort_keys=True).encode()).hexdigest() + '.pt')
    pre = os.path.join(a.out, 'backbone_pretrained.pt'); meta = os.path.join(a.out, 'backbone_meta.json')
    if os.path.exists(cache_path) or (os.path.exists(pre) and os.path.exists(meta) and json.load(open(meta)) == fp):
        # base weights only: adapter shapes depend on --rmax and are fresh anyway
        sd = torch.load(cache_path if cache_path else pre, map_location=d)
        m.load_state_dict({k: v for k, v in sd.items() if not (k.endswith('.A') or k.endswith('.B'))}, strict=False)
    else:
        print('pretraining backbone...', flush=True)
        pretrain_backbone(m, pre_ds, d, a.pretrain_epochs, a.pretrain_lr, a.batch_size)
        torch.save(m.state_dict(), pre); json.dump(fp, open(meta, 'w'))
        if cache_path:
            shutil.copyfile(pre, cache_path)
    base_acc = ev(m, te_eval, d)  # zero-communication control on the deployment distribution
    print(f'pretrained backbone test acc = {base_acc:.4f}', flush=True)
    if a.pretrain_only:
        return

    base_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()
                  if not (k.endswith('.A') or k.endswith('.B'))}
    if a.no_reset_lora:
        initial_state = ls(m)
    else:  # fresh adapters: the LoRA delta must be re-learned federatively
        initial_state = {k: (torch.randn_like(v) * a.lora_init_std if k.endswith('.A') else torch.zeros_like(v))
                         for k, v in ls(m).items()}
    a_init = {k: v.clone() for k, v in initial_state.items() if k.endswith('.A')}
    if a.a_init_from:  # share the adapter basis of an earlier run (cross-process fairness)
        _z = torch.load(a.a_init_from, map_location='cpu')
        _ai = _z.get('a_init', {}) if isinstance(_z, dict) else {}
        if _ai:
            a_init = {k: v.to(d) for k, v in _ai.items()}
            print(f'loaded a_init from {a.a_init_from} ({len(a_init)} tensors)', flush=True)
    b_init = {k: v.clone() for k, v in initial_state.items() if k.endswith('.B')}

    # Real payload constants, measured from the actual model (never hardcoded).
    bytes_per_rank = 4 * sum(int(v.shape[0]) for k, v in m.state_dict().items() if k.endswith('.B'))
    bytes_a_per_rank = 4 * sum(int(v.shape[1]) for k, v in m.state_dict().items() if k.endswith('.A'))
    joint_bytes_per_rank = bytes_per_rank + bytes_a_per_rank  # audited A+B per rank column
    full_keys = [n for n, _ in m.named_parameters() if not (n.endswith('.A') or n.endswith('.B'))]
    n_adapt = sum(v.numel() for v in initial_state.values())
    full_mb = 4 * sum(p.numel() for _, p in m.named_parameters()) / 1e6  # base + adapters at rmax

    all_methods = ['FedLoRA_r4', f'FedLoRA_r{ranks[-1]}', 'Random_rank', 'Adaptive_rank',
                   'DP_cohort', 'DP_same', 'Greedy_value', 'Greedy_vpb', 'SA_rank', 'SA_mono',
                   'Rate_thresh', 'ROBA_style', 'FedARA_svd', 'Oort_utility', 'JointAB', 'FedAvg_offline'] \
                  + ([] if a.no_full_baseline else ['FedAvg_full'])
    all_methods += [f'FedLoRA_r{r}' for r in ranks if f'FedLoRA_r{r}' not in all_methods]
    if a.methods:
        req = [x.strip() for x in a.methods.split(',')]
        methods = [x for x in all_methods if x in req]
    else:
        methods = all_methods[:4] + ([] if a.no_full_baseline else ['FedAvg_full'])
    oort_utils = {i: float(len(pp[i])) for i in range(a.clients)}  # Oort-style utility state
    # Frozen-A methods (everything except JointAB and the FedAvg family) keep
    # B-only server state; A never enters aggregation or server momentum.
    joint = [x for x in methods if x == 'JointAB']
    fedavg = [x for x in methods if x.startswith('FedAvg')]
    frozen = [x for x in methods if x not in joint and x not in fedavg]
    method_states = {x: {k: v.clone() for k, v in (initial_state.items() if x in joint else b_init.items())}
                     for x in methods}
    full_states = {x: {**{k: base_state[k].clone() for k in full_keys}, **{k: v.clone() for k, v in initial_state.items()}}
                   for x in fedavg}
    server_velocity = {x: {k: torch.zeros_like(v) for k, v in method_states[x].items()} for x in methods}
    full_velocity = {x: {k: torch.zeros_like(vv) for k, vv in s.items()} for x, s in full_states.items()}
    ck = os.path.join(a.out, 'checkpoint.pt'); start = 0
    if a.resume and os.path.exists(ck):
        z = torch.load(ck, map_location='cpu')  # server states live on cpu; cuda map breaks cpu-cuda arithmetic after resume
        method_states = z['method_states']; server_velocity = z.get('server_velocity', server_velocity)
        full_states = z.get('full_states', full_states); full_velocity = z.get('full_velocity', full_velocity)
        a_init = z.get('a_init', a_init)
        start = z['round'] + 1
    with open(os.path.join(a.out, 'config.json'), 'w') as f:
        mb = a.mobility_shape.split(',')
        json.dump(vars(a) | {'rmax': a.rmax, 'ranks': ranks, 'device': str(d),
                             'mobility_beta': [float(mb[0]), float(mb[1]) if len(mb) > 1 else 5.0],
                             'bytes_per_rank_b_only': bytes_per_rank, 'fedavg_full_mb': round(full_mb, 4),
                             'budget_grid_mb': BUDGET_GRID,
                             'backbone_pretrained': True, 'backbone_acc': base_acc}, f, indent=2)

    def round_lr(rnd):
        f = a.lr_final_frac
        return a.lr * (f + (1 - f) * .5 * (1 + math.cos(math.pi * rnd / max(1, a.rounds - 1))))

    evalm = new_model()  # per-method eval model
    sched_ms = [0.0]  # cumulative rank-decision time (ms)
    train_ms = [0.0]  # cumulative local-training wall time (ms)
    mob_shape = tuple(float(x) for x in a.mobility_shape.split(',')) if ',' in a.mobility_shape \
        else (float(a.mobility_shape), 5.0)
    if a.corr_skew:  # per-class-half eval sets for the selection-bias study
        te_y = np.asarray(te.targets); half = c // 2
        te_half = [np.where(te_y < half)[0].tolist(), np.where(te_y >= half)[0].tolist()]
        bf = open(os.path.join(a.out, 'bias_eval.csv'), 'a' if a.resume else 'w', newline='')
        bw = csv.writer(bf)
        if not a.resume:
            bw.writerow(['seed', 'round', 'method', 'acc_weak_link_classes', 'acc_strong_link_classes'])
    fn = os.path.join(a.out, 'metrics.csv')
    log_outcomes = a.link_profile == 'sagin'
    if log_outcomes:
        fo = open(os.path.join(a.out, 'link_outcomes.csv'), 'a' if start else 'w', newline='')
        wo = csv.writer(fo)
        if not start:
            wo.writerow(['seed', 'round', 'method', 'client', 'group', 'rank', 'delivered'])
    f = open(fn, 'a' if start else 'w', newline=''); w = csv.writer(f)
    if not start:
        w.writerow(['seed', 'round', 'method', 'accuracy', 'timely_ratio', 'bytes_mb', 'mean_rank',
                    'selected', 'aggregated', 'update_norm'])
    mm_log_rate = np.full(a.clients, 2.5)   # AR(1) log-rate state (mismatch arm)
    MM_AR_MU = 2.5                          # stationary mean = headline log-rate law
    mm_prev_real = None                      # last round's realized rates
    mm_est = np.full(a.clients, 12.0)        # EMA estimate from delivery history
    for rnd in range(start, a.rounds):
        if a.delivery_mismatch:
            if mm_prev_real is not None:
                mm_est = 0.5 * mm_est + 0.5 * mm_prev_real
            gr = np.random.default_rng(a.seed * 10000 + rnd + 500000)
            # Mean-stationary AR(1): reverts to MM_AR_MU so the stationary
            # marginal matches the headline LogNormal(2.5, 0.7) law.
            mm_log_rate = MM_AR_MU + 0.6 * (mm_log_rate - MM_AR_MU) + \
                gr.normal(0, 0.7 * math.sqrt(1 - 0.36), a.clients)
            mm_rate_real = np.clip(np.exp(mm_log_rate), 1, 40)
            mm_prev_real = mm_rate_real.copy()
        for method in methods:
            g = np.random.default_rng(a.seed * 10000 + rnd)  # common link traces across methods
            rate, mob = draw_links(a.link_profile, g, a.clients, mob_shape)
            rate_true, mob_true = rate, mob   # delivery always uses the truth
            if a.rate_est_bias:
                rate = rate_true * (1.0 + a.rate_est_bias)
            if a.mu_est_scale != 1.0:
                mob = mob_true * a.mu_est_scale
            if a.delivery_mismatch:
                rate = mm_est.copy()   # scheduler sees the stale history-based estimate
            is_joint = method == 'JointAB'
            is_full = method.startswith('FedAvg')
            if method == 'Oort_utility':
                # Oort-style selection: sample 2K candidates, admit the top-K by utility
                cand = g.choice(a.clients, min(2 * a.per_round, a.clients), replace=False)
                ch = cand[np.argsort(-np.asarray([oort_utils[i] for i in cand]))[:min(a.per_round, len(cand))]]
                assign = {}
            elif method == 'DP_cohort':
                # Full-scheduling arm: exact DP over ALL clients under the same
                # participation cap plus a byte budget; dummy draw keeps the CRN
                # stream aligned with the other methods' cohort sampling.
                g.choice(a.clients, min(a.per_round, a.clients), replace=False)
                assign = dp_assign(mob, rate, a.deadline, ranks, a.rmax, list(range(a.clients)),
                                   min(a.per_round, a.clients), a.dp_budget, bytes_per_rank)
                ch = np.array(sorted(assign.keys()))
            elif method == 'DP_same':
                # Same-cohort DP arm: identical cohort draw as the fair-cohort
                # methods, exact DP restricted to those candidates.
                ch = g.choice(a.clients, min(a.per_round, a.clients), replace=False)
                assign = dp_assign(mob, rate, a.deadline, ranks, a.rmax, list(map(int, ch)),
                                   min(a.per_round, a.clients), a.dp_budget, bytes_per_rank)
                ch = np.array(sorted(assign.keys()))  # only admitted candidates train
            elif method in ('Greedy_value', 'Greedy_vpb'):
                # Budget-aware greedy arms over the full pool (binding-budget
                # study); dummy draw keeps the CRN stream aligned.
                g.choice(a.clients, min(a.per_round, a.clients), replace=False)
                assign = greedy_assign(mob, rate, a.deadline, ranks, a.rmax, list(range(a.clients)),
                                       min(a.per_round, a.clients), a.dp_budget, bytes_per_rank,
                                       per_byte=(method == 'Greedy_vpb'))
                ch = np.array(sorted(assign.keys()))
            else:
                ch = g.choice(a.clients, min(a.per_round, a.clients), replace=False)
                assign = {}
            ss, ww, rr, deltas = [], [], [], []
            used = ok = 0
            for i in ch:
                t0 = time.perf_counter()
                r = assign.get(int(i)) if method in ('DP_cohort', 'DP_same', 'Greedy_value', 'Greedy_vpb') else \
                    rank_for(method, int(i), mob, rate, a.deadline, g, ranks, a.rmax, rnd, a.rounds,
                             a.no_capacity_term, bytes_per_rank, full_mb)
                sched_ms[0] += (time.perf_counter() - t0) * 1e3
                local = new_model(); load_base(local, base_state)
                set_eff_r(local, r)  # rank-r local training: only the first r coordinates live
                lr_r = round_lr(rnd) * (a.full_lr_scale if is_full else 1.0)
                if is_full:
                    load(local, {k: v for k, v in full_states[method].items() if k.endswith(('.A', '.B'))})
                    load_base(local, {k: full_states[method][k] for k in full_keys})
                    for n_, p_ in local.named_parameters():
                        p_.requires_grad = not (n_.endswith('.A') or n_.endswith('.B'))
                    params = [local.get_parameter(k) for k in full_keys]
                    anchor = {k: full_states[method][k].to(d) for k in full_keys}
                else:
                    load(local, a_init); load(local, method_states[method])
                    params = local.lora_params()
                    if not is_joint:
                        # frozen A: local training only touches B[: , :r]
                        for n_, p_ in local.named_parameters():
                            if n_.endswith('.A'): p_.requires_grad = False
                    anchor = {k: v.to(d) for k, v in method_states[method].items()}
                # Plain local SGD: client momentum amplifies non-IID drift.
                opt = torch.optim.SGD(params, lr=lr_r, weight_decay=a.weight_decay)
                dl = DataLoader(Subset(tr_client, pp[i]), a.batch_size, shuffle=True)
                local.train()
                anchor_pairs = [(local.get_parameter(k), anchor[k]) for k in anchor]
                amp = d.type == 'cuda'
                t_train = time.perf_counter()
                for _ in range(a.local_epochs):
                    for x, y in dl:
                        opt.zero_grad()
                        with torch.autocast(d.type, dtype=torch.bfloat16, enabled=amp):
                            loss = nn.functional.cross_entropy(local(x.to(d)), y.to(d))
                            if a.prox > 0:  # FedProx anchor tames client drift
                                loss = loss + .5 * a.prox * sum(((p_ - t_) ** 2).sum() for p_, t_ in anchor_pairs)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, a.clip_grad); opt.step()
                train_ms[0] += (time.perf_counter() - t_train) * 1e3
                size = full_mb if is_full else size_mb(r, bytes_per_rank) * (.5 if a.fp16_upload else 1.0)
                if is_joint and a.jointab_ab:
                    size = r * joint_bytes_per_rank / 1e6  # true A+B payload at the audited ratio
                used += size
                if a.delivery_mismatch:
                    d_prop, p_h, dlt_h = 0.005, 0.15, 0.02
                    T = d_prop + (dlt_h if g.random() < p_h else 0.0) + 8.0 * size / mm_rate_real[i]
                    lost = g.random() < (1.0 - math.exp(-1.5 * mob_true[i]))
                    delivered = bool(T <= a.deadline and not lost)
                else:
                    pon = 1.0 if method == 'FedAvg_offline' else \
                        math.exp(-mob_true[i]) * math.exp(-8.0 * size / (rate_true[i] * a.deadline))
                    delivered = bool(g.random() < pon)
                if log_outcomes:
                    wo.writerow([a.seed, rnd, method, int(i), SAGIN_GROUPS[group_of(int(i), a.clients)],
                                 r, int(delivered)])
                if delivered:
                    if is_full:
                        cur = full_state(local, full_keys) | ls(local)
                        before = full_states[method]
                        deltas.append(frob({k: cur[k] - before[k] for k in full_keys}))
                        ss.append(cur)
                    else:
                        cur = ls_b(local) if not is_joint else ls(local)
                        if a.fp16_upload and not is_joint:
                            # fp16 compression baseline: real round-trip quantization
                            cur = {k: v.half().float() for k, v in cur.items()}
                        deltas.append(frob({k: cur[k] - method_states[method][k] for k in cur}))
                        ss.append(cur)
                    ww.append(len(pp[i])); rr.append(r); ok += 1
                    if method == 'Oort_utility':
                        oort_utils[i] = len(pp[i]) * (1.0 + deltas[-1])
            if ss:
                if is_full:
                    cand = avg_full(ss, ww)
                    for k in cand:
                        delta = cand[k] - full_states[method][k]
                        full_velocity[method][k] = a.server_momentum * full_velocity[method][k] + (1 - a.server_momentum) * delta
                        cand[k] = full_states[method][k] + full_velocity[method][k]
                    full_states[method] = cand
                elif is_joint:
                    cand = avg_ab(ss, ww)
                    for k in cand:
                        delta = cand[k] - method_states[method][k]
                        server_velocity[method][k] = a.server_momentum * server_velocity[method][k] + (1 - a.server_momentum) * delta
                        cand[k] = method_states[method][k] + server_velocity[method][k]
                    method_states[method] = cand
                else:
                    prev_state = {k: method_states[method][k] for k in ss[0]}
                    cand = avg_b(ss, ww, rr, use_mask=not a.no_rank_mask,
                                 prev=prev_state if a.empty_cols == 'hold' else None)
                    if method == 'FedARA_svd':
                        # next round's global rank from this round's aggregated delta
                        FEDARA['rank'] = svd_energy_rank(cand, prev_state, ranks)
                    for k in cand:  # server momentum damps client-sampling/link noise (B only; A stays frozen)
                        delta = cand[k] - method_states[method][k]
                        server_velocity[method][k] = a.server_momentum * server_velocity[method][k] + (1 - a.server_momentum) * delta
                        cand[k] = method_states[method][k] + server_velocity[method][k]
                    method_states[method] = cand
            # Evaluate THIS method's own post-round state (aggregated if any
            # delivery arrived, else its previous server state).
            load_base(evalm, base_state)
            if is_full:
                st = full_states[method]
                load_base(evalm, {k: st[k] for k in full_keys})
                load(evalm, {k: st[k] for k in st if k.endswith(('.A', '.B'))})
            else:
                st = method_states[method]
                load(evalm, a_init); load(evalm, st)
            acc = ev(evalm, te_eval, d)
            if a.corr_skew:
                bw.writerow([a.seed, rnd, method,
                             round(ev(evalm, Subset(te_eval, te_half[0]), d), 4),
                             round(ev(evalm, Subset(te_eval, te_half[1]), d), 4)])
            w.writerow([a.seed, rnd, method, acc, ok / max(1, len(ch)), used,
                        (float(np.mean(rr)) if rr else ''), len(ch), ok,
                        (float(np.mean(deltas)) if deltas else '')]); f.flush()
        if log_outcomes:
            fo.flush()
        torch.save({'round': rnd, 'model': m.state_dict(), 'method_states': method_states,
                    'server_velocity': server_velocity, 'full_states': full_states,
                    'full_velocity': full_velocity, 'a_init': a_init}, ck)
    f.close()
    if a.corr_skew:
        bf.close()
    if log_outcomes:
        fo.close()
    cfg = json.load(open(os.path.join(a.out, 'config.json')))
    cfg['sched_ms_total'] = round(sched_ms[0], 2)
    cfg['train_ms_total'] = round(train_ms[0], 1)
    cfg['local_epochs_total'] = a.rounds * len(methods) * a.per_round
    json.dump(cfg, open(os.path.join(a.out, 'config.json'), 'w'), indent=2)
    print(f"[sched] rank decisions: {sched_ms[0]:.1f} ms total over the run", flush=True)

if __name__ == '__main__':
    main()
