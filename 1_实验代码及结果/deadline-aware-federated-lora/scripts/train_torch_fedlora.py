"""Deadline-aware adaptive-rank FedLoRA trainer (v2).

Key changes vs the first stable version (kept in .before_stability / zip):
1. CNN backbone with BatchNorm replaces the raw-pixel MLP; LoRA adapters sit
   on two wide head layers (width >= 128) so the rank r in {2,4,8,16} actually
   bounds the update capacity of every adapted matrix.
2. Standard recipe: per-dataset normalization stats, crop/flip augmentation
   for CIFAR pretraining, SGD+Nesterov+weight-decay, cosine LR schedules for
   both the centralized pretraining and the federated rounds.
3. Adaptive_rank now maximizes an expected useful-update objective
   U(r) = capacity_gain(r) * P_deliver(r) with saturating capacity_gain(r) =
   sqrt(r/rmax) and deadline-aware P_deliver(r) = exp(-mob)*exp(-size/(rate*deadline)),
   matching the simulator's link model. The old score was monotone
   decreasing in r and therefore always selected rank 2.
4. Delivery/aggregation unchanged in spirit: masked weighted averaging in the
   shared rmax coordinate system (FLoRA-style heterogeneous-rank aggregation)
   plus server momentum.
5. Optional FedAvg_full baseline: full head fine-tuning at rank-64 payload
   cost, mirroring the simulator's FedAvg representation.
6. Backbone cache is keyed by a config fingerprint so stale pretrained
   backbones are never silently reused.
7. New metric column update_norm: mean Frobenius norm of delivered LoRA
   deltas, a proxy for the update-exposure surface (gradient-inversion
   literature on LoRA fine-tuning).
"""

import argparse, csv, json, math, os, random, time
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

RANKS = [2, 4, 8, 16]
RMAX = 16

class LoRALinear(nn.Module):
    def __init__(self, base, rmax=RMAX):
        super().__init__()
        self.base, self.rmax = base, rmax
        for p in base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(rmax, base.in_features) * .01)
        self.B = nn.Parameter(torch.zeros(base.out_features, rmax))
    def forward(self, x):
        return self.base(x) + (x @ self.A.t()) @ self.B.t()
    def extra_repr(self):
        return f'rmax={self.rmax}'

class LoRAConv2d(nn.Module):
    """Low-rank conv delta via 1x1 factorization: base(x) + B(A(x))."""
    def __init__(self, base, rmax=RMAX):
        super().__init__()
        self.base, self.rmax = base, rmax
        for p in base.parameters():
            p.requires_grad = False
        k = base.kernel_size
        assert k == (3, 3), 'LoRA delta uses 1x1 factors; base stays 3x3'
        self.A = nn.Parameter(torch.randn(rmax, base.in_channels, 1, 1) * .01)
        self.B = nn.Parameter(torch.zeros(base.out_channels, rmax, 1, 1))
    def forward(self, x):
        return self.base(x) + nn.functional.conv2d(
            nn.functional.conv2d(x, self.A), self.B,
            stride=self.base.stride, padding=0)
    def extra_repr(self):
        return f'rmax={self.rmax}'

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
        for x, y in DataLoader(ds, batch, shuffle=True, num_workers=2, persistent_workers=False):
            opt.zero_grad()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            loss.backward(); opt.step()
        sch.step()

def ls(m): return {k: v.detach().cpu().clone() for k, v in m.state_dict().items() if k.endswith('.A') or k.endswith('.B')}
def load(m, s):
    z = m.state_dict()
    for k, v in s.items(): z[k].copy_(v)
    m.load_state_dict(z)
def load_base(m, s):
    z = m.state_dict()
    for k, v in s.items():
        if k in z: z[k].copy_(v)
    m.load_state_dict(z)
def avg(ss, ww, rr, use_mask=True):
    # Aggregate LoRA states in the shared rmax coordinate system; each
    # coordinate is normalized by the clients that actually activate it
    # (FLoRA-style heterogeneous-rank aggregation without rank dilution).
    # Leading-dim slicing covers both 2D linear and 4D conv factors.
    o = {}
    for k in ss[0]:
        z = torch.zeros_like(ss[0][k]); den = torch.zeros_like(ss[0][k])
        for s, w, r in zip(ss, ww, rr):
            v = s[k].clone()
            if k.endswith('.A'): v[r:] = 0
            if k.endswith('.B'): v[:, r:] = 0
            mask = torch.ones_like(v)
            if k.endswith('.A'): mask[r:] = 0
            if k.endswith('.B'): mask[:, r:] = 0
            z += w * v; den += w * mask
        if not use_mask:  # ablation: plain weighted average, no rank masking
            den = torch.full_like(den, sum(ww))
        o[k] = torch.where(den > 0, z / den, torch.zeros_like(z))
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

def ev(m, ds, d, batch=512):
    m.eval(); n = o = 0
    amp = d.type == 'cuda'
    with torch.no_grad():
        for x, y in DataLoader(ds, batch):
            with torch.autocast(d.type, dtype=torch.bfloat16, enabled=amp):
                out = m(x.to(d))
            o += (out.argmax(1).cpu() == y).sum().item(); n += len(y)
    return o / max(1, n)

def rank_for(method, i, mob, rate, deadline, g, ranks, rmax, rnd=0, rounds=1, no_capacity=False):
    size = lambda r: .02 * r + .5  # MB
    # Delivery hazard; 8*size/rate converts MB payload against Mbps link and
    # deadline seconds, so large ranks are genuinely deadline-critical.
    pon = lambda r: math.exp(-mob[i]) * math.exp(-8.0 * size(r) / (rate[i] * deadline))
    cap = lambda r: math.sqrt(min(r, rmax) / rmax)
    if method.startswith('FedLoRA_r'): return int(method.split('_r')[1])
    if method == 'Random_rank': return int(g.choice(ranks))
    if method == 'FedAvg_full': return 64
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
    raise ValueError(method)

def dp_assign(mob, rate, deadline, ranks, rmax, clients, cap, budget, step=0.05):
    """Exact per-round client-rank assignment over ALL clients: maximize
    sum_i G(r_i)*P_on(i,r_i) s.t. sum s(r_i) <= budget MB and at most `cap`
    admissions (multiple-choice knapsack with cardinality constraint).
    Budget grid step 0.05 MB, as in the paper's solver study. Returns {i: r}."""
    size = lambda r: .02 * r + .5
    Bu = int(round(budget / step))
    NEG = -1e9
    V = np.full((cap + 1, Bu + 1), NEG); V[0, 0] = 0.0
    back = []
    for i in range(clients):
        best = V.copy(); act = np.full((cap + 1, Bu + 1), -1, dtype=np.int64)
        fail = math.exp(-mob[i])
        for r in ranks:
            c = max(1, int(math.ceil(size(r) / step - 1e-9)))
            if c > Bu: continue
            val = fail * math.exp(-8.0 * size(r) / (rate[i] * deadline)) * math.sqrt(min(r, rmax) / rmax)
            shifted = np.full_like(V, NEG)
            shifted[1:, c:] = V[:-1, :Bu + 1 - c] + val
            upd = shifted > best
            best = np.where(upd, shifted, best)
            act[upd] = r
        back.append(act); V = best
    k, b = np.unravel_index(np.argmax(V), V.shape)
    assign = {}
    for i in reversed(range(clients)):
        r = int(back[i][k, b])
        if r > 0:
            assign[i] = r
            b -= max(1, int(math.ceil(size(r) / step - 1e-9))); k -= 1
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
    p.add_argument('--deadline', type=float, default=1.0)
    p.add_argument('--mobility-shape', type=float, default=2.0, help='Beta shape of per-round client mobility; lower = more mobile')
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--base-channels', type=int, default=64, help='width of the first conv block; blocks are ch/2ch/4ch')
    p.add_argument('--lora-init-std', type=float, default=.1, help='std of the A init; larger values break the B=0 cold start faster')
    p.add_argument('--full-lr-scale', type=float, default=.4, help='lr scale for the FedAvg_full head baseline')
    p.add_argument('--prox', type=float, default=0., help='FedProx coefficient anchoring local params to the server state')
    p.add_argument('--no-freeze-a', action='store_true', help='train both A and B (default freezes A at the shared init, FFA-LoRA style, so averaging B equals averaging the update)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--no-reset-lora', action='store_true', help='start federated LoRA from pretrained A/B instead of fresh init')
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--no-full-baseline', action='store_true', help='skip the FedAvg_full head fine-tuning baseline')
    p.add_argument('--rmax', type=int, default=16, help='provisioned adapter factor width')
    p.add_argument('--methods', default='', help='comma-separated subset of methods to run (default: all standard)')
    p.add_argument('--no-rank-mask', action='store_true', help='ablation: plain weighted averaging without rank masking')
    p.add_argument('--no-capacity-term', action='store_true', help='ablation: adaptive rule uses delivery probability only (no capacity gain)')
    p.add_argument('--dp-budget', type=float, default=8.2, help='byte budget (normalized MB) for the DP_cohort full-scheduling arm')
    p.add_argument('--resume', action='store_true')
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
    pp = parts(tr.targets, a.clients, a.seed, a.alpha)

    ranks = [r for r in (2, 4, 8, 16, 32) if r <= a.rmax]
    m = Net(c, in_ch, a.width, rmax=a.rmax, base_ch=a.base_channels).to(d)
    # Backbone cache keyed by config fingerprint (never silently reuse stale weights).
    fp = {'arch': 'cnn_lora_v3', 'dataset': a.dataset, 'width': a.width, 'base_channels': a.base_channels, 'pretrain_epochs': a.pretrain_epochs,
          'pretrain_lr': a.pretrain_lr, 'batch_size': a.batch_size, 'augment': augment, 'seed': a.seed}
    pre = os.path.join(a.out, 'backbone_pretrained.pt'); meta = os.path.join(a.out, 'backbone_meta.json')
    if os.path.exists(pre) and os.path.exists(meta) and json.load(open(meta)) == fp:
        # base weights only: adapter shapes depend on --rmax and are fresh anyway
        sd = torch.load(pre, map_location=d)
        m.load_state_dict({k: v for k, v in sd.items() if not (k.endswith('.A') or k.endswith('.B'))}, strict=False)
    else:
        print('pretraining backbone...', flush=True)
        pretrain_backbone(m, tr, d, a.pretrain_epochs, a.pretrain_lr, a.batch_size)
        torch.save(m.state_dict(), pre); json.dump(fp, open(meta, 'w'))
    base_acc = ev(m, te, d); print(f'pretrained backbone test acc = {base_acc:.4f}', flush=True)

    base_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()
                  if not (k.endswith('.A') or k.endswith('.B'))}
    if a.no_reset_lora:
        initial_state = ls(m)
    else:  # fresh adapters: the LoRA delta must be re-learned federatively
        initial_state = {k: (torch.randn_like(v) * a.lora_init_std if k.endswith('.A') else torch.zeros_like(v))
                         for k, v in ls(m).items()}
    all_methods = ['FedLoRA_r4', f'FedLoRA_r{ranks[-1]}', 'Random_rank', 'Adaptive_rank',
                   'DP_cohort', 'SA_rank', 'Oort_utility', 'JointAB'] + ([] if a.no_full_baseline else ['FedAvg_full'])
    if a.methods:
        req = [x.strip() for x in a.methods.split(',')]
        methods = [x for x in all_methods if x in req]
    else:
        methods = all_methods[:4] + ([] if a.no_full_baseline else ['FedAvg_full'])
    oort_utils = {i: float(len(pp[i])) for i in range(a.clients)}  # Oort-style utility state
    method_states = {x: {k: v.clone() for k, v in initial_state.items()} for x in methods}
    # FedAvg_full fine-tunes every base parameter (conv + head + out) at
    # rank-64 payload cost, mirroring the simulator's FedAvg representation.
    full_keys = [n for n, _ in m.named_parameters() if not (n.endswith('.A') or n.endswith('.B'))]
    full_states = {x: {**{k: base_state[k].clone() for k in full_keys}, **{k: v.clone() for k, v in initial_state.items()}}
                   for x in ([] if a.no_full_baseline else ['FedAvg_full'])}
    server_velocity = {x: {k: torch.zeros_like(v) for k, v in initial_state.items()} for x in methods}
    full_velocity = {x: {k: torch.zeros_like(vv) for k, vv in s.items()} for x, s in full_states.items()}
    ck = os.path.join(a.out, 'checkpoint.pt'); start = 0
    if a.resume and os.path.exists(ck):
        z = torch.load(ck, map_location=d)
        method_states = z['method_states']; server_velocity = z.get('server_velocity', server_velocity)
        full_states = z.get('full_states', full_states); full_velocity = z.get('full_velocity', full_velocity)
        start = z['round'] + 1
    with open(os.path.join(a.out, 'config.json'), 'w') as f:
        json.dump(vars(a) | {'rmax': a.rmax, 'ranks': ranks, 'device': str(d),
                             'backbone_pretrained': True, 'backbone_acc': base_acc}, f, indent=2)

    def round_lr(rnd):
        f = a.lr_final_frac
        return a.lr * (f + (1 - f) * .5 * (1 + math.cos(math.pi * rnd / max(1, a.rounds - 1))))

    sched_ms = [0.0]  # cumulative rank-decision time (ms)
    fn = os.path.join(a.out, 'metrics.csv')
    f = open(fn, 'a' if start else 'w', newline=''); w = csv.writer(f)
    if not start:
        w.writerow(['seed', 'round', 'method', 'accuracy', 'timely_ratio', 'bytes_norm', 'mean_rank',
                    'selected', 'aggregated', 'update_norm'])
    for rnd in range(start, a.rounds):
        for method in methods:
            g = np.random.default_rng(a.seed * 10000 + rnd)  # common link traces across methods
            rate = np.clip(g.lognormal(2.5, .7, a.clients), 1, 40); mob = g.beta(a.mobility_shape, 5, a.clients)
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
                assign = dp_assign(mob, rate, a.deadline, ranks, a.rmax, a.clients,
                                   min(a.per_round, a.clients), a.dp_budget)
                ch = np.array(sorted(assign.keys()))
            else:
                ch = g.choice(a.clients, min(a.per_round, a.clients), replace=False)
                assign = {}
            ss, ww, rr, deltas = [], [], [], []
            used = ok = 0
            for i in ch:
                t0 = time.perf_counter()
                r = assign[i] if method == 'DP_cohort' else rank_for(method, i, mob, rate, a.deadline, g, ranks, a.rmax, rnd, a.rounds, a.no_capacity_term)
                sched_ms[0] += (time.perf_counter() - t0) * 1e3
                local = Net(c, in_ch, a.width, rmax=a.rmax, base_ch=a.base_channels).to(d); load_base(local, base_state)
                lr_r = round_lr(rnd) * (a.full_lr_scale if method == 'FedAvg_full' else 1.0)
                if method == 'FedAvg_full':
                    load(local, {k: full_states[method][k] for k in ('fc1.A', 'fc1.B', 'fc2.A', 'fc2.B')})
                    load_base(local, {k: full_states[method][k] for k in full_keys})
                    for n_, p_ in local.named_parameters():
                        p_.requires_grad = not (n_.endswith('.A') or n_.endswith('.B'))
                    params = [local.get_parameter(k) for k in full_keys]
                    anchor = {k: full_states[method][k].to(d) for k in full_keys}
                else:
                    load(local, method_states[method]); params = local.lora_params()
                    if not a.no_freeze_a and method != 'JointAB':
                        # frozen A: averaging B == averaging the update
                        for n_, p_ in local.named_parameters():
                            if n_.endswith('.A'): p_.requires_grad = False
                    anchor = {k: v.to(d) for k, v in method_states[method].items()}
                # Plain local SGD: client momentum amplifies non-IID drift.
                opt = torch.optim.SGD(params, lr=lr_r, weight_decay=a.weight_decay)
                dl = DataLoader(Subset(tr, pp[i]), a.batch_size, shuffle=True)
                local.train()
                anchor_pairs = [(local.get_parameter(k), anchor[k]) for k in anchor]
                amp = d.type == 'cuda'
                for _ in range(a.local_epochs):
                    for x, y in dl:
                        opt.zero_grad()
                        with torch.autocast(d.type, dtype=torch.bfloat16, enabled=amp):
                            loss = nn.functional.cross_entropy(local(x.to(d)), y.to(d))
                            if a.prox > 0:  # FedProx anchor tames client drift
                                loss = loss + .5 * a.prox * sum(((p_ - t_) ** 2).sum() for p_, t_ in anchor_pairs)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, a.clip_grad); opt.step()
                size = .02 * r + .5; used += size
                pon = math.exp(-mob[i]) * math.exp(-8.0 * size / (rate[i] * a.deadline))
                if g.random() < pon:
                    if method == 'FedAvg_full':
                        cur = full_state(local, full_keys) | ls(local)
                        before = full_states[method]
                        deltas.append(frob({k: cur[k] - before[k] for k in full_keys}))
                        ss.append(cur)
                    else:
                        cur = ls(local)
                        deltas.append(frob({k: cur[k] - method_states[method][k] for k in cur}))
                        ss.append(cur)
                    ww.append(len(pp[i])); rr.append(r); ok += 1
                    if method == 'Oort_utility':
                        oort_utils[i] = len(pp[i]) * (1.0 + deltas[-1])
            if ss:
                if method == 'FedAvg_full':
                    cand = avg_full(ss, ww)
                    for k in cand:
                        delta = cand[k] - full_states[method][k]
                        full_velocity[method][k] = a.server_momentum * full_velocity[method][k] + (1 - a.server_momentum) * delta
                        cand[k] = full_states[method][k] + full_velocity[method][k]
                    full_states[method] = cand
                    load_base(m, {k: cand[k] for k in full_keys}); load(m, {k: cand[k] for k in cand if k.endswith(('.A', '.B'))})
                else:
                    cand = avg(ss, ww, rr, use_mask=not a.no_rank_mask)
                    for k in cand:  # server momentum damps client-sampling/link noise
                        delta = cand[k] - method_states[method][k]
                        server_velocity[method][k] = a.server_momentum * server_velocity[method][k] + (1 - a.server_momentum) * delta
                        cand[k] = method_states[method][k] + server_velocity[method][k]
                    method_states[method] = cand
                    load_base(m, base_state); load(m, cand)
            w.writerow([a.seed, rnd, method, ev(m, te, d), ok / max(1, len(ch)), used,
                        float(np.mean(rr) if rr else 0), len(ch), len(ss),
                        float(np.mean(deltas) if deltas else 0.)]); f.flush()
        torch.save({'round': rnd, 'model': m.state_dict(), 'method_states': method_states,
                    'server_velocity': server_velocity, 'full_states': full_states,
                    'full_velocity': full_velocity}, ck)
    f.close()
    cfg = json.load(open(os.path.join(a.out, 'config.json')))
    cfg['sched_ms_total'] = round(sched_ms[0], 2)
    json.dump(cfg, open(os.path.join(a.out, 'config.json'), 'w'), indent=2)
    print(f"[sched] rank decisions: {sched_ms[0]:.1f} ms total over the run", flush=True)

if __name__ == '__main__':
    main()
