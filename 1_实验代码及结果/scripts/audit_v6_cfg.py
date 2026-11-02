import glob, json, os
from collections import OrderedDict
R='/root/autodl-tmp/FedRAS/outputs/real_v6'
KEYS=['dataset','deadline','pretrain_epochs','rounds','per_round','clients','lr','weight_decay',
      'lora_init_std','alpha','mobility_shape','dp_budget','rmax','server_momentum','clip_grad',
      'batch_size','local_epochs','methods','seed','backbone_acc']
groups=OrderedDict()
for d in sorted(glob.glob(f'{R}/*/')):
    name=os.path.basename(d.rstrip('/'))
    if name in ('bb_cache','sim','figs_v6'): continue
    base=name.rsplit('_seed',1)[0]
    groups.setdefault(base,[]).append(d)
print(f'{"group":42s} {"n":>2s} ' + ' '.join(f'{k[:9]:>10s}' for k in KEYS))
for g,ds in groups.items():
    cfg=json.load(open(os.path.join(ds[0],'config.json')))
    vals=[]
    for k in KEYS:
        v=cfg.get(k,'-')
        if isinstance(v,float): v=f'{v:g}'
        if isinstance(v,list): v=','.join(map(str,v))
        s=str(v)
        vals.append(f'{s[:10]:>10s}')
    print(f'{g:42s} {len(ds):2d} ' + ' '.join(vals))
