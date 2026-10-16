"""B6/B7 operator-currency translations for the delivered-capacity story.

B6 (energy): under the standard linear transmit-energy model E = P_tx * 8B/R
(P_tx constant during airtime), energy ratios equal byte ratios, and delivered
capacity per joule equals dc_meas per attempted byte -- both computable from
the logs without any radio constant. The 23 dBm (~0.2 W) smartphone uplink
power is used only to translate the ratio into absolute joules per MB.

B7 (bandwidth equivalence): calibrating the hazard model's (mu, R*tau) from the
two measured timely ratios of fixed r4 and fixed r16 at the stress deadline,
then solving for the rate multiplier m that lifts fixed r16's predicted timely
ratio to FedRAS's measured one.

Reads outputs/journal_stats/delivered_capacity.csv (per round). Writes
outputs/journal_stats/energy_bw.csv. Writes nothing else.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS = os.path.join(ROOT, 'outputs', 'journal_stats')
FAMILY = 'cifar10_dl0.1'
P_TX_W = 0.2          # 23 dBm typical smartphone uplink transmit power
RATE_MBPS = 8.0       # reference airtime rate for the joule illustration
B_R4_MB, B_R16_MB = 0.2253, 0.9011   # serialized payloads, tab:audit


def terminal(family):
    d = pd.read_csv(os.path.join(STATS, 'delivered_capacity.csv'))
    d = d[(d.family == family) & (d['round'] >= 30) & (d['round'] <= 39)]
    done = d.groupby('run')['round'].max()
    d = d[d['run'].isin(done[done == 39].index)]
    g = (d.groupby(['method', 'seed'])
           .agg(dc=('dc_meas', 'mean'), mb=('bytes_mb', 'mean'),
                timely=('timely_ratio', 'mean'))
           .reset_index())
    return g.groupby('method').agg(n=('dc', 'size'), dc=('dc', 'mean'),
                                   mb=('mb', 'mean'), timely=('timely', 'mean'))


def main():
    t = terminal(FAMILY)
    j_per_mb = P_TX_W * 8.0 / RATE_MBPS   # 8 bits per byte, rate in Mbps -> seconds per MB
    rows = []
    for m, r in t.iterrows():
        rows.append(dict(family=FAMILY, method=m, n=int(r.n),
                         timely=r.timely, attempted_mb=r.mb,
                         dc_per_mb=r.dc / r.mb,
                         delivered_mb=r.mb * r.timely,
                         wasted_energy_frac=1.0 - r.timely,
                         j_per_round=r.mb * j_per_mb))
    tab = pd.DataFrame(rows)
    fr = tab[tab.method == 'Adaptive_rank'].iloc[0]
    r16 = tab[tab.method == 'FedLoRA_r16'].iloc[0]
    r4 = tab[tab.method == 'FedLoRA_r4'].iloc[0]
    dc_per_j_rel = 100 * (fr.dc_per_mb / r16.dc_per_mb - 1)
    wasted_saved = r16.wasted_energy_frac - fr.wasted_energy_frac

    # B7: solve (mu, 8/(R*tau)) from measured r4/r16 timely ratios, then the multiplier
    # ln P = -mu - (8/(R*tau)) * B  with B in MB
    lo4, lo16, lofr = np.log(r4.timely), np.log(r16.timely), np.log(fr.timely)
    inv_rtau = (lo4 - lo16) / (B_R16_MB - B_R4_MB)   # = 8/(R*tau), per MB
    mu = -lo4 - inv_rtau * B_R4_MB
    m_mult = inv_rtau * B_R16_MB / (-lofr - mu) if (-lofr - mu) > 0 else np.inf

    pd.DataFrame([dict(item='dc_per_joule_vs_r16_pct', value=dc_per_j_rel),
                  dict(item='wasted_airtime_saved_vs_r16_frac', value=wasted_saved),
                  dict(item='j_per_mb_reference', value=j_per_mb),
                  dict(item='bw_multiplier_r16_to_fedras_timely', value=m_mult),
                  dict(item='hazard_mu_calibrated', value=mu),
                  dict(item='hazard_Rtau_MB', value=8.0 / inv_rtau if inv_rtau else np.nan)]
                 ).to_csv(os.path.join(STATS, 'energy_bw.csv'), index=False)

    print(f'== {FAMILY} terminal window, transmit-energy model (P_tx={P_TX_W} W, R={RATE_MBPS} Mbps) ==')
    print(tab.round(4).to_string(index=False))
    print(f'\nB6: delivered capacity per transmit joule vs fixed r16: {dc_per_j_rel:+.1f}%')
    print(f'    wasted airtime fraction: r16 {r16.wasted_energy_frac:.2f} -> FedRAS {fr.wasted_energy_frac:.2f} '
          f'({wasted_saved:+.2f})')
    print(f'B7: fixed r16 needs {m_mult:.2f}x uplink rate to match FedRAS timely {fr.timely:.2f} '
          f'(calibrated mu={mu:.3f}, R*tau={8.0/inv_rtau:.1f} MB)')


if __name__ == '__main__':
    main()
