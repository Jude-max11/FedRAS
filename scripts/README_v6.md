# v6 重跑：协议、脚本与核对

本目录下的脚本对应 2026-11 的 v6 全量重跑，论文中表 3、表 4、表 5(a)(b)(c)(d)、表 6(a)(b)(c)
与图 2 至图 5 的全部数字均来自这一次重跑，没有复用任何旧结果。

## 一、协议（与论文 6.1 一致）

| 项 | 取值 |
|---|---|
| 客户端 / 每轮参与者 | 20 / 10 |
| 轮数 | 40 |
| 本地训练 | 1 个 epoch，plain SGD（不带动量），lr 0.005 余弦退火到 5%，weight decay 1e-4，batch 128，梯度裁剪 1.0 |
| 服务器 | 掩码聚合，动量 0.3 |
| 候选秩 / 供给宽度 | {2, 4, 8, 16} / 16 |
| 每轮字节预算 | 10 个客户端全取秩 16 所需载荷的 1.1 倍，约 1.0 MB |
| 骨干 | 六层卷积，预训练 6 个 epoch，CIFAR-10 测试准确率 0.839 |
| 链路 | LogNormal(2.5, 0.7) Mbps（截断 [1,40]），移动性 Beta(2, 5) |
| 种子 | 0 至 9，配对；同一种子决定划分、骨干初始化与链路 |

注意：`--pretrain-epochs` 的代码默认值是 6。`run_v6_fedavg.sh`、`run_v6_bias.sh`、`run_v6_tab5d_full.sh`
三个旧脚本曾写死 `--pretrain-epochs 30`，与上表不一致，已由 `run_v6_fedavg6.sh`、`run_v6_bias6.sh`、
`run_v6_tab5df.sh` 取代；旧脚本保留仅供追溯，不应再运行。

## 二、目录与产出（共 910 个 run）

| 脚本 | 产出目录 | 论文位置 |
|---|---|---|
| `run_v6_matrix.sh` (p1a) | `p1_tab3_*`（90） | 表 3 |
| `run_v6_matrix_v2.sh` (p1b…) | `p1_tab4_*`（30） | 表 4 |
| `run_v6_chain2.sh` | 串行驱动 p1b、p2a、p2b、p3a、p3c、p3d | — |
| `run_v6_matrix_v2.sh` (p2a) | `p2_tab5c_*`（20） | 表 5(c) |
| `run_v6_matrix_v2.sh` (p2b) | `p2_tab5d_*`（120） | 表 5(d) 四方法 |
| `run_v6_tab5df.sh` | `p2_tab5df_*`（120） | 表 5(d) 补齐 SA/Oort/EL |
| `run_v6_matrix_b.sh` | `p3_tab6b_*`（60）、`p3_tab6c_*`（50） | 表 6(b)(c) |
| `run_v6_matrix_v2.sh` (p3a) | `p3_tab6a_*`（60） | 表 6(a) |
| `run_v6_matrix_v2.sh` (p3c/p3d) | `p3_fig4_*`（40）、`p3_fig5_*`（20） | 图 4、图 5(a) |
| `run_v6_fedavg6.sh` | `p5_fedavg6_*`（30） | 6.1 参考点 |
| `run_v6_bias6.sh` | `p6b_*`（120） | 6.4 估计误差 |
| `fedlora/run_experiment.py` | `outputs/real_v6/sim/*`（6 组） | 表 5(a) |
| `scripts/dp_scheduler_study.py` | `solver_study.csv` | 表 5(b) |

## 三、汇总、作图与核对

| 脚本 | 用途 |
|---|---|
| `collect_v6.py` | 从 `metrics.csv` 汇总各表的均值与标准差 |
| `emit_tab3_zh.py` | 直接生成表 3 的 LaTeX 行 |
| `agg_5ab.py` | 汇总模拟器与求解器到表 5(a)(b) |
| `report_pending.py` | 表 5(d) 七方法对照与误差实验汇总 |
| `make_v6_figs.py` | 图 2、图 3 |
| `make_v6_fig4.py` | 图 4 |
| `make_fig5a.py` | 图 5(a) |
| `check_numbers.py` | 表 3、表 4、表 6 的 433 格与原始 run 逐格比对 |
| `check_t5.py` | 表 5(a)(b)(c)(d) 的 153 格比对 |
| `check_5d7.py` | 表 5(d) 扩成七方法后的 105 格比对 |
| `audit_v6_cfg.py` | 抽出每个 run 组的 config.json，逐项对比协议 |

核对脚本的约定：准确率容差 6e-4，其余量 6e-3。最后一次全量核对（表 5(d) 扩展前）为
433 格与 153 格均 0 处不符；扩展后的 105 格核对脚本已就位，待数据机恢复后运行。

## 四、运行顺序

```
bash scripts/run_v6_matrix.sh p1a 0 0 &      # 表 3，两个驱动按 seed 奇偶分工
bash scripts/run_v6_matrix.sh p1a 1 1 &
bash scripts/run_v6_chain2.sh &              # 等 p1a 结束后串行跑其余阶段
bash scripts/run_v6_fedavg6.sh &             # 参考点
bash scripts/run_v6_tab5df.sh &              # 表 5(d) 补齐（等误差实验结束）
bash scripts/run_v6_bias6.sh &               # 误差实验（等参考点结束）
```

## 五、已知的等价关系

- 表 3 的 FedRAS（2.0 s）与表 6(a) 的“完整系统”行是同一批 run，数值相同。
- 表 5(d) 的“基准”行复用表 3 的 2.0 s 列（七种方法）。
- 表 6(b) 的“单客户端规则”与表 3 的 FedRAS 名义档共享字节 0.90 MB、秩 16.00、及时率 0.726。
- 三个数据集共用同一批信道实现，故除 Oort 外六种方法的及时率、字节、送达秩逐位相同。
