# MP-RBPS / GCDP 可复现实验项目

本目录实现论文中的移动性感知风险预算保护选择与精确组基数动态规划（GCDP）。项目包含纯 NumPy 小型验证、CIFAR-10/PyTorch 真实数据实验、基线、可复现随机轨迹、CSV 指标和绘图。运行不依赖论文 LaTeX 文件。

## 已完成的本地验证

- GCDP 与穷举搜索在 40 个小随机实例上的最大目标值差为 `0.000e+00`。
- NumPy 端到端 demo 已完成，覆盖链路生成、动作估计、调度、本地训练、随机传输、聚合、CSV 和图片。
- demo 中 GCDP 平均建模目标值为 `1.409949`，高于 greedy repair 的 `1.374865` 和 two-stage 的 `1.336949`。demo 仅验证代码链路，不作为论文正式实验结论。

现有 demo 结果位于 `outputs/demo/`。

## 投稿关键证据实验（新增）

`scripts/run_submission_evidence.py` 将每个场景下的所有方法运行在相同的五个种子、客户端划分和随机链路轨迹上，并统一导出可用于论文的逐种子结果。它覆盖：名义场景、高移动性、紧 deadline、低/高 exposure budget、SecAgg 最小组规模与 exposure 权重敏感性；方法包括 GCDP、greedy repair、two-stage、reactive、fixed SecAgg、compression-only、fixed AEAD 和 FedAvg。

先用 NumPy 进行全矩阵筛查，再使用 CIFAR-10 运行最终矩阵：

```bash
python scripts/run_submission_evidence.py --phase screen
python scripts/report_submission_evidence.py --root outputs/submission_evidence/screen

# GPU 上运行；默认采用 full_cifar10.yaml 的 150 rounds。
python scripts/run_submission_evidence.py --phase cifar
python scripts/report_submission_evidence.py --root outputs/submission_evidence/cifar
```

输出 `confidence_intervals.csv`、`paired_gcdp_comparisons.csv` 和 `submission_evidence_report.md`。其中配对差值定义为 `GCDP - baseline`；正值是否有利取决于指标（例如 objective 和 success ratio 越大越好，而 bytes、energy、Brier 和 ECE 越小越好）。因此不要仅根据正负号解释，须结合指标方向和 95% 区间。

该矩阵用于支持“联合决策、鲁棒性、敏感性与 solver trade-off”主张，**不能**替代威胁特定的 reconstruction 或 membership-inference 攻击校准；在没有该校准和 DP accountant 时，不应声称正式隐私保证。

## 远程机器：按顺序运行

要求 Python 3.10+；完整实验建议一张 CUDA GPU、32 GB 内存和至少 20 GB 空闲磁盘。

```bash
cd mprbps_experiments
bash scripts/setup_remote.sh
source .venv/bin/activate
python -m mprbps.cli run --config configs/pilot_cifar10.yaml
```

`pilot_cifar10.yaml` 只运行 2 轮、2 种方法。它会首次下载 CIFAR-10，并验证远程 PyTorch、数据、GPU/CPU、训练和绘图全链路。确认生成 `outputs/pilot_cifar10/summary.csv` 后，才启动正式实验：

脚本会创建项目专用 `.venv`，不会向 `/opt/conda` 安装或升级任何软件。如果主机已有与 NVIDIA 驱动匹配的 CUDA PyTorch，`.venv` 只读继承该 PyTorch；项目自身仍与系统环境隔离，也不会下载第二份大型 torch wheel。

```bash
bash scripts/run_full.sh
```

后台运行可用：

```bash
mkdir -p logs
nohup bash scripts/run_full.sh > logs/full.log 2>&1 &
tail -f logs/full.log
```

SLURM 集群可用：

```bash
mkdir -p logs
sbatch scripts/slurm_full.sbatch
squeue -u "$USER"
```

如集群的分区、时限或 GPU 请求语法不同，只需调整 `scripts/slurm_full.sbatch` 的 `#SBATCH` 行，不需修改 Python。

## 无 GPU 或离线安装

`device: auto` 会自动回退 CPU，但正式 CIFAR-10 实验会很慢。若远程机器要求指定 CUDA 版 PyTorch，请先按该机器 CUDA 版本安装官方 torch/torchvision，再运行：

```bash
python -m pip install -e .
python scripts/preflight.py --config configs/pilot_cifar10.yaml
```

CIFAR-10 默认自动下载到 `data/`。完全离线的计算节点需提前把 torchvision 的 CIFAR-10 数据目录上传至该位置。

## 配置和正式实验规模

- `configs/demo.yaml`：无 torch 的快速合成 demo。
- `configs/pilot_cifar10.yaml`：远程真实后端冒烟测试。
- `configs/full_cifar10.yaml`：500 客户端、150 轮、7 方法、5 随机种子。

正式配置计算量很大。每个方法和种子从相同初始化、相同客户端划分、相同链路过程和共享随机种子开始；模型因各策略选择和聚合结果不同而自然分叉。所有结果写入配置中的 `output_dir`，不会覆盖其他配置目录。

## 输出文件

- `round_metrics.csv`：每轮准确率、选择/成功数、目标值、SecAgg 组规模、求解时间和剩余预算。
- `action_metrics.csv`：每次选中动作的策略、可靠性、实际成功、时延、通信/能耗/失真和暴露成本。
- `summary_by_seed.csv`：每个方法、每个种子的最终指标。
- `summary.csv`：跨种子的均值与标准差。
- `accuracy.png`、`delivery_runtime.png`、`policy_distribution.png`：论文结果图基础版本。

## 验证与测试

```bash
source .venv/bin/activate
python scripts/verify_gcdp.py
python -m pytest -q
python -m compileall -q src scripts
```

若长实验意外中断，当前版本为保证状态一致性，不会从半轮 CSV 自动续跑；保留已有输出后重新运行对应配置即可。正式运行前务必先完成 pilot。不要把 demo 数值写入论文，论文数值应以完整多种子实验的 `summary.csv` 为准。
