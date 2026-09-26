# 长文修订日志（main_lora_clean.tex，Computer Networks 审稿响应）

> 隔离提醒：本文件与所有修改仅属于期刊版（`FedLoRA lunwen/`）。
> 会议版（`WCNC_conference/`）不受任何影响；期刊图脚本只写本文件夹，
> 期刊数据只读 `../outputs/`（会议版只读自己的 `data_snapshot/`，见 `../WCNC_conference/ISOLATION.md`）。

## 审稿概况（第一轮）

- 推荐决定：Reject and Resubmit / Major Revision（Computer Networks 标准）。
- 10 条意见。第（1）条"表格为 values to be regenerated 占位符"**与仓库所有版本不符**
  （7 个表格数值齐全；全项目含 backups 内 zip 均无 regenerated 字样）——需登录投稿系统
  核对实际提交的源码包，重投时以当前完整源稿重新打包，并在 response letter 说明。

## 修订记录 1（2026-09-09，零成本迁移 + 内部错误修复）

对应审稿意见编号见括号。

1. **换掉旧数据批次图**：`trainer_dl025.png` 是 Sep 7 07:52 旧批次（y 轴 0.929–0.932，
   与正文 0.914 矛盾），已用 `outputs/real_fedlora_v3/figures/cifar10_dl0.25_accuracy.png`
   （Sep 7 20:11 重跑批次，y 轴 0.910–0.917）覆盖。与 ISOLATION.md 备忘第 1 条对应。
2. **口径统一**（审稿意见未直接指出，但属同类问题）：Q2 段 "0.24 (nominal)" 改为
   "0.28 (nominal, Table I)"，全程锚定表 I 的 30–39 轮窗口；Summary 相应从 46%/83% 改为
   **54%/86%**（(0.28−0.13)/0.28、(0.28−0.04)/0.28）。模拟器 86% 的锚点措辞从
   "from 2 s to 0.25 s" 改为 "from the nominal deadline to 0.25 s"（数据：0.261→0.037）。
3. **预算网格敏感性**（审稿点 4）：Q4 新增网格扫描结果（0.2–0.01 MB，每格 30 实例，
   数据 `outputs/wcnc_extra/grid_sensitivity.csv`，脚本 `scripts/wcnc_grid_sensitivity.py`）：
   宽松预算 0% 偏差；12 MB 束缚时默认 0.05 MB 步长偏差 ≤3.3%，0.025 MB 步长 ≤1.3%
   （1.7× 运行时，27→45 ms）。
4. **模型误差鲁棒性**（审稿点 2/3）：Sensitivity 小节新增 "Robustness to model error" 段：
   P_on 预测值加 σ=0.2 对数正态噪声 → 目标损失最紧格平均 ≤5.1%（最差实例 9.4%），宽松格 <1%；
   G(r) 换线性/对数 → 准入变化 ≤0.5%、目标变化 ≤1.4%
   （数据 `outputs/wcnc_extra/scheduler_robustness.csv`，脚本 `scripts/wcnc_scheduler_robustness.py`）。
   并新增 G(r) 次线性的实证锚点：trainer 中 rank 4→16 的平均更新范数增长 1.57×
   （线性应为 4×；由 `outputs/real_fedlora_v3/all_metrics.csv` 的 update_norm 列算得）。
5. **10-seed 统计**（审稿点 8）：新脚本 `scripts/journal_stats.py`
   （合并 seeds 0–4 主实验与 seeds 5–9 复现，`outputs/wcnc_extra/real_v3_ext/`，
   输出 `outputs/journal_stats/paired_stats.csv`）：
   - 0.25 s 压力 vs r16：字节 −14.5%（bootstrap 95% CI [−18.6%, −12.4%]，Wilcoxon p=0.002，
     秩双列 |r|=1.0）；及时率 10/10 种子对更高（p=0.016）；精度 p=0.40（parity）。
   - nominal vs r16：字节 −0.83%（p=0.002）；精度 p=0.30。
   - vs FedAvg：字节 −54%/−61%（nominal/压力，均 p=0.002）。
   写入位置：摘要（一句）、Q2 段末、Setup（seeds 说明 + R={2,4,8,16}）、Summary 统计注记。
6. **措辞软化**（审稿点 10）：
   - 定位段删 "first work"：改为 "We do not claim any individual ingredient as new…" 的组合式
     条件陈述，并声明以 Related Work 综述为条件。
   - Summary (i) "Rank choice is accuracy-neutral" → "did not measurably change accuracy for
     this frozen backbone and 40-round horizon…at this operating point"。
   - 模拟器两处 "dominate(s)" → "outperform(s)"；敏感性图题注加 "across the tested ranges"。
7. **可复现性补缺**（审稿点 9 的具体缺口）：Setup 写明 rank 候选集 R={2,4,8,16}（此前缺失，
   源自 `scripts/train_torch_fedlora.py` 的 RANKS）。

编译：pdflatex + bibtex 通过，29 页，无错误/未定义引用。

## 修订记录 2（2026-09-09，形式化改造）

对应审稿意见（2）（7）（9）的文字部分；无需新实验。

1. **聚合形式化（审稿点 7）**：Sec. 5.2 重写。
   - 补全公式：掩码平均 $\bar B_{:,j}=\frac{1}{W_j}\sum_{i:r_i\ge j} n_i [B_i]_{:,j}$（eq:maskavg，
     零填充 + 按激活坐标归一化，与 `scripts/train_torch_fedlora.py::avg` 逐行对应）；
     服务器动量 $v^t=\beta v^{t-1}+(1-\beta)\Delta^t,\ S^t=S^{t-1}+v^t,\ \beta=0.3$
     （与 trainer 主循环一致）；权重 $n_i=|\mathcal D_i|$（本地样本数）。
   - 新增**命题 2**（Exactness of frozen-factor aggregation）+ 证明：
     (i) 掩码平均在每个 rank 切片上是激活客户端 updates 的数据加权平均（逐坐标精确）；
     (ii) 全局归一化变体恰为统一权重的 update 平均、但高 rank 坐标被 $W_j/W$ 稀释；
     (iii) 两者当且仅当 delivered ranks 全相等时重合。
   - **修正过宽表述**："averaging B across clients is exactly averaging the induced update"
     → 明确为"per rank slice 精确、坐标依赖权重，非统一权重整体平均"（审稿人指出的问题，
     已同时证明 + 修正）。Related Work、贡献列表、消融段落三处同步引用命题 2。
2. **μ/hazard 准确化（审稿点 2/9）**：Sec. 3.3 重写 eq:hazard 段。
   - μ 的含义与量纲：round-integrated mobility intensity，无量纲，吸收 (1) 式非传输项
     （切换/中断/丢包/训练处理超时）；$e^{-\mu}$ 为全轮无故障存活概率。
   - 传输因子的推导关系：$\exp(-m/\tau)$（$m=8B/\bar R$）与指数服务成功率 $1-e^{-\tau/m}$
     在 $m\ll\tau$ 时一阶等价，且由 $e^{-x}+e^{-1/x}\le 1$ 知其**从不高估**交付概率
     （保守 surrogate）。
   - 估计方法如实描述：实验中 $\mu_i^t\sim\mathrm{Beta}(2,5)$ 每轮抽取、作为可观测链路状态
     （理想化预测器；**与代码一致**——此前"EWMA failure prior"的说法与 trainer/模拟器实现
     均不符，已删）。部署路径指向 robustness（σ=0.2 噪声 ≤5.1% 损失）与 Discussion 的
     isotonic/Platt + Brier 校准。
3. 编译：30 页（+1），无错误/未定义引用；目检第 8 页（hazard 段）与第 12 页（命题 2+证明）。

## 修订记录 3（2026-09-10，新实验：审计 / 校准 / full-DP 对比）

1. **模型与载荷审计（审稿点 6）**：新脚本 `scripts/journal_model_audit.py`
   （输出 `outputs/journal_stats/model_audit.csv`）。论文 Setup 新增 "Model and payload
   audit" 段 + 表 (tab:audit)：8 个 LoRA 层、1283 因子参数/rank 列（FP32 10.25 KB）；
   冻结骨干 1.28M 参数 = 5.13 MB FP32；rank-16 实际序列化 0.082 MB（FP16 0.041）；
   s(r) 的 0.5 MB 协议余量与 ~2× 保守斜率显式说明；FedAvg_full 实际可训练更新 5.21 MB，
   以 rank-64 载荷 1.78 MB 表示**低估其字节账单 2.9×**（对比偏向 FedAvg，结论保守）。
2. **P_on 校准研究（审稿点 2）**：新脚本 `scripts/journal_calibration.py`
   （输出 `outputs/journal_stats/calibration.csv` + 期刊图 `fig_calibration.png`）。
   500 客户端、每格 150 round-instances：(a) 可靠性曲线——oracle 状态下 hazard 预测器
   构造性校准（池化 ECE 0.002–0.004，如实说明）；(b) 误校准敏感性——γ∈{0.5,2} 温度失真
   ECE 至 0.21、σ=0.2 噪声 ECE 至 0.035，对应 DP 指派目标损失最紧格 ≤1.8%、其余 ≤0.5%。
   论文 "Robustness to model error" 段扩写 + 图 (fig:calib)；3.3 节指向同步。
   注意：真实/公开无线 trace 上的校准仍是开放项（已在文中指向 Discussion）。
3. **full-DP vs fair-cohort 真实训练对比（审稿点 5，核心新实验）**：
   - `scripts/train_torch_fedlora.py` 新增 `DP_cohort` 方法：每轮对全部 20 客户端解
     精确多选背包 DP（同 cap K=10 + 8.2 MB 预算 = 10×s(16)，0.05 MB 网格），
     与 fair-cohort 臂共享候选集；冒烟验证后由 `scripts/journal_dp_runs.sh` 发车
     （seeds 0–9，backbone 缓存复用，输出 `outputs/journal_dp/`，只写期刊区）。
   - 分析 `scripts/journal_dp_compare.py`（`outputs/journal_stats/dp_vs_faircohort*.csv`）：
     **精度 0.9152 vs 0.9151（p=0.75，持平）；及时率 0.56 vs 0.42（p=0.002）；
     字节 7.88 vs 8.13（p=0.002，再省 3%）**；delivered rank 14.7 vs 15.9（预算内降秩）。
   - 论文新增小节 "Full scheduling versus fair-cohort in the trainer"（sec:dpvfair）
     + 表 (tab:dpvfair)；Sec.4 部署变体段、贡献列表、Summary (iii) 三处同步引用。
4. 编译：32 页，无错误/未定义引用；目检表 7、图 9、Summary 页。

## 修订记录 4（2026-09-10，基线十种子扩展完成）

后台运行（`scripts/journal_dp_runs.sh`）全部结束（exit 0），10 个基线 run 完成。

1. **基线 10-seed 复现（审稿点 8 收尾）**：SA_rank/Oort_utility/JointAB ×
   {cifar10 nominal, cifar10_dl0.25} × seeds 5–9（`outputs/journal_ext/`），与
   seeds 0–4（`real_fedlora_v3` 的 cifar10_ext* 数据集）拼成十种子。
   5-seed 均值与论文 tab:ext 逐项核对一致；分析脚本
   `scripts/journal_baseline_stats.py`（输出 `outputs/journal_stats/baseline_10seed` 相关 CSV）。
   结论：精度 parity 贯穿（最大差异 0.03 个百分点，全部 p≥0.09）；SA 按其固定低秩路径
   及时率更高、字节更少（p≤0.004）；Oort 省字节持平（p=0.002）；JointAB 精度持平但
   压力下及时率低于 adaptive（−0.019，p=0.016）。
   论文更新：External baselines 小节加十种子复现句；Summary 统计注记同步。
2. 编译：33 页，无错误；PDF 中 "ten-seed replication" 出现 4 处（摘要/Q2/Summary/基线）。

## 修订记录 5（2026-09-14，可复现性 artifact 打包）

1. **打包脚本 `scripts/build_repro_package.sh`** → 仓库根目录
   `fedlora_reproducibility_package.zip`（6.5 MB，432 文件）。
   - **镜像仓库布局**（scripts/ + outputs/ + logs/ + README + requirements.txt），
     包内所有 README 命令原样可跑；已从暂存目录冒烟验证 journal_model_audit /
     journal_stats / journal_dp_compare / journal_baseline_stats 四个脚本，
     输出与论文数字逐项一致。
   - 内容：trainer + 模拟器 + 求解器 + 全部分析/图脚本；主矩阵 1.8 MB
     all_metrics.csv、模拟器 9 格全轮数据、DP 臂与基线的每 run metrics+config、
     分析输出 CSV、每 run 原始日志、现成 figures。
   - **排除**：模型 checkpoint（*.pt）、训练数据（torchvision 下载）、会议版目录——
     隔离原则同样适用于打包。
   - README（`FedLoRA lunwen/ARTIFACT_README.md`）：环境、实验矩阵、
     "论文元素 ↔ 数据文件 ↔ 再现命令" 映射表、重跑成本估算、溯源说明。
2. 论文 Data availability 声明同步（明确列出 code/configs/per-seed outputs/raw logs）。
3. 编译：33 页，无错误。

## 待办（收尾）

- [ ] 20 seeds（如审稿人坚持）与跨数据集（cifar100/mnist）的 seeds 5–9 扩展。
- [ ] 公开无线 trace 校准（审稿点 2 收尾）：或保持 limitation 论证（当前状态）。
- [x] 可复现性 artifact 打包（审稿点 9）：`fedlora_reproducibility_package.zip`（6.5 MB），
  布局镜像仓库、README 命令冒烟验证通过（见修订记录 5）。
- [ ] Response letter（REVISION_LOG.md 已按审稿点组织，可直接作底稿）。
- [ ] 投稿前：为 package 申请永久 DOI（如 Zenodo），替换 Data availability 的占位表述。

## 修订记录 6（2026-09-14，参照三篇已录取 CN 论文的全稿语言与结构修订）

参照论文：FedCD（hierarchical cluster-distillation）、vehicular FedDRL（reward shaping +
adaptive clipping）、ISAC-THz FedDRL（6G IoV）。提炼其共性并落实到本稿：
Introduction 大背景→具体挑战→三点式局限→方案→带标题贡献列表；方法按阶段推进、
每段先目的后机制；实验图句呼应（Fig. X shows ... This indicates ...）；短陈述句为主。

### 顺序执行（用户指定）：方法 → 实验 → 分别单独审 → 合审 → Intro/Related

1. **方法章（§3–§5）**
   - 拆长句：eq.(3) 二因子一段拆为三句；π_i 与 r_max 的 where 从句独立成句；
     §3.1 流水线长句拆二；§5.1 复杂度句去掉分号。
   - 去冗余：§5.2 不再重复引用实验数字（0.901/0.909 移除，只留定性陈述+前向引用，
     数字仅在 §6.5 消融出现一次）；"五步"改为 "as follows"（原文只列四步）；
     \emph 强调全部去除；\emph{Rank-masked averaging.} 改 \textbf 与 §6 段首风格统一。
   - 段落：§5.2 首段按"机制/为何"拆两段；Prop 2 首条内联公式拆成两个短公式避免溢出右边距。
2. **实验章（§6）**
   - 补方法交代缺口：模拟器 "bounded accuracy recursion" 首次给出显式递推式
     （a←a+ηĝ(1−a)−εa，η=0.08，ε=0.0005，ĝ 为按参与上限归一的交付期望有用值），
     并写明模拟器 π_i 以 U(0.15,0.45) 抽样替代 Oort 先验（\textbf{Scoring proxy.} 段）。
   - 超长段全部拆分/压缩：§6.1 audit 段拆二；§6.5 拆三；§6.5 消融拆二；
     §6.6 拆二；§6.7 拆二；§6.8 拆二；§6.9 鲁棒性拆二；§6.11 拆四；
     §2.1 首段拆二；Discussion 七个 run-in 条目改为独立段落（原先在源码中无空行、
     被排成一个大段）；Conclusion 拆二。
   - 标题与小节名：删悬空的 "(Q4)"；"Broader stress matrix:" 冒号改 with；
     "in the prototype" → "in the trainer"。
3. **单独审（方法）**：术语统一（adaptive rank / client–rank assignment / timely ratio /
   full-parameter updates / provisioned width r_max / the trainer），删除
   "prototype/adaptive rule/network-aware rule/knob" 等不一致叫法；每个模型构件
   （hazard、G(r)、DP、Prop 1、Prop 2、两种部署变体）在方法内部自洽。
4. **单独审（实验）**：逐表核对正文数字与表格一致（0.9145/0.9142、7.07/8.20、
   14.5%、54%/86%/86%、0.56/0.42、33–836 ms、27–45 ms、5.1%/9.4%、0.21、1.8% 等）；
   摘要 73% 与模拟器数据复核一致（全轮次口径 74.1%）。
5. **合审（相互支持）**：方法承诺 ↔ 实验验证闭环
   （hazard↔校准研究+σ=0.2 噪声；G(r) 平方根↔1.57× 更新增长实证；DP 最优性↔
   求解器研究+网格扫描；复杂度↔1000 客户端线性扫描；fair-cohort↔§6.6 端到端对比；
   frozen-A↔lr=0.02 消融；延迟丢弃↔所有 timely 指标）；实验展示但方法未交代的
   只有两处（模拟器评分代理、π 抽样），均已在 §6.1 补齐。
6. **Intro/Related/摘要**：贡献列表四条全部重写为"标题句+短句展开+独立章节指引"；
   poster 版本差异段拆句；Positioning 去掉 (i)(ii)(iii) 冒号列举改为三句；
   摘要改两段并压缩至 13/12 编译行，去掉不必要的冒号/破折号/括号。
7. **标点规范**：正文 \emph 清零；em-dash 清零（表格占位符 --- 保留，属表格惯例）；
   题注冒号清零；贡献/正文括号指针改写为完整句子（公式、引用、缩写、CRediT 必要括号保留）。
8. **图表同页**：全部 19 个浮动体改 [H]（float 宏包）并移动到紧邻首引段落；
   终态 14 个与引用同页、5 个相邻（±1 页，均因 §6.2–6.5 图表物理过密）、0 个远离。
   缩图：8 图 0.8\columnwidth、4 表 0.92\columnwidth 以提高同页成功率。
9. **图修复**：`scripts/make_paper_figs.py` DP 运行时面板原来把三个 budget 的点
   混画在一条折线上（x=500 处出现竖直锯齿伪影），改为只取 budget=300 MB 序列，
   dp_study.png 重新生成，曲线单调清晰。
10. **过程中修复的自伤**：浮动体脚本曾把 18 个浮动体写到 \end{document} 之后、
    并把消融段复制 4 份——已全部修复并以"句子级重复审计 vs 备份"确认无残留重复。
11. 编译：34 页，0 错误；judge 三轮验收，末轮仅剩的摘要两段 14/15 行已压至 13/12 行，
    其余全部通过（引用 [2] 作者名、Table 2 数值、ms 单位均为低分辨率误读，源码正确）。

## 修订记录 7（2026-09-14，换题 + 机制命名 + 打包）

1. **题目**：改为 "FedRAS: Deadline-Aware Rank Scheduling for Federated LoRA in Mobile
   Edge Networks"（保留 Deadline-Aware / Federated LoRA / Mobile Edge 检索词）。
2. **机制命名**：方法统一命名为 **FedRAS**（federated rank-adaptive scheduling）。
   - 摘要与 §1 贡献引言处给出命名句；正文、表格（"FedRAS (ours)"、
     "Fair cohort (FedRAS rule)"）、题注（图 1 "Round pipeline of FedRAS"、
     算法 1 "FedRAS scheduling round"）全部替换。
   - 保留两处通用用法：关键词 "Adaptive rank"（检索词）与 "rank adaptation"
     （泛指机制，如 "does rank adaptation preserve accuracy"）。
   - 图例同步：`make_paper_figs.py`、`make_more_figs.py`、`summarize_real_fedlora.py`
     的 LABEL 更新为 FedRAS (ours)，8 张含 Adaptive 字样的图全部重新生成；
     顺带修复 sim_accuracy.png 中 SA_rank 与 r=16 撞色问题（SA-annealing 改橙色）。
   - 数据侧方法键 `Adaptive_rank` 保持不变（CSV 列名/脚本参数未动），在
     ARTIFACT_README.md 加命名对应说明。
3. **命名引发的排版回归修复**：摘要第 1 段因加命名句超至 14 行，等量压缩后恰 13 行；
   §1 贡献引言与 §6.7 各出现一处 1–2 词孤行，已改写消除。
4. **打包**：
   - `fedras_latex_package.zip`（约 3.4 MB）：main_lora_clean.tex / references.bib /
     main_lora_clean.bbl / 9 张图 / 编译好的 main_lora_clean.pdf；
     已在干净目录解压试编译通过（34 页，0 错误）。
   - `fedras_code_package.zip`（约 6.7 MB）：由 build_repro_package.sh 重建
     （scripts/ + fedlora/ + outputs/ + logs/ + README），镜像仓库布局，
     不含 WCNC_conference、.pt、数据集；脚本均为最新版（含 FedRAS 图例）。

## 修订记录 8（2026-09-17，文献检索更新：0917 包 CN_reviewed.tex）

对象：用户提供的 `fedras_latex_package0917.zip`（换题后版本，标题
"FedRAS: Network-Conditioned Rank Allocation for Federated LoRA"）。
**版本隔离**：原包原样保留于 `FedLoRA lunwen/CN_package_0917_original/`；
更新版在 `CN_package_0917_litupdate/`（tex + bib + 重编译 PDF + 图），
zip 于仓库根 `fedras_latex_package_0917_litupdate.zip`。原图与实验数字一律未动。

1. **新增 4 条文献**（全部经 Crossref / ACL Anthology / ACM DL / arXiv 逐条核实）：
   - HetLoRA（Cho et al., EMNLP 2024 main, pp.12903–12913, DOI 10.18653/v1/2024.emnlp-main.717）
     —— 秩自剪枝 + 稀疏度加权聚合；
   - HeLoRA（Fan, Su, Tarkoma, Hui, ACM TOIT 25(2):11, 2025, DOI 10.1145/3723877）
     —— 零填充对齐的 LoRA-异构联邦微调；
   - FedARA（Wu, Hu, Min, Wang, IEEE TC 75(4):1650–1663, 2026, DOI 10.1109/TC.2026.3655161）
     —— 截断 SVD + 动态秩分配（与本方案最近的秩分配工作）；
   - FedSN（Lin et al., IEEE TMC 24(3):1293–1307, 2025, DOI 10.1109/TMC.2024.3481275）
     —— 异构 LEO 卫星网络联邦学习框架。
2. **修正错误 bib 条目**：`yi2024fedlora`（作者 Jingang Yi 等 + arXiv 2310.13283）
   经查为张冠李戴——arXiv:2310.13283 实为 **pFedLoRA**（Liping Yi, Han Yu 等）。
   已替换为期刊版 `yi2026pfedlora`：pFedLoRA: Model-Heterogeneous Personalized
   Federated Learning With Homogeneous Low-Rank Adapter Sharing on Mobile Edge
   Devices, IEEE TMC 25(9):13898–13911, 2026, DOI 10.1109/TMC.2026.3674996
   （作者列表依 Crossref：Yi, Zhang, Yu, Wang, Liu, Hu）。正文 \cite 键同步。
3. **正文修改 4 处**（均为文献定位句，实验与声明口径不变）：
   - §1 Intro：原"rank is fixed, shared by all clients and all rounds"与
     HetLoRA/HeLoRA/FedARA 冲突，改为"selected from data and system constraints
     alone, possibly varying across clients [HetLoRA,HeLoRA] or adapting across
     rounds [FedARA,SA-FedLoRA], but not from the state of a lossy,
     deadline-constrained link"——动机反而更聚焦；
   - §2.1 首段：原四合一列举拆开，新增异构秩对齐一句（自剪枝/stacking/零填充/SVD
     四机制各配其引文，含 FLoRA）；
   - §2.1 SA-FedLoRA 段：加一句区分"respond to training progress or device
     capability, not to the delivery state of a deadline-constrained link"；
   - §2.3：FedSN 补入非地面网络 FL 一句；"These methods generally treat update
     size as fixed" 仍然成立。
4. **核对结论**：SA-FedLoRA（2405.09394）至今仍为 arXiv preprint，维持原引法。
5. **编译**：pdflatex + bibtex 通过，32 页，0 错误、无未定义引用；参考文献
   30 → 34 条（[12]HetLoRA [13]HeLoRA [14]FedARA [17]pFedLoRA [32]FedSN 均正确渲染）；
   bibtex 8 个 empty-pages warning 均为原有会议条目（无页码），非新增。

## 修订记录 9（2026-09-17，审稿意见核验 + Phase 0/1 执行：真实字节 era 启动）

**审稿意见核验结论**（老师 10 条意见，逐条对照代码与数据验证）：
9 条完全属实，第 8 条部分属实。关键实证：
- 评估串用（意见2）：dl0.25 下 FedAvg 141/200 零交付轮准确率与 Adaptive_rank 逐位相同；
- 审计脚本（意见4）：B 因子误取 rank 维，1,283 应为 2,563（A+B）/1,408（B-only），
  论文自身"10.25 KB"与"1,283"同句矛盾；
- mean rank（意见10）：r16@dl0.25 全轮均值 12.48 vs 交付口径 16.0，且 tab:stress（16）
  与 tab:ext（12.5）同格数值互相矛盾；
- 预算离散化（意见6）：论文 round / study greedy 连续 / trainer ceil 三套口径并存。

**Phase 0 决策（老师已确认）**：
1. 主实验改真实序列化字节（frozen-A 下 B-only，1,408 参数/rank = 5,632 B/rank；
   FedAvg_full 全量 5.286 MB ≈ 58.7× rank-16）；deadline 轴重校准：
   nominal 2.0 s / stress 0.1 s / severe 0.05 s / loose 10.0 s；
   stylized s(r) 降级为模拟器 normalized-cost 敏感性格子（--payload-model stylized）。
2. 全链路统一 ceil（保守）预算离散，网格 0.01 MB：trainer DP 臂、solver study
   （DP+贪心+cohort+fixed）、模拟器准入、校准研究四处一致。

**Phase 1 代码修复**（备份 backups/scripts_backup_20260917_pre_realbytes.zip）：
1. `train_torch_fedlora.py` → v4：rank-r 本地训练（LoRA 前向按 eff_r 切片，梯度只达
   前 r 坐标）；每方法独立评估模型（零交付轮评估自己上一轮状态，串用 bug 根除）；
   A 真冻结（冻结法服务器状态只含 .B，聚合与动量剔除 A；JointAB 保留旧联合平均）；
   真实字节记账（bytes_mb 列，运行时从模型实测 bytes_per_rank=5,632、full=5.286 MB）；
   mean_rank 空缺制（零交付轮不留 0，aggregated 列为权重）；新增 DP_same 臂
   （同 cohort DP，分离候选池效应）、--link-profile sagin（地面/空中/卫星三类链路
   + per-client 结果日志）、FedAvg_offline 对照（无交付建模的精度上界）。
2. `journal_model_audit.py`：秩轴修正（B 取 d[0]×prod(d[2:])）；双口径
   （A+B 2,563 / B-only 1,408）；FedAvg_full 5.286 MB（参数口径，不含 BN buffer）。
3. `dp_scheduler_study.py`：ceil 统一；真实字节；格子改为 deadline{2,1,0.1}×
   budget{30,4,1.2}MB；新增 --grid-sweep（0.02–0.005 vs 0.0025 参考步长）。
4. `fedlora/run_experiment.py`（模拟器）：real/stylized 双载荷模式；ceil 准入；
   FedAvg 载荷 5.286 MB（真实模式）。
5. `journal_stats.py`：bootstrap 修为重采样内取均值（mean-of-ratios + ratio-of-means
   双口径 CI）；新增 TOST 等价检验（accuracy，预设界 ±0.5 pp）；mean_rank 交付加权。
6. `journal_calibration.py`：真实字节 + 0.01 网格 + 新格子 (2.0,30)/(0.1,30)/(0.1,1.2)。

**冒烟验证**（3 轮小规模，全部通过）：bytes_mb 与解析值一致；r4/r16 update_norm 分层
（0.094 vs 0.165，rank-r 本地训练生效）；FedAvg 零交付轮评估自身状态；冻结法 checkpoint
只含 .B；DP 预算束缚路径（0.5 MB 预算恰好 50 格用满、DP 降秩+跳过）。

**Phase 2 发车**（后台）：批次 A（CIFAR-10 nominal/0.1/0.05/10.0 ×5 seeds，20 runs）
→ B（ext 基线、DP_cohort、DP_same、Random mask/nomask 对照、FedAvg_offline，35 runs）
链式执行；批次 D（solver study+网格扫描、模拟器 7 格、校准）并行。骨干缓存复用 v3
（0.9138 校验一致）。批次 C（10-seed 扩展、跨数据集、压力矩阵、SAGIN）待 A+B 数据
体检后放行。旧版输出 real_fedlora_v3 原样保留。

## 修订记录 10（2026-09-19，Phase 3 执行：真实字节 era 全文改稿完成）

**实验收官**：155 个正式 run + 10 个 dp@0.1s 补充 + 20 个消融 run 全部落地。
中途事故与修复：mnist 数据路径错误（data → data/mnist，10 run 重跑）；
sagin 分组移动性参数高低反转（地面应为最低失序，已纠正重跑）；
C 批次看门狗清单制/心跳检测上线（pgrep 会被任务包装 shell 永久误匹配，已弃用）；
训练进程 setsid 脱离任务树 + flock 防双写（根因：harness 回收任务树连带收割训练进程）。

**统计（10 seeds，TOST 尾向 bug 已修复）**：
- 名义 2.0s：FedRAS ≡ 固定 r16（字节/及时率逐位一致），精度差 0.11%，TOST p=4.7e-14；
- 压力 0.1s vs r16：字节 −23.4%（CI [−26.6,−20.6]，p=0.002，秩双列 1.0）；
  及时率 +22.2%（[17.9,29.3]，p=0.002）；精度 TOST p=1.5e-7（±0.5pp 界内等价）；
- vs FedAvg_full：字节 −98.3%，及时率 0.726 vs 0.174（名义）；压力下 FULL 及时率恰为 0。

**数据卫生修复**：assemble_v4.py 标签改为"目录名去 seed 后缀"——原正则把 dp/dpsame/
dp_dl0.1/dpsame_dl0.1 全部混入 cifar10 标签（dpvfair 表曾混入压力数据，已修正）。

**Phase 3 产出（FedLoRA lunwen/CN_package_v4/）**：
1. `CN_reviewed.tex` 全文重写（32 页编译通过，0 错误无未定义引用）：
   - 摘要/贡献/结论按真实字节新故事线重写（58.7×、−23%/+22%、TOST 等价、FULL 坍塌）；
   - §3 式(4)改为真实序列化载荷（B-only，D=1,408 参数/rank，rank16=0.090MB）；
   - §4 预算统一 ceil、0.01MB 网格；§5 复杂度式含 B̃=B_max/q；
   - §6 全部重写：deadline 阶梯校准（2.0/0.1/0.05/10.0s）、审计表双口径、
     tab:trainer/stress/extended/ext/dpvfair(10-seed+压力分解)/sim(7 regime 含 stylized
     敏感性)/dp(新格子)/消融/sens+sagin 分组/校准；
   - dpvfair 新分解结论：DP 增益全部来自候选池扩大（名义 +9pp、压力 +6pp 及时率），
     cohort 内 DP 与 argmax 逐位一致（名义）；
   - FedAvg_offline 对照小节（分离"学不动 vs 传不出"）；
   - rank-r 本地切片训练、G(r) 实证锚点 1.86×（r4→r16 更新范数）、调度成本 208μs/决策；
   - 机械修复：\ead 移至 Tao Hu 后、僵尸文献 13 条清除、"trace-driven"→synthetic
     distribution-driven、Acc/MB 删除、fairness 措辞改 uniform sampling、[H]→[t]。
2. `scripts/make_v4_figs.py`：9 张图全部按 v4 数据重新生成（trainer 轨迹、rank 动力学、
   跨数据集、simulator 轨迹+regime 汇总、solver gaps+runtime、敏感性、calibration）。
3. `scripts/print_v4_tables.py` + `outputs/journal_stats/v4_tables.txt`：论文全部数字的唯一
   权威来源；`paired_stats_v4.csv` 为统计权威来源。

**待回填**：tab:ablation 中 nocap/nomom/lr02 对 4 行（占位值，20 个 run 收尾后以真实数据
回填并重编译——占位值绝不出稿）。

## 修订记录 11（2026-09-19，目录封存重构：三文件夹结构）

仓库根 `FedLoRA_remote_package/` 现仅含三个文件夹 + 隐藏技术目录（.venv/.pytest_cache/
.ipynb_checkpoints/.gitignore 留在根，.venv 移动会破坏环境）：

1. `1_实验代码及结果/`：scripts/ fedlora/ src/ tests/ configs/ data/ outputs/ logs/
   generated_figures/ artifact_staging/ deadline-aware-federated-lora(旧代码快照)/
   README* pyproject requirements*；`归档包/` 内存 fedras_code_package.zip、
   fedlora_reproducibility_package.zip、data.zip。
2. `2_latex及附属文件历史版本/`：`FedLoRA lunwen/`（含本日志、历次 tex、
   CN_package_0917_original、CN_package_0917_litupdate、旧图与 cover letter）、
   WCNC_conference/、backups/、WCNC.zip、FedLoRA_WCNC.zip、fedras_latex_package.zip、
   fedras_latex_package_0917_litupdate.zip、FedLoRA_remote_package.zip。
3. `3_ComputerNetworks期刊latex版本/CN_package_v4/`：当前投稿版（tex+bbl+pdf+
   bib34条+9张v4图）。

**路径注意事项**：
- 基于 `__file__` 解析 ROOT 的脚本（journal_stats / print_v4_tables / make_v4_figs /
  assemble_v4）在新位置自洽，已验证；运行时 CWD 需在 `1_实验代码及结果/` 内或用绝对路径。
- `journal_calibration.py` 的 PAPER 常量仍指向旧路径 `ROOT/'FedLoRA lunwen'`（一次性
  已完成任务，若重跑需改指本目录或新包）。
- run_v4_matrix*.sh 的 ROOT_DIR=scripts/ 的父目录 → 1_实验代码及结果/，自洽。
- Qwen 邻居任务（python -m fedlora.run）已于重构前结束，不受影响；若重启需注意其
  fedlora 模块解析路径。
- 迁移为同文件系统 mv（瞬时完成），迁移前后项数核对一致（28 项全部入位）。

## 修订记录 12（2026-09-19，nature-response 流程：叙事补强 + 全文英文润色）

按叙事逻辑分析的三点补强计划执行（模式 revise，decision type N/A=作者主导修订，
备份于 2_latex.../CN_package_v4_backup/CN_reviewed_pre_polish.tex）：
1. 引言第三层补因果链（秩→字节→交付概率的 omitted coupling，两句话替换式扩写）；
2. 新增命题 3（Cohort-local optimality of the per-client rule）：全 rmax 轮廓预算可行时
   逐客户端 argmax 即受限问题最优解，DP 与之重合——与 dpvfair 分解实验互为理论/验证；
   命题 1 补标签 prop:dp；dpvfair 与 Summary 两处引用命题 3；
3. Positioning 补理论层级声明：理论贡献是机制正确性（指派/聚合精确性）而非学习性能保证。
全文润色 24 处：ceil→ceiling、mobility-related 连字符、re-fit/expose it 等语法、
836ms→421ms 与"64% in binding regimes"→"28–55% binding / 64% val-byte"（摘要+结论口径修正）、
"miss every deadline"→"miss every stress deadline"、外部基线段 JointAB 稳定性表述与消融
结论对齐（不再声称 stability cost）、SAGIN 段删除无数据支撑的"卫星类降秩"表述（改为
"类别差距来自链路定律而非秩分配"）、Generality 的 20/1000 客户端归属精确化、
headroom 措辞改为 transmission time consumes 3%。
交付：干净版 34 页 0 错误；latexdiff 红标版 CN_reviewed_marked.pdf 29 页 0 错误。

## 修订记录 13（2026-09-19，nature-writing 结构审查计划全项执行）

结构审查（algorithmic 论证链 + 读者五问 + 五维拒稿风险）结论：论证成立、无硬伤，
8 项可改进项全项执行：
P1-1 §6.9 更名 "Sensitivity, link profile, and predictor calibration"（标题与内容对齐）；
P1-2 SAGIN 段删除无数据支撑的 tight-deadline 预测句；
P2-1 §7 新增 "When FedRAS does not help" 段（收拢三处自报边界：名义恒等/α=0.1 漂移/
卫星类低交付，附使用判据，替换原 heterogeneity 单句段）；
P2-2 §6.8 末新增 "Deployment recipe" 三条规划规则（deadline 阶梯按真实载荷校准、
预算=10×r_max 网格成本、Prop 3 判据决定 argmax vs DP）；
P2-3 §3.4 载荷公式加 Table 1 交叉引用；
P3-1 引言三问加答案节映射（3/5/6 章 respectively）；
P3-2 §6.1 加 "FedAvg baseline is referred to as full-model updates" 术语约定；
P3-3 Summary 压缩：删与 §6.8/结论重复的运行时句、第三段统计注记并入第二段。
编译：干净版 34 页 0 错误；红标版 30 页 0 错误。验证：新段落/映射在 PDF 中命中。

## 修订记录 14（2026-09-19，nature-figure 流程：投稿级图重绘）

后端门控：用户指定 Python（nature_figure_backend.py set python 已持久化）。
导出契约（Computer Networks/Elsevier）：单栏 90mm/双栏 190mm 终尺寸、Arial/Helvetica
7pt 基号 5pt 下限、pdf.fonttype 42 可编辑文字、矢量 PDF 主 + 600dpi PNG。
9 张数据图全部按五点契约重绘（make_v4_figs_submission.py）：
统一方法色族（FedRAS 信号红）、多面板 (a)(b)(c) 标题位标签、误差带 n=5、
对齐门 require_matplotlib_panel_alignment（1.5pt strict）前置。
QA 三件套逐 PDF：validate_figure（17 pass/READY）、audit_pdf_text（最小字形 5.8pt PASS）、
audit_figure_collisions——三轮修复（面板标签改 set_title(loc=left) 消除页顶裁切、
DP 运行时标注改白底/最终删除、图例移出柱区），终态 8 PASS + 1 REVIEW REQUIRED
（cross_dataset x 轴刻度标签贴近柱底基线的常规几何，3pt pad 后仍标记，按规程目检
记录为可接受）。
tex 切换：9 处 \includegraphics 去 .png 后缀改用矢量 PDF（graphicx 自动选 .pdf），
fig:sim/fig:dp 题注 Left/Right 改 (a)/(b)，删除包内旧 PNG。
编译：34 页 0 错误；包体积 926KB→539KB（矢量图更小）。
工具链：journal_calibration.py 增 calibration_curves.csv 导出（供面板 a 重绘）、
PAPER 常量改指 3_.../CN_package_v4。

## 修订记录 15（2026-09-19，独立审计修复：种子口径 + 校准表述 + 小项）

依据作者独立逐表格核对报告执行修复（审计确认无数字造假，全部可溯源）：
① **校准表述反向修正**（实质性问题）：数据中最大目标损失 0.95% 在最宽松格
  (2.0s/30MB, noise 变体)，最紧格仅 0.46%——§6.9 两处与 Fig.8 题注的
  "confined to the tightest cell" 全部改写，并按误差结构如实描述
  （γ 失真成本集中于压力 deadline ≤0.45%，噪声成本集中于名义松弛格 0.95%）；
② **种子口径统一**：setup 明示 CIFAR-10 名义/0.1s、ext 基线与调度臂为 ten seeds；
  表 2/表 5 题注改 ten seeds；表 3 题注与 §6.3 改"0.1s 十种子 / 0.05s 五种子"；
  表 6 下半 FedRAS 行回填真 5-seed 值（0.9144±.0018/0.51/0.70/13.0，消除混用）；
  表 9 Full system 行回填 5-seed 值（0.9144±.0013）——③表 5 与表 9 的 JointAB
  表面冲突随题注标注自动消除；
④ **小项**：same-cohort std .0018→.0019；sim 表 FedAvg timely .35→.34；
  "thirteen rank-16"→"twelve on the ceiling grid"（网格口径自洽）；
  1.86×→about 1.9×；208μs→about 0.1 ms/decision；CIFAR-100 backbone 0.692→0.693；
  全模型基线 0.4× 学习率披露；proxy 公式补 ω~N(0,5e-4) 噪声项；
  FedAvg 容量信用假设在 Simulator 段显式披露；Eq.(5) "including headers"改
  "B_meta set to zero in the metering"；§6.2 恒等句加浮点非确定性限定；
  5 处图题种子标注改为实际口径（10-seed/混合）。
未改：图数据本身（以诚实标注替代重过滤）；stylized cell 数值（精确吻合）。
编译：干净版 34 页 0 错误；红标版 30 页 0 错误。

## 修订记录 16（2026-09-21，作者复验遗留三处收尾）

作者逐项复验修订记录 15 后判定"有条件通过"，遗留三处文字问题全部修复：
1. §6.3 正文漏改：'the five-seed stress trajectories'→'the ten-seed stress
   trajectories'（Fig.5/fig:trainer_dl 图与图注均为 10 种子聚合，正文唯一漏网处）；
2. §6.1 总说明加限定：scheduling-arm comparison 后补 'at the nominal deadline'
   （表 6 上半名义臂为十种子、下半 0.05s 臂为五种子，原句口径过宽）；
3. §5.1 DP 内存声明补全：'with rolling value arrays; the back-pointer table
   retained for reconstruction adds O(|D|K B̃_max)'（与 dp_assign 实现
   back 列表逐 client 存储 (cap+1)×(Bu+1) 动作表一致）。
同步：CN_reviewed_marked.tex 三处按 latexdiff 规范标记（新增蓝色波浪、
five-→ten- 红删蓝加）。编译：干净版 34 页 0 错误；红标版 30 页 0 错误；
PDF 渲染逐处目检（p10/p12/p16）确认标记正确。复验遗留问题清零。

## 修订记录 17（2026-09-21，Fig.1 系统架构图重绘）

按 nature-figure 规程（Python 后端）重绘 Fig.1：原 TikZ 七框流水线升级为
投稿级系统架构示意（schematic-led composite，190mm 双栏幅面）：
- 服务器带：Link states → Per–client–rank prediction（eq.3 hazard、eq.5
  capacity 公式）→ 红色英雄框 Deadline-aware scheduler（exact DP 目标式与
  预算约束）→ Rank-masked aggregation（B̄ 列平均、Ā=A frozen）；
- 客户端带：ground/aerial/satellite 三类卡片（frozen backbone、shared A
  never sent、B 按列分片着色 16/8/4、assigned r_i 胶囊）；
- 管线箭头：蓝色虚线 broadcast（w + ranks）、红色 uplink（B_i(r_i) only，
  经 on-time gate τ，timely/late-lost 分流）、灰色虚线 outcomes 反馈闭环；
- 底注真实字节：1,408 params/rank column（5.63 KB）、r16 0.090 MB vs
  5.29 MB（58.7×）。图内数字全部与 §3/§6 一致。
质量闸门：validate_figure READY；字形审计 ≥5pt PASS（mathtext 母字号
7.6pt、Unicode 重音替代 \hat/\tilde 消二级脚本缩放、exp 合并指数）；
碰撞审计 0 fail 0 warn（实测 bbox 驱动布局，脚本内置 fit-check 断言）；
独立视觉验收两轮（第一轮抓到卫星支管未接总线的设计缺陷，修复后 PASS）。
tex 集成：clean 版 TikZ 块替换为 \includegraphics（0.98\columnwidth）+
新题注（引 eq.3/5、Prop.1/3）；marked 版按 DIFDELCMD 惯例标记替换。
顺带修复：红标版 CN_reviewed_marked.bbl 长期缺失导致参考文献表整表缺
失、正文 51 处 [?]——复制 bbl 重编译后 0 undefined。
编译：干净版 34 页 0 错误；红标版 35 页 0 错误 0 未解析引用。

## 修订记录 18（2026-09-26，Fig.1 重排版：严格栅格 + 矢量小插画）

按作者反馈（"不太板正、希望横平竖直、加小插画"）重排版 Fig.1：
- 严格栅格：服务器四盒等宽等高同行共边、等间距，箭头统一走中线；
  三张客户端卡片等大同列；全部连线改为直角曼哈顿路由，锚点对齐
  （蓝下降线在卡片中心、红上升线在 center+8、蓝主干对准红框中心）；
- 矢量小插画（matplotlib 原语，非位图）：服务器机架（带标签）、
  手机/四旋翼/卫星（三类客户端）、冻结锁形（backbone 与 shared A 芯片）、
  τ 时钟（on-time gate）；
- 轴改全出血 add_axes，导出即真实毫米口径，字号不受子图边距缩放。
质量闸门：fit-check 36 项断言通过；字形 ≥5pt PASS；碰撞 0 fail 0 warn；
独立视觉验收两轮（第一轮抓到红上行箭头未锚定 gate 盒、绿箭头无杆两处
缺陷，修复后逐像素确认 PASS）。
两版重编译：干净版 34 页 0 错误；红标版 35 页 0 错误 0 未解析引用。

## 修订记录 18（2026-09-26，结构重组：8 节 → 7 节 + RW 扩写 + R1–R5 去重）

按已确认的结构计划执行（干净版 CN_reviewed.tex 单轮连续搬移，label 全部保留原名）：
① 骨架重建：原 §3.1+3.2 合并为 "Mobile-edge federated learning architecture and
  round protocol"；原 §4 降为 3.4 "Deadline-constrained client--rank scheduling
  formulation"；原 §5 改名 "Solution method and heterogeneous-rank aggregation"
  并拆为 4.1 离散化 / 4.2 DP+Algorithm 1 / 4.3 正确性与复杂度（Prop 1–2）/ 
  4.4 frozen-factor rank-masked aggregation（Prop 3）；原 §6 改名 "Experimental
  evaluation" 并分两层：5.2 真实 trainer（6 个三级小节，消融归队 5.2.6）、
  5.3 大规模仿真（500 客户端行为 / solver / delivery-predictor sensitivity），
  5.4 敏感性与鲁棒性（r_max/cohort/α/SAGIN），取消 5.5 Summary（并入 5.4 收尾
  过渡段）；Discussion 拆 6.1 解释与部署（deployment recipe 迁入）/ 6.2 范围与
  局限；Conclusion 缩至 1 段主体+范围+future work。
② R1–R5 去重：轮次管线只在 3.1 正文与 Algorithm 1 各保留一次，Fig.1 题注压缩；
  hazard 限定语三处分工（3.2 模型性质 / 5.1 oracle 披露 / 6.2 外推边界）；
  0.090/5.29/58.7× 收敛至摘要、Table 1、5.1 计量、5.2 各一次；调度变体定义移入
  5.1（离散化句留在 4.1）；objective 免责三处分工（3.4/5.1/6.2），摘要与结论
  不再重复；结论仅保留一组代表性数字（22%/23%）。
③ RW 扩写（1→2 页）：2.1 改名并补 FedPETuning（Findings ACL 2023, DOI 核实）与
  FedDPA（Yang et al., NeurIPS 2024, arXiv 2403.19211）；2.3 按问题维度重组并补
  FedCS（ICC 2019）、TiFL（HPDC 2020）、LESSON（Future Internet 2023）；2.4 改写
  为研究缺口定位，新增对比表 tab:rwcompare（9 行 × 7 列，单元格逐篇对照原文核实，
  HeLoRA/SA-FedLoRA 未核实格标 n/r）。5 篇新引用全部带 DOI 入 references.bib。
④ 悬挂引用修复：行 47 "Sections…develop the model"→Section 3 统称；Table 1 题注
  stylized 协议措辞随 sec:simsim 新位置改写；sec:summary 引用随 5.5 取消改为段内
  指代；红标版对被删旧句的 sec:summary 残留以幽灵 label 兜底。
⑤ 编译：干净版 35 页 0 错误 0 未解析引用；红标版（latexdiff 自基线 3b8f207 重新
  生成，41 页）0 错误 0 未解析引用；浮动体逐页检查无堆积（每页 ≤2 个）。
⑥ 比例实测（正文 28 页）：Intro 2 / RW 2 / 模型+建模 3 / 方法 3 / 实验 15 /
  Discussion 2 / Conclusion 1；实验占比 54%（期刊常见带 40–60%）。

## 修订记录 19（2026-09-26，Fig.1 四盒加机制小漫画）

按作者反馈（四框信息量少、不好看）为服务器四个流程盒各加一个矢量小图：
- Link states：基站塔 + 信号弧 + "R̄_i over rounds" 波动曲线（链路状态随
  时间起伏）；
- Per–client–rank prediction：两条迷你曲线——红色 P̂ ↓ vs rank（秩越大
  越难准时）+ 绿色 G ↑ saturating（容量 √ 饱和），一眼看出调度器仲裁的
  权衡；
- Deadline-aware scheduler（英雄框）：准入头像 ≤ K + 字节预算条按
  [16][8][4][4] 装包（虚线格=未用），下方 B̃_max 括号；
- Rank-masked aggregation：16/8/4 秩柱 → 平均后的 B̄ 方块 + Ā 冻结条
  （浅蓝 + 锁）。
盒高 78..118（+10mm），画布 190×122。修复过程中发现并解决：D 盒冻结条
hatch='///' 的未裁剪路径 bbox 罩住全盒文本（碰撞审计 5 fail 根因，改纯色
填充）；绿箭头/灰反馈线在重排后悬空（视觉验收抓到，改为 gate 顶双立管：
绿直上聚合盒底、灰 71.5 转左入带底，两端全锚定）；late/lost 加 ✕ 标记。
质量闸门：fit-check 50 项通过；字形 ≥5pt PASS；碰撞 0 fail 0 warn；
独立视觉验收第三轮逐像素确认 PASS（"Recommend acceptance"）。
两版重编译：干净版 35 页 0 错误（图变高整体后移 1 页）；红标版 41 页
0 错误 0 未解析引用。

## 修订记录 19（2026-09-26，去数字化：叙事/方法节只留逻辑，数字归位实验节）

按"超参数与复现细节只在实验节出现、前文只讲叙事与算法流程"的原则执行：
① 摘要：载荷字节对（0.090/5.29）、p 值、28–55%/64%/23–421ms 全部删除；仅保留
  22%/23% 一组代表性结果与 ≤1% 目标损失上界，其余改定性表述。
② 引言：贡献第 4 条去 22%/23%/0.1s/"ten seeds"，改为定性收益表述。
③ §3.3：删除 D=1,408、b=32、B(16)=0.090 示例与 1.9×/4× 经验数值——载荷公式
  只留符号，实测值归 Table 2 与 5.1；√ 形状的经验支持改为指向实验节。
④ §4.1：网格宽度 q 符号化（原 q=0.01MB），数值移入 5.1 solver study 块；
  §4.4：动量系数 β 符号化（数值已在 5.1 声明 momentum 0.3）。
⑤ §5 内代码风味数字精简（超参数本身保留）：payload audit 段删 2,563/1,155/
  5.63KB（保留 1.28M、D=1,408、0.090/5.29）；5.2.2 统计保留 p=0.002/CI 约
  [-27,-21]%/±0.5pp，删 TOST p 值与 CI 末位小数；5.2.5 删 0.82/0.73/0.55/0.51/
  0.87/0.70（Table 7 承载）；5.3.2 删 41–48%/28–36%/48–55%/23–421ms/851ms/
  grid sweep 步长序列（保留 ≤64% 与 sub-second）；5.3.3 删全部 ECE 小数
  （0.003–0.004/0.22/0.078/0.45%/0.95%），保留 <1% 上界与误差结构叙述；
  5.4 删 0.45→1.80MB/0.75→0.70/0.1ms 序列。
⑥ §6.1：0.32pp、satellite 0.49 改定性。
⑦ 编译：干净版 35 页、红标版（latexdiff 自基线重生成）41 页，均 0 错误
  0 未解析引用。58.7× 全文仅存 Table 2 与 5.2.1 两处。

## 修订记录 20（2026-09-26，8 节定稿结构：System+Threat Model / Design / Theory 拆分）

按确认的期刊经典弧线重排（1 Intro / 2 RW / 3 System+Threat / 4 Design / 5 Theory /
6 Experiments / 7 Discussion / 8 Conclusion）：
① §3 改名 "System model and threat model"，新增 3.4 "Adversary and trust model"
  （honest-but-curious 客户端、受信聚合器、uplink 窃听者；frozen-A + rank 切片
  = 更小的梯度反演暴露面；显式不主张 DP/拜占庭，正交组合），问题建模顺延为 3.5；
② §4 改名 "FedRAS design"，新增 4.1 "Design overview"（三组件 + 一句无训练侧
  逻辑声明，未复述 §3.1 轮次协议）；4.2 离散化 / 4.3 DP+Algorithm 1 /
  4.4 聚合机制；
③ 新 §5 "Theoretical analysis"：5.1 Prop.1（DP 精确性）/ 5.2 Prop.2（argmax
  规则最优条件）/ 5.3 Prop.3（frozen-factor 聚合精确性）/ 5.4 复杂度 +
  合并的 "Scope of the analysis" 段（三处命题后散落的 scope 句全部收拢于此）；
④ 去重：§4.4 原收尾段（ablation 重复指针 + 稳定性声明）删除，指针只在 §4.4
  留一处；Discussion Myopia 段与 5.4 scope 去重（改指 Section 5）；Security 段
  收缩并指向 3.4；实验节 grid exactness 引用 sec:solver→sec:theory；
⑤ Intro 路线图更新为 8 节版本。
⑥ 编译：干净版 36 页、红标版（latexdiff 自基线重生成）43 页，均 0 错误
  0 未解析引用。新增文字（威胁模型、overview）不含任何实验数字，延续
  修订记录 19 的"数字只在实验节"原则。章节起始页：§3 p.6 / §4 p.9 / §5 p.11 /
  §6 p.13–26 / §7 p.27 / §8 p.29。

## 修订记录 21（2026-09-26，实验章段落版式统一：加粗首句 + 一段一事 + 段间零重复）

① 6.1 Experimental setup 按维度重排为 14 个主题段（每段 \textbf{X.} 开头）：
  Environment / Datasets and partitions / Backbone and adapters / Federated
  training hyperparameters / Link models / Deadline calibration / Model and
  payload audit / Simulator / Scoring proxy / Baselines / Scheduling variants /
  Solver study / Metrics / Statistical protocol。原 "Real trainer" 段拆为
  Environment+Backbone；原 "Federated protocol and links" 段拆为
  Datasets/Hyperparameters/Link models；新增 Baselines（内/外部基线定义自
  6.2.4 前移）、Metrics（自表题与收尾段聚合）、Statistical protocol（种子/
  CRN/Wilcoxon/bootstrap/预注册 TOST 边界，自协议段与 6.2.2 前移）。
② 后续小节每段加粗首句：6.2.2 Delivered-rank adaptation + Mechanism and
  replication；6.2.4 Baseline behavior（定义段前移后与行为段合并，删除重复旧段）；
  6.2.5 Arms under evaluation + Where exactness pays；6.2.6 Component
  contributions + Stability at elevated learning rate；6.3.1 Regime comparison +
  Trajectories；6.3.2 Heuristic gaps and runtime + Grid sweep；6.4 Width, cohort,
  and heterogeneity + SAGIN link profile + Cross-tier summary。单段小节
  （6.2.1/6.2.3/6.3.3）由小节标题承担首句职责，不加粗以免与标题重复。
③ 段间去重：种子/CRN 口径只在 Statistical protocol；±0.5pp 边界只在协议段
  定义、结果段引"pre-registered margin"；外部基线定义只在 Baselines；
  "descriptive unless tested" 只在协议段；6.2.2 段首不再复述种子数。
④ 编译：干净版 36 页、红标版 44 页，均 0 错误 0 未解析引用。

## 修订记录 22（2026-09-26，方法论算法表体系 AC/GDP/MA + 贡献三段式 + 严格图文去冗余）

① §4 算法体系（小节名=算法名，正文简称引用）：
  4.1 Overview + Algorithm 1 "FedRAS Round"（主算法，端到端七步，按名调用
  三个子算法）；4.2 Action Construction (AC) + Algorithm 2（payload/hazard/
  capacity/utility/整数成本打分）；4.3 GridDP Assignment (GDP) + Algorithm 3
  （DP 递推+回溯，原主算法内联 DP 拆出）；4.4 Masked Aggregation (MA) +
  Algorithm 4（零填充/逐列加权平均/动量）。新增 \label{sec:ac}、sec:gdp。
② 贡献改三段式 enumerate：Framework（"To our knowledge, first framework
  that treats the LoRA rank as a deadline-constrained, network-conditioned
  schedulable resource"，与 2.4 诚实定位调和）/ Theoretical analysis
  （Prop.1–3 + 复杂度）/ Experiments（22%/23% + 准入兑现 + 鲁棒性）。
③ 正文组件引用改简称：§3.1 轮次协议、Fig.1 题注、6.2.5（sec:gdp）、
  6.3.3（alg:fedras）；旧 alg:scheduler 标签引用清零。
④ Q4 严格图文去冗余（表内数字正文不复述）：6.1 audit 段删 0.090/5.29/
  D=1,408；6.2.1 删 58.7×/0.17/0.73/5.29（改"orders of magnitude"）；
  6.2.3 删 0.9107/0.9139/0.9022（留派生值 0.32pp）；6.2.4 删 0.29MB/rank 5；
  6.3.1 删 0.914/0.781/0.763/0.797/0.413/0.404/0.374（留派生 21%）；6.3.1
  与 setup Metrics 重复句删除。
⑤ Fig.1 框架图与算法对应（修改并行会话的 make_fedras_architecture.py 并
  重生成）：Box B → "Action Construction (AC)"、Box C → "GridDP Assignment
  (GDP)"、Box D → "Masked Aggregation (MA)"，Box A "Link states" 对应主算法
  第 1 步；conda python 渲染（venv 的 matplotlib 触发 gate.late 边界假阳性），
  fit-check 50 项全过。
⑥ 编译：干净版 37 页、红标版（latexdiff 自基线）45 页，均 0 错误
  0 未解析引用。

## 修订记录 23（2026-09-26，实验章"一实验一主表+并联图"重组）

① E1（6.2 真实 trainer）：原 6 张小表（T3 阶梯/T4 压力/T5 扩展/T6 外部基线/
  T7 调度臂/T8 消融）合并为一张大主表 Table 3（五个块 a–e，堆叠 tabular），
  配 2×2 四联图 Fig.2（名义精度/压力精度/rank 动态/跨数据集）；fig_cross_dataset
  保留为面板 (d)（其数据与主表 (b) 块部分重叠，题注注明"numbers in Table 3(b)"）。
② E2（6.3 规模实验）：模拟器表+求解器表合并为 Table 4（a/b 两块）。
③ 6.4 解散：SAGIN 段并入 6.2.3，敏感性宽图保留为 6.2.3 全宽图（宽幅三联图
  缩半不可读，visual-judge 判定），跨层总结移至 6.3 末。全文表格 10→4 张，
  图 10→7 幅。
④ 引用全面改写：tab:trainer/stress/extended/ext/dpvfair/ablation/sim/dp →
  tab:trainermain(a)–(e) / tab:scalemain(a)(b)；fig 引用改面板号。
⑤ 视觉验收四轮（visual-judge）：修复主表溢出页底（页码叠印）、(b) 块被
  resizebox 放大 1.4×、E1 网格敏感性面板不可读（改回全宽）、E2 四联宽图
  缩半不可读（恢复 4 张全宽图，每张内部 2 联并列）等问题；主表 scriptsize
  + tabcolsep 3pt + arraystretch 0.9 + (e) 块左右并排后 PASS。
⑥ 编译：干净版 35 页 0 错误 0 未解析引用；红标版 45+ 页同口径。
