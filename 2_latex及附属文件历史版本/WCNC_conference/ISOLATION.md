# 版本隔离说明（会议版 ↔ 期刊版）

本文档说明两个论文版本之间的隔离机制，以及今后分别修改时的操作规则。

> **修订记录**：2026-09-09 合并作者 0909 修订版（`backups/user_revision_0909_original.zip`）：
> 摘要/贡献/结论改为更克制的表述、章节标题去掉 Q1–Q4、Related Work 新增 5 条 WCNC 文献（引用现为 [1]–[24]）、
> 末页用 `\IEEEtriggeratref{8}` 平衡。合并时**重新应用了数据隔离**——随包带来的 `make_wcnc_figs.py`
> 是读 `../outputs/` 的旧版，本文件夹保留读 `data_snapshot/` 的隔离版（两者除路径/注释外逐行相同；
> 其图片与本文件夹图片逐像素一致）。原始 zip 已存 `backups/user_revision_0909_original.zip`。
>
> **修订记录 2**（同日，审稿意见响应）：表 I 改 ± 记法；软化 dominate 措辞；跨数据集证据写入正文
> （FedAvg 0.04 / adaptive 0.16）；rank-64 对齐、fair-cohort 顺序、种子(0–4)、共享带宽局限性、
> 算法反馈逻辑等澄清；修复 Fig. 4 y 轴标签裁切（改 "Scheduling accuracy"，proxy 移入题注）。
> 注：作者添加的 `\IEEEtriggeratref{8}` 因内容增长导致第 7 页，已移除——现末页天然平衡。

> **修订记录 3**（同日，第二轮审稿响应）：fair-cohort 协议精确化（采样→秩→预算准入顺序）；
> 式(3) 改述为 hazard-form surrogate（μ 吸收丢包/切换/中断）；补 B(0)=0、G(0)=0 定义；写明仿真器
> 代理递推公式；摘要/结论加 volume-aligned 限定；表 II 表头改 "12 MB budget / 1000 clients"；
> 算法 1 注明 EWMA 先验；聚合节声明 frozen-factor 承自 FFA-LoRA。新增鲁棒性实验
> `scripts/wcnc_scheduler_robustness.py` → `outputs/wcnc_extra/scheduler_robustness.csv`
> （预测误差 σ=0.2 → 目标损失 ≤5.1% 均值；线性/对数效用 → 准入变化 ≤0.5%）。注意：第二轮 R2 所审
> 的 `WCNC1.0.pdf` 为旧版，其乱码/±格式/r_max 指控在现稿均不存在。

> **修订记录 4**（同日，种子扩充实验完成）：新增 10-seed 复现实验（seeds 5–9，cifar10 名义 +
> 0.25 s 压力），运行器 `scripts/wcnc_run_seeds59.sh` → `outputs/wcnc_extra/real_v3_ext/`，
> 数据快照 `data_snapshot/wcnc_seeds59__all_metrics.csv`（**新增文件**，未覆盖原快照）。
> 结论已写入 Summary：10 种子下 adaptive vs r16 字节节省 p=0.002、及时率更高 p=0.016、
> 精度 parity p=0.40（R1-2.7 统计性质疑就此关闭）。表 I/图仍为 5 种子主分析，文中注明复现。
> **修订记录 5**（同日，第三轮审稿响应——Weak Accept/小修）：μ 如实改写为 trace 移动性特征
> （Beta(2,5) 公共随机数流；EWMA 为部署扩展，代码中无 EWMA——修正了上一轮算法行的失实表述）；
> π 如实改写（仿真器 U(0.15,0.45)，原型 π≡1）；补 B(0)/G(0)、B_meta 去向、B_i(r) 统一、
> 摘要 4% 配置限定、结论 pretrained-CNN 范围限定、DP 运行环境（Xeon 8358 单线程 NumPy）、
> 固定时延偏差方向、R̄ 有效速率/并行上传语义。**新增表 II（三数据集联邦精度）**，
> 相应删除信息量最低的 trainer 压力轨迹图与 Summary 小节（统计句移入结论）。
> 事故与修复：删除 Summary 时误删 Discussion 节，已从编辑史完整重建；表 IV 题注数字
> 与表体对齐。回归验证：6 页、[1]–[24]、终审通过。

> **修订记录 6**（第四轮审稿响应——Weak Reject/大修）：符号与模型修复（LoRA 维度 B∈R^{d×rmax}、
> A∈R^{rmax×k}；秩约定统一为 B 前 r 列 + A 前 r 行；payload 用 β bits 且 d,k 以百万计；
> DP 状态改 (k,e)、代价 ⌈B_i^t(r)/δ⌉、复杂度含 B_max/δ、u_i 用 hat-P、补 rank set R={2,4,8,16}；
> p_i 经 μ 吸收显式化；τ=上传窗口）。呈现修复（新增 Setup "two tiers" 定位句、solver-study
> 6.1% 缺口与 artifact 指引、表 III 增 SA-annealing 行 + Acc.→Proxy、九 regime 改
> "matches best to within 0.001"、结论 10-seed means 标注）。完整投稿包：
> `backups/WCNC_submission_complete_*.zip`（tex+bib+bbl+图+pdf，已在独立目录验证可编译）。

> **修订记录 8**（同日，精简重写 + 补图）：全文消除四轮补丁式冗余（删"三问"段、Summary 小节、
> 流程图 Fig 2——其内容由 III-B 文字承载；合并重复表述约 30 行）。腾出篇幅**新增两张实验图**：
> Fig 2 交付秩动态（三 deadline 下 adaptive 逐轮秩 + SA 盲扫对比，`fig_rank_dynamics.png`）；
> Fig 3 真实训练器中 DP-joint vs fair cohort vs r16 的逐轮按时交付（`fig_dp_timely.png`）。
> 图例均移至绘图区外。修复：Fig 2 正文引用补齐、结论 10-seed 表述改为明确的
> "A 10-seed replication (the five-seed tables plus seeds 5–9)" 框架。数据快照新增
> `wcnc_dp__all_metrics.csv`。投稿包同步更新（7 个文件，独立可编译验证）。

> **修订记录 9**（压页定稿）：采用用户提供的 Algorithm 1 全文（术语对齐：\hat P、G、δ、
> e 网格单位、q=⌈B/δ⌉），转移式合并为单行、循环压为一行，保持全部语义步骤；
> 正文配合削减约 20 行（V-D/VI/V-C/V-B/Intro）；Fig 2 纵轴标签缩短为 "On-time updates"
> 修复裁切；表 II 行标签统一 "Adaptive (fair-cohort)"；Eq(4) 的 b 改 β 与正文一致；
> 删除 poster 引用 [11]（R3-minor-7，出版信息不全）——引用现为 [1]–[23]；
> 末页以 \IEEEtriggeratref{4} 平衡双栏。**最终 6 页，投稿定稿**，
> 完整包 `backups/WCNC_submission_complete_20260910.zip`。

> **修订记录 10**（合稿——作者最终版 × 远程实验版）：以作者最终版 main.tex 为基底
> （自绘 wcnc.pdf 架构图、linespread 0.96、引言 P2 重写、恢复 poster 引用、扩展版算法 +
> footnotesize、表 III 完整小数），重新应用远程版本的全部实验与实质修复（μ 代理/EWMA、
> β、B(0)/G(0)、fair-cohort 协议、two arms 段、δ 内存、表 I DP-joint 行、10-seed、
> 鲁棒性、网格扫描、Proxy 表头、volume-aligned）。远程实验数据为准：10-seed 统计、
> DP-joint 行数值均来自 outputs/wcnc_extra 与 data_snapshot 快照。
> 作者图中笔误待改：satellite clients 标注为 D_a，应为 D_s（需在绘图源文件修改）。
> **最终 6 页，[1]–[24]，投稿定稿**。投稿包已更新。
> 遗留决策：是否在真实 trainer 中实现完整 DP 调度（R4-major1 选项 a），或维持现状定位
> （选项 b，已在 Setup 加 two-tier 边界句）。

> **修订记录 11**（基金资助信息）：Acknowledgment 节加入依托项目（英文译名 + 编号
> FFX24641X028），仅此一处改动；重编译仍 6 页、[1]–[24]。同步重建投稿包时发现旧包
> 缺 `wcnc.pdf` 架构图（此前未做独立编译验证），已补入；包内独立编译复验通过
> （6 页、无错误、基金号在文中）。

> **修订记录 12**（新增机制图 Fig 2）：作者确认架构图 D_a 标注不改；应用户要求补一张
> 帮助理解的图。新增 `fig_tradeoff.png`（生成脚本 `make_mech_fig.py`，纯解析曲线、
> 不读任何实验数据：hazard 估计器 P(r)、负载 s(r)=0.02r+0.5、增益 G(r)=√(r/16)、
> τ=0.25 s，稳定链路 16 Mbps/μ=0.1 vs 退化链路 6 Mbps/μ=0.4；右panel按曲线归一，
> 峰值标注 r*=4 / r*=16 与离散秩集 {2,4,8,16} 一致）。图插入 IV-A 权衡句处，正文
> 仅加 "(Fig. 2)" 括号引用。为控 6 页删 8 行真实冗余（引言路线图句、V 开头路线图句、
> "Omitting this factor" 防御句、III-C 与 Setup 重复的 Beta/CRN 句、III-C 重复的
> plug-in 句、表 IV 图注第二句、表 III 数字复述、结论 future-work 句（与 Discussion
> 重复）、0.04 of 10 复述）。judge 验收 6 页：两处 fail 经源码与放大渲染核实均为
> 误报（图例颜色正确；表 III Random 行 stress proxy 确为 0.359 与正文一致）。
> 最终 6 页、Figs 1–5、[1]–[24]。投稿包（10 文件，含 fig_tradeoff.png）独立编译复验通过。

> **修订记录 13**（作者反馈：Fig 2 放大 + 图文同页）：作者确认 Fig 2 偏小且要求图表与
> 引用正文同页。(a) fig_tradeoff.png 从 3.6×1.18 加高至 3.6×1.50 英寸（与其他图一致），
> 图例/标注字号 6.5→7。(b) 排查全部图表：Fig 1/2/3/5 与 Table III/IV 本已同页；
> Fig 4（引用在 p4、图在 p5）和 Table II（被 F4 挤到 p5）违规。将 fig:rank_dyn 移至
> fig:dp_timely 之后、tab:datasets 移至 Setup 段落之后（tab:trainer 之前曾短暂放错
> 导致表号互换，已纠正：trainer=Table I 不变），四个浮动体以 trainer/datasets/
> dp_timely/rank_dyn 顺序整体插到 Setup 之后、Impact 小节之前——在 LaTeX 排 p3 右栏
> 时被登记，自然填入 p4 左栏（Table I+II）与右栏（Fig 3+4）。最终：p2 Fig1、p3 Fig2、
> p4 Fig3+Fig4+表I+表II、p5 Fig5+表III+表IV，全部图文同页，6 页、[1]–[24]。
> judge 验收 p3–p6 全 pass（并复核上轮图例颜色误报不复现）。投稿包独立编译复验通过。

> **修订记录 14**（作者反馈：贡献不设小节）：删除 `\subsection{Contributions}` 标题，
> 贡献列表由 "This paper makes three contributions:" 直接引出并入引言正文（无编号变化，
> 无 label 引用该小节）。仅影响第 1 页版面（省 ~2 行），浮动体分布、6 页、[1]–[24]
> 全部不变。投稿包同步，独立编译复验通过。





> **修订记录 7**（同日，DP-in-trainer 实验——回应四轮审稿的核心遗留）：发现共享训练器中
> `DP_cohort` 完整 DP 调度臂**早已实现**（dp_assign，逐坐标回溯，--dp-budget 默认 8.2 MB）但从未运行。
> 新增运行器 `scripts/wcnc_run_dp_trainer.sh` → `outputs/wcnc_extra/real_v3_dp/`（零代码修改，
> backbone 逐种子复用）。结果（5 种子配对，全部同向，p=0.062 为 n=5 下限）：
> 名义 timely 0.57 vs 0.44、字节 7.88 vs 8.14；0.25 s 压力 timely 0.222 vs 0.160（vs r16 0.146）
> 且字节 7.56 vs 8.20；精度全部持平 0.914。已写入表 I（第 6 行 DP-joint）、Q1/Q2 文字、
> Setup two-tier 句重写、摘要与贡献提及。原 "Adaptive (ours)" 行改标 "Adaptive (fair-cohort)"。

## 隔离模型总览

| 资源 | 会议版（`WCNC_conference/`） | 期刊版（`FedLoRA lunwen/`） | 是否隔离 |
|---|---|---|---|
| 正文 | `main.tex`（IEEEtran） | `main_lora_clean.tex`（elsarticle） | ✅ 各自独立文件 |
| 参考文献 | `references.bib`（本地副本，19 条被引用） | `references.bib`（31+ 条） | ✅ 独立副本，互不影响 |
| 实验图 PNG | `fig_trainer_stress.png`、`fig_sim_stress.png`（本文件夹内） | `trainer_*.png`、`sim_*.png` 等 | ✅ 独立副本 |
| TikZ 矢量图 | `main.tex` 内嵌（Fig. 1/2） | `main_lora_clean.tex` 内嵌 | ✅ 各自源文件内 |
| 图片生成脚本 | `make_wcnc_figs.py`（**只读本文件夹 `data_snapshot/`**） | `scripts/make_paper_figs.py`（写 `FedLoRA lunwen/`） | ✅ 读写路径互不相交 |
| 实验数据 | `data_snapshot/`（**冻结快照**） | `../outputs/`（活的实验输出） | ✅ 会议版不再读取 `outputs/` |
| 编译产物 | `main.pdf/aux/bbl/...` | `main_lora_clean.pdf/...` | ✅ 各自文件夹内 |

## 数据快照溯源（data_snapshot/）

快照时间：2026-09-08。SHA256 与源文件比对一致，重生成的图与快照前逐像素一致。

| 快照文件 | 来源 | SHA256（前 16 位） |
|---|---|---|
| `real_fedlora_v3__all_metrics.csv` | `outputs/real_fedlora_v3/all_metrics.csv` | `0ab14341cfb2fd25` |
| `paper_sweep__all_round_metrics.csv` | `outputs/paper_sweep/all_round_metrics.csv` | `50f4e260d250fb22` |

## 今后的操作规则

### 修改会议版（不影响期刊版）
- 直接编辑 `WCNC_conference/` 内任何文件，随意改图、改数据快照、重编译。
- 会议版**唯一对外读取路径已清零**，怎么改都不会波及期刊版。
- 若要用新实验结果更新会议图：把新 CSV 复制进 `data_snapshot/`（覆盖前建议核对口径），再 `python make_wcnc_figs.py && pdflatex main && bibtex main && pdflatex main && pdflatex main`。

### 修改期刊版（不影响会议版）
- 编辑 `FedLoRA lunwen/main_lora_clean.tex`、其 `references.bib`、重新生成该文件夹内的 PNG，随便改。
- 重新跑实验、更新 `outputs/`：**会议版不会受任何影响**（它只读自己的快照）。
- 期刊版重新生成图片用的是 `scripts/make_paper_figs.py`，它只写 `FedLoRA lunwen/`，不会碰 `WCNC_conference/`。

## 期刊版遗留问题备忘（修改期刊版前必读）

1. **`FedLoRA lunwen/trainer_dl025.png` 是旧数据批次**：其 y 轴约 0.929–0.932，与正文/表格的 0.914 矛盾
   （会议版发现后已用 `all_metrics.csv` 的 `cifar10_dl0.25` 数据重制）。改期刊版时应同样替换该图，
   正确数据源：`outputs/real_fedlora_v3/all_metrics.csv`，dataset = `cifar10_dl0.25`。
2. **期刊正文 0.24 vs 表格 0.28 口径混用**：期刊版 Q2 段落的 "0.24 (nominal)" 是全程 40 轮平均，
   而表 I 是 30–39 轮窗口平均（0.28）；随后的 "lose 46%/83%" 也基于 0.24。改期刊版时建议统一口径
   （会议版的修法：正文锚定表 I 的 0.28，百分比改为 54%/86%，见 `WCNC_conference/main.tex`）。
3. 期刊版 `references.bib` 含未被引用的条目（如 `mcmahan2017communication`、`lai2021oort` 等），无害但可清理。

## 冻结备份

`../backups/` 下有本轮隔离完成时两个版本的 zip 快照（含日期），即使误改也可整体回滚。
