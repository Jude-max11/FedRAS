# R2 审稿回应跟踪（外部专家审稿，2026-10-12）

审稿依据：main (21).pdf（20 页版）。本文件跟踪每条意见的处置状态。
状态：FIXED（已改）/ RUNNING（GPU 批次运行中）/ PENDING（待数据落地后写入）/ USER（作者决定）。

## 逐条处置

### 大点 1：创新性表述过强，遗漏直接相关工作
- 1a. 加入 ROBA（IEEE TVT 2025, DOI 10.1109/TVT.2025.3623104）、TSFA（arXiv 2505.00333）、
      Adaptive FL（arXiv 2505.23555）三篇近邻 — **FIXED**（reference.bib + §2 新段落）
- 1b. 相关工作新增 "Wireless-aware rank and resource allocation" 段，定位差异收窄为
      "期望时延 vs 逐负载交付概率 + 联合准入" — **FIXED**
- 1c. 摘要/引言的全称否定表述收窄（"choose from training-side signals or average channel
      statistics; none conditions on the timely-delivery probability of the chosen payload"）— **FIXED**
- 1d. 定位表 tab:rwcompare 加入三行无线感知条目，caption 说明其不满足逐负载 delivery-aware — **FIXED**

### 大点 2：主实验未回答"通信换来什么学习收益"
- 2a. 分布偏移任务（干净预训练 vs 偏移客户端语料，--shift-severity，确定性协变量偏移）— **RUNNING**（批次2b）
  - 试点结论：CNN 层协变量偏移（sev 0.1/0.15/0.2，标准与强化协议 lr.01×2ep）冻结骨干 LoRA 均不恢复
    （骨干 0.914→0.83/0.75/0.63，联邦后持平或下行）→ **该层定位为鲁棒性边界**，非旗舰适配任务；
    旗舰证据 = ViT 层（骨干 0.807，适配空间大）+ 弱骨干不重叠语料层（2a）
- 2b. 预训练/客户端语料不重叠变体（--pretrain-frac 0.5）：weakdisj（弱骨干）+ ViT 偏移层均带不重叠 — **RUNNING**（2a/2b）
- 2c. ViT 骨干层（6 块/192 宽/patch4，LoRA on qkv/proj/MLP，41,472 B/rank，阶梯 2.0/0.5/0.25 s；骨干 80ep 预训练达 0.807）— **RUNNING**（预训练缓存 20 个排队中，2a 干净层 + 2b 偏移层）
- 2d. 零通信对照显式化（backbone_acc on shifted test；FedAvg-stress 行重定性为零通信对照实现）— **FIXED（文字）+ RUNNING（各层表行）**
- 2e. 到目标精度的累计字节/轮数指标（make_shift_adapt_fig.py 已就绪并冒烟通过）— **PENDING**（待数据）
- 2f. 弱骨干第二强度（5-epoch 预训练，10 种子 × 2 期限）— **RUNNING**（批次1 C 组）
- 2g. **骨干质量扫描（转换律）**：补 2/3/10-epoch 三档（bb2/bb3/bb10 族 × 6 方法 × 10 种子 ×
      2 期限，共 60 run，审计 0 修复），与 1/5/30 既有锚点构成 6 点曲线 — **FIXED（数据+正文）**：
      转换律单调——骨干精度 0.53/0.68/0.75/0.82/0.87/0.91 处，r16−r4 耦合 +1.54→+0.85→+0.43→
      +0.09→−0.06→−0.05pp；FedRAS−SA（stress，链路盲视对照）+1.06（p=.002）→+0.64（p=.049）→
      +0.29（p=.020）→+0.06（ns）→−0.01（ns）→~0。边界结论：**骨干精度 ≲0.8 时容量兑现为精度**，
      调度器捕获名义耦合的 2/3–3/4。正文 §Main Experiment（fig_exchange d）+ §weakbb 重写为
      "capacity-conversion law" + tab:failures 静态策略失败表（四策略×四域，FedRAS 最差格 0.41pp）
- 2h. **交付容量单一货币指标**（journal_delivered_capacity.py：dc = aggregated×update_norm，
      逐轮/累积/每字节/terminal 窗口 + 配对 Wilcoxon）— **DONE**（CSV 落地：
      cifar10_dl0.1 stress 下 FedRAS 交付质量 100% 最优、较 r16 +8.3% p=0.02、较 r4 +42% p=0.002；
      nominal 与 r16 恒等）——回应"通信换来什么"：换来的是按时聚合的更新质量
- 2i. **静态策略失败表**（print_static_failure_table.py，四策略×四域）— **FIXED（tab:failures）**：
      r4 处处按时最优但弱域 −1.5pp、容量恒 54–73%；r16 nominal 最优但 stress 按时 65%；
      SA 强骨干域与 FedRAS 打平、弱域 −1.4~−1.5pp；**FedRAS 四域最大损失 0.41pp、交付容量全格 100%**
      （SA 强骨干补跑 sa_strong 族 20 run，骨干缓存复用命中）
- 2j. **换算三货币**：能量（每焦耳交付容量较 r16 +41%，浪费空占 59%→51%）、带宽等价
      （r16 需 1.54× 速率追平 FedRAS 按时率）、容量视图（make_capacity_view_fig.py，
      accuracy vs 累计交付容量）— **DONE（分析）**，正文 Deployment 段已落数字

### 大点 3：网络侧结论与真实期限可行性
- 3a. τ 明确定义为上传交付窗口（本地训练完成后开启），模型节改写 — **FIXED**
- 3b. 本地训练耗时实测入 config（train_ms_total），论文报告 — **RUNNING**
- 3c. DP 开销与 τ 的关系（调度计入轮次准备而非 τ；23–421 ms vs <2 ms/决策）— **FIXED**
- 3d. §6.5.2 配置补全（期限 0.1 s 声明、结构化交付项明细）— **FIXED**
- 3e. 共享频谱→速率相关的独立性局限声明 — **FIXED**（limitations）

### 大点 4：DP 作用与命题 2
- 4a. §8.1 三处"预算宽松 ⇒ = 固定 rank16/回退 r_max 最优"的曲解改写（r* 闭式支撑）— **FIXED**
- 4b. "Greedy fail exactly when" → "can fail only when" — **FIXED**
- 4c. r* = R̄τ/16c 闭式解 Remark（含 μ 消去、慢链路小 rank 推论）— **FIXED**（eq:rstar）
- 4d. trainer 绑定预算实验（0.4 MB：DP vs value/vpb 贪心 vs 未检查逐客户端规则）— **RUNNING**
- 4e. 12.7 vs 13.0 澄清（同调度、独立交付抽样噪声）— **PENDING**（tab:arms 文字一句）
- 4f. 独立精确求解器交叉验证 — **PENDING**（verify_gcdp.py 已有暴力验证，写作时引用；如不足补做）

### 大点 5：√r 效用与移动性
- 5a. §6.2.1 "测得性质而非建模假设"软化 + 假设边界 — **FIXED**
- 5b. 容量指数敏感性扫描（solver study 加指数参数）— **PENDING**
- 5c. μ 不影响逐客户端 rank 的代数性质已随 r* Remark 写入 — **FIXED**
- 5d. LoRA 缩放系数/初始化说明 — **PENDING**（setup 一句：A 高斯 0.1 初始化、B 零初始化已有；补 lora-init-std 说明）

### 大点 6：外部基线
- 6a. 忠实单调收缩退火（SA_mono：20% 初始化期 16 → 单调降至 2）— **RUNNING**（批次1 E 组）
- 6b. 速率阈值基线（8 Mbps 阈值，r16/r2）— **RUNNING**（批次1 E 组）
- 6c. FP16 对比扩到 10 种子（p=0.062→若 10/10 则 0.002）— **RUNNING**（批次1 B 组）
- 6d. JointAB 按真实 A+B 负载端到端重跑（1.82×B-only 审计比）— **RUNNING**（批次1 I 组）
- 6e. **近邻文献机制的正面对照（审稿人明确要求"忠实复现近邻方法"，首轮漏排，已补）**：
  - ROBA/TSFA 式"期望时延、交付有保障"规则（读估计速率、按期望传输时间取最大可行 rank，
    无丢包项、无联合准入——与逐负载交付概率正面对照）— **RUNNING**（批次2c，冒烟通过）
  - FedARA 式 SVD 秩分配（服务端对聚合增量做逐层 SVD 能量截断→下一轮全局 rank，
    训练侧自适应正面对照）— **RUNNING**（批次2c，冒烟通过：rank 16→2 由能量驱动）
  - 弱骨干层同跑两法（按种子与既有 weak 运行配对）— **RUNNING**（批次2c）
- 6f. 基线命名与主张收窄（"after X / faithful-style under shared protocol"表述，结论只对
      复现机制不对原系统整体）— **PENDING**（随 6.3 重写）

### 大点 7：掩码聚合
- 7a. Algorithm 4 补空轮/W_j=0 规则；两种 W_j=0 回退消融（zero vs hold）— **FIXED（文字）+ RUNNING（weakhold 消融）**
- 7b. 链路-类别相关偏差实验（弱链路类 hold 特定类别，per-class-half 精度）— **RUNNING**

### 检查表
- §5.4 预算取整 → floor（论文+代码；实验预算均为网格倍数，数值零影响）— **FIXED**
- §6.5.2 数字无法复核 → AR(1) 截距 bug 已修（均值平稳到 2.5），5 种子重跑 — **RUNNING**
- Beta(0.7,5) 更温和 → 重定性 + Beta(2,1) 真压力场景（均值 0.67 vs 0.29）— **RUNNING**
- 1.64 MB vs 0.164 MB → 每客户端/每轮总量 + 1.8× 澄清 — **FIXED**
- 12.7/13.0 — **PENDING**（见 4e）
- 统计口径 → 全部精度改末 10 轮均值；"一个标准差内"降级为描述性；配对差 CI — **FIXED（协议文字）/ PENDING（表格数字重刷）**

## 附带发现（审稿人未点破）
- **AR(1) 均值衰减 bug**：原更新式无截距，log-rate 依 0.6^t 衰减到 0，实现速率塌到 1 Mbps
  地板——§6.5.2 的"严酷结构化信道"主要是这个伪影。已修复（+0.4×2.5 截距）并重跑。
- 工作区 clip_grad 裸名笔误（git HEAD 为 a.clip_grad）——已修复，否则所有新运行首轮即崩。

## 最终状态（2026-10-15 全部完成）
- 全部 13 家族 230+ 运行通过完整性审计（0 缺口）；主控 `scripts/run_r2_master.sh`（自适应并发 6/9/14/22 + 显存保险丝）。
- **论文回填完成**：§6.2 适配任务小节 + fig:adapt 双图（weakdisj/ViT偏移，含零通信线与 bytes 视角）；
  §6.3 近邻机制段（期望时延/SVD/单调退火/速率阈值）+ tab:extbaselines 扩 5 行 + FP16 十种子（p=0.002）+ JointAB 真实 A+B（0.28/1.64MB）；
  §6.4 绑定预算段（DP 0.69/0.34MB vs value-贪心 +12pp，10/10 p=0.002）+ W_j hold 消融（+1.7pp，方差减半）；
  §6.5 mm2 重写（0.59/0.73 vs r16 0.52/0.90）+ 相关偏差段（4.9 vs 5.8/3.2pp）+ 实测训练时长（2.1s/客户端轮）+ Beta(2,1) 数字；
  §6.6 不变；§8 结论补齐（+3pp 弱骨干不重叠、+1pp ViT 偏移、绑定预算 +12pp、近邻基线族结果）。
- 编译 0 错 0 未定义 0 重复 label，22 页。叙事要点：链路感知规则族整体优于链路盲调度（弱骨干 0.547-0.565 vs 0.536-0.544），
  FedRAS 族内最少字节/最多及时；ViT 干净层=持平、偏移层=恢复 +1pp 的诚实框架。
