# WCNC 往届录取论文的语言与结构分析

## 1. 样本与结论边界

本分析检索了 Crossref 中由 IEEE 登记、会议名为 *IEEE Wireless Communications and Networking Conference (WCNC)* 的 2023–2025 年 proceedings article。检索到 2023 年 444 条、2024 年 842 条、2025 年 725 条。记录可能包含 workshop 或特殊会场，因此这些数字用于建立论文样本，不应解释成主会官方录取数。DOI 与题目属于可核验题录；Crossref 没有提供这些记录的摘要，故标题统计是定量结果，正文语言与结构是结合公开论文、IEEE 双栏通信会议论文和当前稿件做出的定性归纳。

题目统计如下：

| 年份 | 题录样本数 | 平均词数 | 中位词数 | 使用冒号 | 使用问号 |
|---|---:|---:|---:|---:|---:|
| 2023 | 444 | 10.3 | 10 | 12.2% | 0.2% |
| 2024 | 842 | 10.5 | 11 | 19.4% | 1.0% |
| 2025 | 725 | 10.6 | 11 | 20.3% | 0.7% |

这说明 WCNC 题目通常约 10–11 个英文词，采用直接陈述式；问句极少，冒号虽在增加但仍不是默认写法。

## 2. 可核验的代表性录取论文

以下论文都可由 IEEE DOI 核验，主题与当前稿件的联邦学习、边缘计算、资源调度或学习方法较接近：

- 2023, *Discrepancy-Aware Federated Learning for Non-IID Data*, https://doi.org/10.1109/WCNC55385.2023.10118804
- 2023, *Deep Reinforcement Learning-based Quantization for Federated Learning*, https://doi.org/10.1109/WCNC55385.2023.10118781
- 2023, *Federated Learning Framework for Dynamic Power Management in RAN Data Plane Systems*, https://doi.org/10.1109/WCNC55385.2023.10119007
- 2023, *Graph based Joint Computing and Communication Scheduling for Virtual Reality Applications*, https://doi.org/10.1109/WCNC55385.2023.10118639
- 2024, *Adaptive Model Pruning for Hierarchical Wireless Federated Learning*, https://doi.org/10.1109/WCNC57260.2024.10571275
- 2024, *Model Selection Based on DRL: Improving Personal Model Performance in Federated Learning*, https://doi.org/10.1109/WCNC57260.2024.10571185
- 2024, *Learning User Scheduling and Hybrid Precoding with Sequential Graph Neural Network*, https://doi.org/10.1109/WCNC57260.2024.10570940
- 2024, *Delay-aware Distributionally Robust Trajectory UAV-assisted MEC with Uncertain Task Size*, https://doi.org/10.1109/WCNC57260.2024.10571255
- 2025, *Integrated Sensing-Communication-Computation Based Online Federated Learning with Limited Cache*, https://doi.org/10.1109/WCNC61545.2025.10978235
- 2025, *Learning-Based Low-Latency Collaborative Inference for Multi-Branch Models in D2D-Assisted MEC*, https://doi.org/10.1109/WCNC61545.2025.10978329
- 2025, *Contention-Aware Microservice Deployment in Collaborative Mobile Edge Networks*, https://doi.org/10.1109/WCNC61545.2025.10978411
- 2025, *Throughput-Optimal Multi-Link Access for Wi-Fi 7 via Multi-Agent Reinforcement Learning*, https://doi.org/10.1109/WCNC61545.2025.10978161

## 3. 常见结构

### 摘要：五步压缩叙事

典型摘要用一个段落完成五件事：场景或任务、明确缺口、本文方案、关键技术、带数字的主要结果。最常见的句间逻辑是：

1. `X is promising/important for Y, but ...`
2. `Existing methods ...; however, ...`
3. `To address this issue, we propose ...`
4. `Specifically, ...` 或 `We formulate ... and develop ...`
5. `Simulation/experimental results show that ...`

篇幅紧，背景通常只有一到两句。结果不能只写 “outperforms existing methods”，应给出指标、比较对象和幅度。

### 引言：漏斗式四段到六段

高频顺序是：应用背景与技术价值；现有研究按问题分组；精确指出尚未解决的矛盾；用一句话提出核心想法；列出二至四项贡献；最后给出文章组织。WCNC 偏网络与系统，因此缺口最好落到无线条件、时延、能耗、频谱、可靠性、移动性或可部署复杂度，而不能只停留在模型精度。

### 正文：问题—方法—验证

最常见的主干为：

1. `Introduction`
2. `System Model` / `Problem Formulation`
3. `Proposed Method` / `Algorithm Design`
4. `Performance Evaluation` / `Simulation Results`
5. `Conclusion`

`Related Work` 可以独立成节，也常压缩进引言以节省六页篇幅。理论型论文会在方法节加入定理、证明、复杂度或收敛性；优化型论文先定义变量、目标和约束，再给求解算法；学习型论文应说明训练/推理流程、通信代价和无线系统接口。

### 实验：先可复现，再回答问题

结果节通常先集中给出参数、数据集、信道/拓扑、硬件或仿真平台、基线和指标，然后按结论组织图表。图题应能独立说明横纵轴和场景。强论文不仅展示平均性能，还包括消融、复杂度/运行时间、规模扩展、敏感性或极端网络条件中的至少一项。

## 4. 语言特点

- **直接、压缩、以动词推进。** 常用 `formulate`, `derive`, `design`, `develop`, `prove`, `evaluate`, `demonstrate`，每个动词对应一项可核查工作。
- **限定结论范围。** 常写 `under the considered setting`, `for the evaluated scenarios`, `up to X%`，避免 `always`, `significantly` 或 `state-of-the-art` 等没有统计或全面比较支撑的词。
- **段首给主张，段尾给意义。** 中间放公式、机制或证据，读者不必从细节反推本段目的。
- **贡献是成果清单。** 每条包含“提出了什么 + 为什么新 + 有什么证据”，不把背景、章节预告或实现步骤当贡献。
- **术语保持一一对应。** 一个对象固定一个名称和符号；缩写首次出现时定义，此后不换同义词。
- **结果句具体。** 推荐 `reduces the transmitted bytes by 14% while maintaining ...`，弱化 `achieves superior performance`。
- **谨慎使用破折号、括号和多重并列。** 双栏论文行宽短，过长句和连续插入语会明显降低可读性。

## 5. 与当前稿件的对照

当前 `main.pdf` 为 IEEE letter 双栏 6 页，约 5030 个 PDF 文本词；题目 9 个左右的实词，接近往届题目的典型长度。摘要约 179 词，包含问题、方案、算法、实验和量化结果，结构正确。

主要优点：

- 网络侧缺口明确：LoRA rank 被定义为与链路和 deadline 联动的调度变量。
- 方法链条闭合：交付概率、联合指派、精确 DP、异构 rank 聚合都有对应机制。
- 结果给出数值、基线和运行时间，符合 WCNC 对系统与网络证据的偏好。
- 主干是 `System Model → Scheduling → Performance Evaluation`，与该会议常见结构一致。

优先修改项：

1. **把贡献从四条压到三条。** 当前“模型、求解器、聚合、两层评估”都写得很长。可合并模型与求解器，保留“联合建模与精确求解、异构 rank 聚合、端到端验证”三条。
2. **摘要减少数字堆叠。** 179 词可接受，但最后两句同时出现数据集、seed 数、500 clients、多个 deadline 数字和 solver 时间。保留两个最能支撑主张的结果，其余留在实验节。
3. **压缩独立 Related Work。** 六页中同时保留独立相关工作、四条贡献、四个 Q、小结和 Discussion/Limitations，结构略碎。可把相关工作压成引言中的两段，释放空间给假设解释或关键实验。
4. **弱化容易触发审稿质疑的绝对表达。** `solved exactly` 应始终与“预算离散化后”“对估计的单轮问题”绑定；`preserve backbone accuracy` 更适合写成 `achieve statistically comparable accuracy in the evaluated tasks`，除非已有显著性检验。
5. **缩短引言的自我防御句。** 关于 preliminary poster、数字不可比和 `claim no universal accuracy dominance` 的整段占用黄金位置。投稿系统若要求披露重叠，可简化成一句；科学边界放到 Discussion。
6. **让 Q1–Q4 标题陈述变量关系。** `Accuracy parity... (Q1)` 等问题式标签利于内部写作，但最终稿可改为 `Impact of Rank on Accuracy and Traffic`、`Adaptation under Deadline Stress` 等，让目录和页首信息更直接。
7. **检查核心收益与目标的一致性。** 论文目标是 expected useful updates，实验最显眼结果之一却是 accuracy parity。主图和摘要应优先证明 deadline-aware rank selection 相比 fixed-rank、greedy、full update 在相同预算/精度条件下改善 timely utility。

## 6. 可直接采用的结构

建议六页终稿采用：

1. `Introduction`：背景、相关工作与 gap、三条贡献。
2. `System Model and Problem Formulation`：架构、rank-dependent payload、及时交付概率、目标与约束。
3. `Deadline-Aware Adaptive-Rank Scheduling`：DP、复杂度、异构 rank 聚合。
4. `Performance Evaluation`：setup、accuracy/traffic、deadline stress、基线与消融、runtime/scalability。
5. `Conclusion`：三到五句，只回答做了什么、发现什么、适用边界。

这一结构能减少章节切换，把审稿人最关心的链条放在同一阅读路径上：rank 改变 payload，payload 改变及时交付概率，调度器据此选 client–rank，实验验证在相同约束下的收益。

## 7. 数据来源

- Crossref REST API，2023–2025 WCNC proceedings-article 题录：https://api.crossref.org/works
- IEEE DOI 解析与论文落地页：上列各论文 DOI。
- 当前稿件：`main.tex` 与 `main.pdf`，用于章节和字数对照。

