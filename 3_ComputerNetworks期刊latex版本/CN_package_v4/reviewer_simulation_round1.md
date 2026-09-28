# FedRAS (CN_reviewed.tex) 模拟评审 · 第 1 轮
日期 2026-09-28。三名隔离审稿人独立评审（互不可见），材料包 = CN_reviewed.tex (38pp, commit e1bb2f4) + 渲染 PDF。
期刊口径：Computer Networks (Elsevier)。结论：三人一致 Major revision，无 Blocking=yes 项。

## Orchestrator 事实核查（综合前验证）
- R1-M1（图7 与 "≤1%" 矛盾）→ **驳回**。纵轴 0–1.0%，最高柱 ≈0.96%，图文一致（审稿人误读刻度）。
- R3-m1（§6 导语硬编码 Section~5.2/5.3/5.4 旧编号）→ **属实**（tex L304，指错到理论章）。
- Capacity 前向引用（L144 "delivered-update norms … (Section 6)"）→ **属实**，§6 无任何 update-norm 测量。
- "offline control" 句（L596）→ **属实**，全文无定义无数字无表图锚点。
- R1-M2（两层字节记账冲突）→ **部分成立**：更可能是模拟器骨干/载荷配置未写明（r16 每客户端 0.100 MB vs trainer 0.090 MB），而非记账矛盾；缺的是模拟器 payload 参数披露。
- R2-m4（FedAvg stress 精度有方差 vs "0 送达"矛盾）、R2-m11（DP 运行时"可达网格收缩"解释）→ 待查日志/代码后定。

---
（以下为三份完整报告原文 + 交叉综合）


# 交叉综合（post-review synthesis，审稿人不可见）

## 共识优势（≥2 人独立提及）
- 问题形式化干净（MCKP+准入上限），三命题正确且范围声明严格（3/3）。
- 端到端字节审计自洽，deadline 按真实载荷标定的方法学可信（3/3）。
- 统计协议高于领域平均（公共随机数、TOST、Wilcoxon、bootstrap），局限披露坦诚（3/3）。
- 命题 2 的"廉价规则何时最优"与 6.2.5 三臂分解是有信息量的设计（2/3）。

## 共识重大问题（按建议修复优先级排序）

**P1 ｜ 主张与自家表格的支配关系（最重，R2-M1 + R3-M1，R1 佐证）**
Table 3(a) 中 Random rank 在 0.1 s 下 timely 0.56 / 0.42 MB，Table 3(c) 中 annealing 0.62 / 0.29 MB，双双同时优于 FedRAS 的 0.49 / 0.69 MB；而 TOST 等效结论说明 FedRAS 唯一占优的 mean delivered rank（12.7 vs 6.2/5.0）在该评估视野内无精度回报。摘要只与 fixed rank 16 对比，构成弱基线叙事。
修复：摘要与 6.2.2 重述为"容量保持型取舍"（timely/bytes/rank 三指标并列），正面披露支配关系；要么给出保留容量的经验收益（更弱骨干/更长视野/分布偏移），要么把主张限定到容量有回报的域。

**P2 ｜ Capacity 模型 (Eq.5) 的经验锚点缺失（3/3 共识：R1-M3, R2-m5, R3-M2）**
L144 声称平方根容量律概括"trainer 中 directly 展现的 update-norm 递减"，§6 无此测量；π_i 仅在模拟器定义，且 Table 4(a) utility 列与目标函数同构（部分循环）。
修复：补按秩 update-norm 测量（一张小图），或删除指涉并明确为建模假设+形状敏感性分析；为 trainer 定义 π_i；4(a) 列名改 Utility。

**P3 ｜ Proposition 2 缺假设（R1-M4 + R3-M3）**
证明写 "uses no more than the cap"，仅当 |候选集|≤K 成立；按 5.2/7.1 的"默认跑廉价规则"在 500 客户端全池上会输出不可行解。
修复：命题显式加 |候选集|≤K 前提（或改为 top-K 入围再逐客户端最大化），同步修正 5.2 末段与 7.1 部署配方。

**P4 ｜ 交付模型不可复现（3/3 共识：R1-M7, R2-M2, R3-M5）**
μ 构造、Δd_h、丢包生成、SAGIN 三类链路律、B_meta 数值、模拟器载荷配置、各制度预算均未报告；全部网络侧头条数字产生于 oracle 合成链路；Data availability 无匿名链接。
修复：正文或附录补齐参数表；提供匿名仓库；摘要为头条增益加 "synthetic links, oracle estimates" 限定。

**P5 ｜ 统计报告与措辞（R2-M3 + R1-m9 + R3-M5）**
TOST 全部结论零细节（无统计量/CI/种子数/多重比较），"pre-registered" 无注册工件；22% 头条无法从表复算（0.49/0.41=+19.5%），21% vs 20% 同类问题（3/3 提及）。
修复：为每个比较报 TOST 统计量与 CI（标种子数），或改 "fixed in advance"；定义 22% 的估计量（逐种子配对比值均值）或在表中给一位小数。

**P6 ｜ 已核实的成稿错误（改文字即可，全部 15 分钟内可完成）**
1. L304 硬编码 Section~5.2/5.3/5.4 → 改 \ref 指向 6.2/6.3/6.3.2–6.3.3（R3-m1，已验证）。
2. 6.3.2 "up to 64%" 归入 binding 域，实为 slack 单元值（binding 内最大 54.8% cohort-myopic / 47.8% val-per-byte）（R1-m6+R2-m2+R3-m2，3/3）。
3. L596 "offline control" 无出处 → 补数字或删句（R2-m7+R3-m7）。
4. 贡献 1 第二个无限定 "for the first time" 与自家 2.3 节 deadline-aware 参与线矛盾 → 删或改写（R3-M4）。
5. 3.4 "strictly smaller dimensionality" 攻击面表述 vs 公开冻结 A 可重构 → 改为传输维度精确陈述（R1-M6+R3-m8）。
6. Fig.3 图题 "flat in α" 与 Table 3(b) α=0.1 的 0.32pp 下降矛盾 → 限定（R2-m6）。
7. Table 4(a) "Acc." 列与 trainer 精度同名同值 → 改 Utility（R2-m1+R3-m5）。

**P7 ｜ 待查日志后定（不修文字，先核数据）**
- FedAvg stress 精度 .9153±.0015 vs "零送达应零方差"（R2-m4）。
- DP 运行时随预算收缩的"可达网格"机制解释（R2-m11；此句出自本轮扩写，若实现是状态饱和应改为"运行时随 B̃_max 伪多项式缩放"）。
- 22% 是否确为逐种子配对均值（R1-M5）。
- 表4(b) Greedy value 与 Fixed r=16 九格损失完全相同（R1-m5）。

**P8 ｜ 实验补充类（量大，可选/下一步）**
- 压缩基线（fixed rank + FP16/量化）以隔离字节轴贡献（R2-M4）。
- 容量回报机制实验：弱骨干/更长视野/分布偏移下的 stress 对比（R2-M1/R3-M1 的修复选项 B）。

## 其余 minor（详见各报告）
表 3(b) 无标准差、精度混用、0.05s 列种子数、(d) 为 (a) 子集脚注、B 符号双义、TOST 措辞、表1 Oort 行约定、μ/B_meta、FP16 列用途、FedAvg 0.4×学习率依据、1000-client 无数字支撑、图例色依赖、灰度可达性、2.4 节补 FedPara 类与"通信联合更新规模调度"邻近线定位等。

## 综合判断
三审一致 Major revision、零 Blocking。核心案例（vs fixed rank 16、vs full-model）内部证据成立；被击中的是主张标定（弱基线支配未披露）、一个未兑现的前向引用、一条缺假设的命题、可复现参数缺口与一批可立即修复的成稿错误。无需推翻结论，无需要新的大型实验（若选"主张重构"路线）；P6 一天内可清零。

---

# Reviewer 1（侧重：建模与理论严谨性）— 推荐 Major revision

**Overall assessment**
这篇稿件把 LoRA 秩当作受截止期与字节预算约束、可被网络侧调度的资源，问题形式化清楚（多选择背包加准入上限），三个命题本身正确且范围声明谨慎，trainer 层的数值内部一致性给我留下很深的印象（我逐字节核对了表2与表3的载荷与吞吐，5,632 字节每秩列、0.90 MB、52.86 MB 等均能精确对上）。但稿件在自身证据链上存在若干必须修补的裂缝，最严重的是图7右图与其"至多1%"文字声明直接矛盾（※经主持复核：图纵轴 0–1.0%，最高柱 0.96%，该项驳回）、模拟器层与 trainer 层的字段记账互相冲突（涉及贡献1的"real serialized sizes"主张）、以及式(5)容量模型所引用的训练器经验证据在正文中缺失。这些问题均可通过修订解决而不需要推翻中心结论，因此我的立场是大修而非拒稿。

**Who would be interested**
移动边缘与星地一体网络中的 FL 研究者（截止期感知的参与与秩联合调度）；联邦 PEFT 社区（异构秩冻结因子聚合精确性）；网络运营者（真实序列化字节的 deadline 标定与预算规划）；资源分配算法研究者（MCKP 在 FL 调度中的应用与贪心适用边界）。

**Major strengths**
- 形式化干净，式(6)(7) 统一参与与秩选择，式(8) DP 正确，复杂度界无误。
- 命题2给出廉价规则已最优的可检查条件；命题3精确刻画两种归一化取舍。
- 评估协议规范：公共随机数、十种子复制、TOST、solver study 同一 ceiling 网格、网络侧与精度分报。
- trainer 层载荷审计自洽到字节级；deadline 标定与式(3)手算一致。
- 局限讨论异常坦率，声称范围克制。

**Major Concerns**
- R1-M1（图表一致性）6.3.3 与图7 "≤1%" vs 右图柱高。※经核实为误读，驳回。
- R1-M2（可复现性/贡献1）"real serialized sizes" 下 trainer 按无元数据记账、模拟器各行隐含每客户端约10KB元数据，B_meta 从未给值，两层口径冲突。修复：给 B_meta 数值与构成，统一口径（※复核：更可能是模拟器载荷配置未写明）。
- R1-M3（机制证据）式(5) 引用 "trainer 直接展现的 update-norm 递减" 在 §6 缺失；容量项无实测支撑， FedRAS 对退火基线的优势完全依赖未验证的曲线形状。修复：补测量或形状敏感性。
- R1-M4（命题2完备性）证明 "uses no more than the cap" 仅当候选≤K；500客户端全池条件下恒假，部署规则输出不可行解。修复：加 |候选|≤K 前提。
- R1-M5（统计严谨）22%（及21%）无法从任何表复算，估计量未定义。修复：定义或对表。
- R1-M6（威胁模型）"strictly smaller dimensionality" 不成立：A 公开，截获 B 切片可重构秩 r 诱导更新。修复：改传输维度精确陈述。
- R1-M7（可复现性）μ 构造、SAGIN 各类链路律、d_i、Δd_h 未给值；oracle 状态；无匿名仓库。修复：补参数并给链接。

**Minor Comments**
R1-m1 P_on 与 P̂_on 记号漂移+3.5节回指未定义；R1-m2 骨干参数与表2核对（1.28M→FP32 5.12MB vs 5.29MB）、D 的数据集差异；R1-m3 表3(b) 无标准差、精度混用、0.05s 种子数、Beta(0.7,5) 标签；R1-m4 表3(d) 为 (a) 种子子集应脚注；R1-m5 表4(b) 两列九格损失全同需解释；R1-m6 "up to 64%" 域归属错；R1-m7 B̃_max 上取整松弛至多超预算一个 q；R1-m8 表1 Oort 行约定不一致；R1-m9 pre-registered 无注册入口；R1-m10 Data availability 无可核查链接；R1-m11 B 符号双义；R1-m12 "first" 保持对冲并复核检索词。

**Technical failings** R1-M1（已驳回不成立）、R1-M2、R1-M3、R1-M4 必须处理；M5/M7 实质性修补，M6 技术性更正。

**Criteria** (a) 强契合 (b) 总体成立但有四处需修复 (c) 可辩护的组合式新颖 (d) 协议好但外效度受合成链路与容量项缺证削弱 (e) 高于平均 (f) 参数缺失+无链接，暂不足。

**Recommendation posture**: Major revision。方法骨架与 trainer 证据链扎实；若逐项落实（多数可由现有日志完成），本文有望成为该方向一个有用且诚实的参照点。

---

# Reviewer 2（侧重：实验效度与统计）— 推荐 Major revision

**Overall assessment**
面向 CN 读者定位准确，问题界面新颖，机制级命题正确。但就作者自报的网络侧指标而言，真实训练器中随机秩与退火基线在 0.1 s 同时超过 FedRAS 的及时率与字节数，而论文自身等效性结论表明保留的秩容量在该视野内无精度收益，实质性削弱实用显著性论证。交付规律、统计报告与压缩基线均需补齐。

**Who would be interested**
移动边缘与星地一体化 FL 调度、无线 deadline-aware 系统设计研究者（MCKP 形式化、Prop 2 部署判据）；边缘部署工程师（载荷审计与 deadline 阶梯方法学）；通信高效 FL 社区（调度与压缩可组合接口）。

**Major strengths**
- 接口新颖且定位诚实；Table 1 维度明确，贡献有限定词。
- 三命题正确且范围严格；Prop 2 在 Table 3(d) 名义格点被验证为恒等。
- 方法学高于平均：载荷审计、deadline 阶梯、公共随机数、同网格 solver study、量化误差测量、预测器失真研究。
- 局限部分异常坦诚，含多个不利自身的阴性结果。

**Major Concerns**
- R2-M1（基线公平/显著性）Random 0.56/0.42 与 annealing 0.62/0.29 在 0.1 s 同时占优 FedRAS 0.49/0.69，精度不可区分；FedRAS 唯一占优的交付秩容量被自家 TOST 判定无回报；摘要弱基线叙事。修复：重构主张或给出容量收益机制（弱骨干/长视野/偏移）。
- R2-M2（可复现性）μ 构造、J/H 生成、Δd_h、T_train/T_proc 取值缺失；两层及时率差异（0.17 vs 0.34）无解释；模拟器各制度预算未写明。修复：完整规格化。
- R2-M3（统计严谨）TOST 结论零细节、五种子格点做 ±0.5pp TOST 功效存疑、多重比较未处理、pre-registered 无记录。修复：全量报告或改措辞。
- R2-M4（缺失比较轴）Table 2 自示 fixed r16+FP16 仅 0.045 MB，平凡组合即超 23% 节省；无压缩/稀疏化基线。修复：补一个压缩基线或限定字节主张为 FP32 冻结因子格式内。

**Minor Comments**
R2-m1 表4(a) "Acc." 列实为 utility 代理应改名并给离散度；R2-m2 "up to 64%" 域归属与表格矛盾；R2-m3 22% 聚合口径未标注；R2-m4 FedAvg stress 精度非零方差 vs "零送达" 应核实并软化绝对表述；R2-m5 update-norm 前向引用未兑现；R2-m6 Fig.3 图题 "flat in α" 与表3(b) 矛盾；R2-m7 "offline control" 无出处；R2-m8 FedAvg 0.4× 学习率无依据；R2-m9 1000-client 制度无数字支撑；R2-m10 B_meta 无值、FP16 列用途未说明；R2-m11 DP 运行时"可达网格收缩"解释与常规 DP 分析不符；R2-m12 逐格种子数未标注；R2-m13 表3(e) 精度混用；R2-m14 建议在 2.3/2.4 预防性定位"截止期约束下自适应更新规模/工作量调度"邻近族。

**Technical failings** R2-M1、R2-M2、R2-M3、R2-M4。均可用修改、补充报告与少量针对性实验解决。

**Criteria** (a) 高度契合 (b) 命题正确，缺口在交付规格与运行时解释 (c) 可信但建议加固 (d) 协议扎实但弱基线叙事/缺压缩基线/统计报告不足 (e) 清晰坦诚，少量不一致 (f) 超参数好，交付规律与代码可获取性不足。

**Recommendation posture**: Major revision。若补统计细节与交付规格、增加压缩与容量回报实验、按证据重构摘要叙事，有望达到录用标准。

---

# Reviewer 3（侧重：创新定位与期刊契合）— 推荐 Major revision

**Overall assessment**
写作异常坦率，payload 审计与统计协议高于平均，核心数字自洽（已核对表2、deadline 校准、DP 网格代价）。核心问题不在数据而在校准与呈现：Table 3(a)(c) 显示 random 与 annealing 在 0.1 s 双指标优于 FedRAS 而摘要只与 fixed rank 16 比；capacity 模型声称的实测依据在 §6 不存在；Prop 2 缺一条必要假设；另有一处明确过时的交叉引用。

**Who would be interested**
移动边缘与无线 FL 读者（载荷作为显式决策变量、deadline 阶梯方法学、500 客户端 MCKP 毫秒级可行性）；SAGIN/NTN FL 与异构聚合读者（Prop 3）；部署工程师（部署处方）。

**Major strengths**
- 形式化清晰诚实，精确性边界写明。
- 端到端字节审计可信（1,408 参数/秩列、5.29 MB、58.7×、1.0 MB=十格代价全互洽）。
- 统计协议认真且声明不把网络侧优势读成精度优势。
- 局限披露彻底，preliminary 版本披露充分。
- Prop 2 适用域分析与 6.2.5 三臂分解有信息量。

**Major Concerns**
- R3-M1（主张校准）头条对比只选 fixed rank 16；random/annealing 双指标占优未被叙述正面处理；唯一占优维度（交付秩）在评估域内无可测收益。修复：重述为容量保持型取舍，明示被谁占优，给容量收益或限定主张域。
- R3-M2（机制证据）Eq.5 声称的 trainer 范数证据缺失；π_i 仅模拟器定义；utility 列与目标同构部分循环。修复：补数据、定义 π_i、明示列性质。
- R3-M3（技术正确性）Prop 2 缺候选≤K 假设；500 客户端全池检查约 45MB 对 30MB 恒假而表格显示 slack 中 fixed r16 已 0% 损失。修复：补假设并限定 5.2/7.1。
- R3-M4（新颖性措辞）第二个无限定 "for the first time" 与自家 2.3 节 deadline/reliability-aware 参与线矛盾。修复：删除或改写。
- R3-M5（可复现性）μ 映射、Δd_h、丢包、SAGIN 类律、基线重实现参数、pre-registered 工件缺失。修复：补齐或删词。

**Minor Comments**
R3-m1 §6 导语硬编码旧编号 Section~5.2/5.3/5.4（※已验证属实）及 2.4 硬编码；R3-m2 "up to 64%" 归属错；R3-m3 摘要跨层归因（22%/23% 出自 trainer，模拟器应单列且约 +19%/−20%）；R3-m4 22% 估计量不明；R3-m5 表4(a) 表头与 "best or tied" 应限定列；R3-m6 秩规则 P_on/P̂_on 记号；R3-m7 offline control 无出处；R3-m8 3.4 攻击面表述隐含隐私收益且无引用；R3-m9 图注颜色依赖与图例可读性；R3-m10 表1 payload accounting 列区分度弱；R3-m11 建议补 FedPara 类与"通信联合更新规模"邻近线定位；R3-m12 表3(e)/(a) 同配置两均值应脚注。

**Technical failings** R3-M1 与 R3-M2（绑定）、R3-M3、R3-M4、R3-M5；R3-m1/m2 为必须修正的成稿错误。

**Criteria** (a) 强契合 (b) 总体成立，Prop2 缺假设+容量锚点缺位 (c) 可辩护但第二处 first 越界 (d) 高于平均但 oracle 合成链路、20客户端、proxy 对齐与低秩基线占优限制强度 (e) 清晰坦诚，存在过时引用与两处文表不符 (f) 种子/超参好，交付模型关键参数与代码链接缺。

**Recommendation posture**: Major revision。以分析与重写为主、不需新的大型实验（若走主张重构路线）；完成前，稿件建立的只是"对 fixed rank 16 有改进"这一较窄案例。
