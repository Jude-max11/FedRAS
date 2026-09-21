# WCNC 会议版本说明

本文件夹是面向 **IEEE WCNC**（Wireless Communications and Networking Conference）的会议版本，
由期刊版 `../FedLoRA lunwen/main_lora_clean.tex`（Elsevier `elsarticle`，Computer Networks 投稿版）改写而来。
**原期刊版文件未做任何修改。**

## 文件清单

| 文件 | 说明 |
|---|---|
| `main.tex` | 会议版正文（IEEEtran 会议模板，双栏；0909 作者修订版） |
| `main.pdf` | 编译产物（6 页，符合 WCNC 初审页数上限） |
| `references.bib` | 参考文献库（含 0909 修订新增的 5 条 WCNC 文献，正文引用 [1]–[24]） |
| `WCNC_ACCEPTED_PAPER_WRITING_ANALYSIS.md` | WCNC 往届录用论文语言与结构分析（Crossref 题录统计 + 代表论文清单） |
| `fig_trainer_stress.png` / `fig_sim_stress.png` | 实验图，由 `make_wcnc_figs.py` 从**本地数据快照**生成 |
| `data_snapshot/` | 冻结的实验数据快照（会议版唯一数据来源；含 5 种子主数据与 10 种子复现数据） |
| `make_wcnc_figs.py` | 图片重制脚本（只读 `data_snapshot/`，不读 `../outputs/`） |
| `ISOLATION.md` | 版本隔离说明：溯源、修改规则、期刊版遗留问题备忘 |
| `../backups/` | 两个版本的冻结 zip 备份（带日期），含 0909 修订原始包 |

## 依据的 WCNC 2026 官方要求（wcnc2026.ieee-wcnc.org/submission-guidelines）

- 初审投稿 **最多 6 页**（10 号字），超页直接拒审；终稿最多可加 2 页（每页 100 美元）
- IEEE 标准 LaTeX 会议模板（IEEEtran，双栏），PDF 经 EDAS 提交
- 会议时间地点：2026-04-13～16，吉隆坡

## 相对期刊版的主要改动

1. **模板**：`elsarticle`（单栏 12pt preprint）→ `IEEEtran [conference]`（双栏 10pt）；
   摘要压缩约 40%，关键词改为 IEEE Index Terms，参考文献样式 `elsarticle-num` → `IEEEtran`。
2. **结构**：按 WCNC 录用论文惯例重组为 I. Introduction（含贡献列表）→ II. Related Work →
   III. System Model → IV. Scheduling（问题、精确 DP、聚合）→ V. Evaluation → VI. Discussion → VII. Conclusion。
3. **内容精简**（期刊版全部结论与数字保留）：
   - 期刊版 5 张表 → 3 张表（表 I 合并 nominal+0.25 s stress；表 II 仿真 4 regime 压缩列；表 III 求解器研究）；
     敏感性扫描、扩展 stress 矩阵、外部基线表改为文字叙述，数字完整保留。
   - 期刊版 5 张图 → 4 张图：**新增 Fig. 1 系统架构图**（TikZ 矢量绘制：地面/无人机/卫星三类客户端 +
     边缘聚合器 + 成对上下行链路 + 各段 rank 徽章，补齐 WCNC 网络类论文 System Model 的标配图）；
     Fig. 2 调度流程图（原架构 TikZ）；Fig. 3/4 为重制的实验图。
   - 参考文献按引用密度收敛至 19 条（删除项均为期刊版中并列引用的次要条目）。
4. **图片修复**：期刊版复用的 `trainer_dl025.png` 系旧数据批次（y 轴 0.929–0.932，与表格 0.914 矛盾），
   会议版两图直接从 `all_metrics.csv` / `all_round_metrics.csv` 重生成，数值与表 I/表 II 逐一核对一致；
   仿真图由双面板改为单面板（0.25 s stress），图例文字按单栏宽度重新设计，保证可读。

## 重新编译

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## 投稿前待办（作者事项，无法代办）

- EDAS 注册：题目、作者列表须与 PDF 完全一致
- 若数据可用性声明需要，准备代码仓库 DOI（正文未包含该声明，期刊版有）
- 终稿阶段如需 7–8 页，需按页付费并经 IEEE Conference eXpress 提交
