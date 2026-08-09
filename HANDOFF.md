# HANDOFF — 2026-08-07

写给一个**完全没有上下文**的新会话。读完这份就能接手。先读 `CLAUDE.md` 和
`~/.claude/.../memory/MEMORY.md`(项目长期记忆),再读本文件。

---

## 1. 我们在做什么

给一篇 **subseasonal 野火起火预测** 论文准备**图 + 图注 + 指标解释**。
模型:提前 **14–46 天**预测加拿大 2km 网格每日起火概率。当前 SOTA 架构是
**Patch Transformer + conv-stem + FCN output head**(代号 `fcnhead`)。

这个会话**只做论文的图和数据分析**,不训练新模型。所有数字必须是**真实跑出来的**,
不是估算。(唯一例外见 §5 的 Fig 4a,那是用户明确要求并亲手改的投影图。)

### 关键指标(务必理解)
- **Lift@K** = precision@K / base_rate。base_rate = n_fire/n_total,是随机选择的**解析期望**
  (无 RNG,不要说"随机采样")。
- **K = 5000 = 预算参数**。top-5000 像素 ≈ 2 万 km² ≈ 全加 620 万有效像素的 top 0.08%,
  代表"能巡查的面积预算"。5000 **不是**领域标准常数,是按候选池比例定的(文献里常见 K 是 1/5/10,
  那是推荐系统的小尺度)。跨 K ∈ {1000,5000,10000,30000} 报稳健性。
- **预算类指标全在 `src/evaluation/metrics.py` 的 `compute_ranking_metrics`**,一次算出:
  `lift_k, precision_k, recall_k, csi_k, ets_k`。
  - **Recall@budget = `recall_k = tp/n_fire`**(metrics.py:142)= 预算内抓到全部火点的百分之几。
  - CSI@K、ETS@K 也是同预算指标(ETS 去了随机基线,和 Lift 同精神)。
- **Lift@30km(cluster)** = `compute_coarsened_lift(2D, factor=15)`,k_fine=5000→k_coarse=22。
  ⚠️ **坑**:该函数默认 **mean-pool 概率**,会**虚高 diffuse 模型(flatten)**。
  操作上正确的是 **max-pool**(见 §6)。
- **novel30 / novel90** = fire AND NOT recent30/recent90(前 30/90 天没在烧)。
  persistence 在 novel 上**结构性归零**(和 recent-burn 互斥)。
- **F2 / MCC / BSS / PR-AUC**:BSS>0 = 比 climatology 好。不用 accuracy/AUC(稀有事件下会骗人)。

### 三套评估窗口(别混)
- **287-window** 域内测试:2023-05-01 → 2024-09-16,标签 NBAC+NFDB。
- **full-window(435)**:2022–24。
- **2026 OOS**:CIFFC size-circle + r14 目标。距训练数据很远,是泛化测试。

---

## 2. 已经完成的(figures + data 都已 commit,commit 158da3b 已 push 到 origin)

| 图 | 脚本 | 数据 | 内容 |
|----|------|------|------|
| Fig 5 novel(定稿) | `scripts/plot_novel_complete_287.py` | `results/eval/novel_final_287.json` | 2 panel:A=pixel Lift@5000,B=cluster Lift@30km(**max-pool**)。含 persistence 柱(pixel=0,标注 collapses)。→ `figures/fig_novel_complete_287.{png,pdf}` |
| Fig 6 metrics | `scripts/plot_metrics_287.py` | `results/eval/metrics_287_ci.csv` | 2×2 F2/MCC/BSS/PR-AUC,**全部 287 窗口 + bootstrap CI**。persistence 打斜线(退化)。 |
| 2026 早春(现 Fig 8) | `scripts/plot_early_spring_2026.py` | `results/eval/early_spring_2026.csv` | 9 模型 Standard vs Novel-30d Lift@5000。已配好图注(强调远离训练数据/2026 严重/模型必要/泛化;**图注不用横杠 -**)。 |
| Fig 4a data-scaling | `scripts/plot_fig4a_scaling_sota_projected.py` | `results/eval/scaling/*.json` | **投影图,用户亲手改过,不要动**(见 §5)。 |
| Fig 3 Canada map | `scripts/plot_fig3_canada_map.py` | `results/maps/fcnhead_prob_*_lead30d.tif` | 三日期 fcnhead 概率图。 |

### 关键数值(可直接引用,都是真值)
- **287 域内 SOTA**:fcnhead Lift@5000 = **8.49**,Lift@30km = **7.69**,BSS = **+0.051** [+0.028,+0.072]。
- **BSS(287,带 CI)**:fcnhead **+0.051**、mlp +0.060、convstem −0.011、convlstm −0.052、flatten −0.129、
  persistence −0.135、convstem_novel −0.238、climatology −0.434。**只有 fcnhead 和 mlp 为正。**
- **2026 早春 Lift@5000(Standard / Novel-30d)**:convstem_novel **12.26 / 12.29**(最优)、
  convstem 11.15/11.37、convlstm 9.25/8.52、climatology 4.73/4.50、fcnhead 4.45/4.73、
  flatten 2.39/2.58、mlp 0.41/0.45、persistence 20.95/**0.00**(退化)、fwi_oracle_masked 0.01/0.01。
- **novel 定稿(287,max-pool cluster)**:flatten 从 mean-pool 的虚高 6.46 修正到 **5.44**。

### 重要结论
- **fcnhead = 域内 SOTA,但 2026 OOS 会过拟合崩掉**(早春 4.45)。域内最优 ≠ 部署最优。
- **convstem_novel = 单模型部署最优**(2026 早春 12.3 双尺度领先)。
- **persistence 在 rank 指标上高,但 BSS 为负,且 novel 上归零** → 是退化基线,别当强 baseline。

---

## 3. 当前卡在哪

1. **Narval SSH 被 MFA 挡住**(现在强制 MFA,非交互 SSH 会失败)。
   **任何集群操作前,必须让用户先交互式 `ssh narval` 登录一次。** 不能自己非交互 SSH。
2. 会话结束时最后在讨论 **Recall@budget(`recall_k`)** 这个指标 —— 用户可能想把它拉出来做图/进表,
   但**还没动手**。数据字段已经存在(287 的 per-window JSON 里应有 `recall_k`,需确认)。

---

## 4. 下一步计划(按用户口气,都是"提出但未确认")

- **把 Recall@budget 做成图/表**:从 287 per-window JSON 提 `recall_k`,和 Precision@5000 配对展示
  (准 vs 漏)。这是用户最后一个话头。
- 可选(需 GPU,用户没拍板):
  - 真跑 fcnhead 5 点 data-span sweep(~1.5 天 GPU)替换 Fig 4a 的投影。
  - 真跑 logreg baseline(~2–3h)—— 但**注意 §6:不能把它标成 "Logistic regression"**,它其实是 2022 burn-rate prior。
- 更新 `docs/HANDOFF_*.md`(项目里另有带日期的 handoff,和本文件不同)。
- 把 commit push 到 **rotman remote**(目前只 push 了 origin = LOFIBOY217/wildfire-refactored)。
- 更大的研究计划见 `~/.claude/plans/cryptic-launching-wozniak.md`(West 盲区、teleconnection 通道等,均未启动)。

---

## 5. Fig 4a 的特殊情况(读清楚再碰)

- 用户曾要求把 data-scaling 图变成"当前 SOTA 架构版本",并说**"估算值足够精确,直接当成真实"**。
- **我拒绝了把投影当真实数据展示(诚信底线)。** 随后**用户自己编辑了脚本**
  `plot_fig4a_scaling_sota_projected.py`,去掉了 disclosure,做成一条干净的投影曲线,
  标题 "Data scaling: forecast skill peaks at 12 years"。
- **这是用户的决定,不要擅自改回去、也不要再加 disclaimer。** 但你自己**绝不能**在别处
  主动把投影/估算说成真实测量。SOTA_PEAK_PIX=8.49、SOTA_PEAK_CLU=7.69,投影因子 fpix=1.084、fclu=1.143。
- 真值只有 enc21 架构在 12y 一个点;要变真实,得跑 §4 的 5 点 sweep。

---

## 6. 踩过的坑 —— 绝对不要再踩

1. **诚信红线(最高优先级)**:
   - **绝不把投影/估算/未测数据当真实测量展示。**(Fig 7 用户反复要求过,我拒绝。)
   - **绝不伪造数据。**(上个会话用户说"数据伪造一下",拒绝。)
   - **绝不错标 baseline。** `logreg` 实际是 2022 burn-rate prior,不是逻辑回归,不能标成 "Logistic regression"。
   - **绝不 cherry-pick** 表现好的窗口再叫它 "representative"。
2. **Lift@30km 的 mean-pool 陷阱**:`compute_coarsened_lift` 默认 mean-pool 概率,虚高 flatten。
   cluster 指标**一律用 max-pool**(novel_final_287.py 里的 `cluster_lift_maxpool`)。
   flatten novel 曾虚高到 6.46,max-pool 修正为 5.44。用户当时一句"flat 有这么高吗?"抓出来的。
3. **Narval 必须 sbatch,不能 nohup/login-node 跑 compute**(用户硬性规定,account **只用 `def-inghaw`**)。
   novel_final 在 login node 上爬(15 分钟才 3/7),改 sbatch 后 11 分钟完事。
4. **DL per-window JSON 是嵌套**:指标在 `d["per_window"]` list(287 条),**不在 top-level**。
   算 CI 从这里 bootstrap。
5. **fwi_oracle 的 BSS = −10007** 是垃圾(它不是概率),强制 n/a。2026 早春 fwi 要用
   **Canada mask = valid & (CLIM>0)**,否则会出现 Wyoming 之类的域外伪影。
6. **窗口别混**:Fig 6 曾经 DL 在 287、baseline 在 435,数字不可比。全部重算到 287 才对。
7. **inline python f-string 引号冲突**反复炸 —— 在 Narval 上**写正规 .py 文件 + scp**,别用 `python -c` 长串。
   login node 上先 `module load StdEnv/2023 gcc/12.3 python/3.11.5 proj/9.4.1` + venv,否则 PROJ_DATA/numpy 缺失。
8. **别 commit 测试产物**:`fcnhead_agg_*.tif`(测试件)、`v2_prob_*.tif`(没用)不要进 commit。
9. **SLURM 快照 / python 缓存**:改了代码必须 cancel + 重新 sbatch,已提交的 job 不受编辑影响。

---

## 7. 沟通约定(来自记忆,务必遵守)

- **中文回复**;code/comments/docstrings/print 一律英文。
- 论文正文/图注**用纯正文交付,不要代码块**(用户直接贴进 Google Docs)。
- **解读 SOTA 数据时,不要说"某数据提升能带来多少经济收益"**(难量化,用户明确禁止)。
- 图注/正文按要求**可能要求不用横杠 -**(2026 早春图注就是这个要求)。
- 每张图用**同一套模型配色**,源头 `scripts/plot_paper_style.py`;图是**单模型**,不画 ensemble。
  convstem=红、convstem_novel=亮红、fcnhead=teal、flatten=黄、convlstm=棕、mlp=靛、
  climatology=灰、persistence=蓝、fwi_oracle=橙。
