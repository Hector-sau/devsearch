# R5 integration evidence / R5 接入证据

**Historical preparation snapshot, superseded for current status by the [2026-10-10 completed CPU/holdout report](r5-independent-holdout-2026-10-10.md).** Access was restored, real weights and selected assets were verified on the external drive, review/scoring completed, and quality gates failed. The dated preparation evidence below is preserved.

**本页为历史准备快照，当前状态以 [2026-10-10 完成报告](r5-independent-holdout-2026-10-10.md)为准。**访问已恢复，真实权重及选定资产已在外置盘核验，审核／计分完成，但效果门槛未通过。下方带日期准备证据保留。

Base source: published private v1.0.0 `e0ba2c4a2856a43d41d0e69ddf1a1f5f7c360955`. This change does not modify the tag, original training, test seals, evaluated BM25 default or historical metrics. Final implementation source and CI are the attached PR's head, not this base identity.

基准为上述已发布私有第一版，不修改标签、原训练、测试封存、正式 BM25 默认或历史成绩。本次实现源码及 CI 以关联 PR 当前提交为准，不归到基准提交。

- Shared rank fusion matches **606/606** historical query×seed full rankings with **zero mismatches**, using recorded raw-ranking permutations to reconstruct ordinal logits. Cache SHA-256 `5e28564770f93de98f1752a94845088714d41a6cd72f1ebc245e5163078c84ba`. This verifies rank math and tie handling; no R5 weights were loaded and no new quality claim follows.
  共用融合与历史 606 组完整排名全部一致；使用原排名构造序数分数，只验证公式及同分处理，没有加载 R5 权重，不代表新增效果。
- Independent code review found a fusion-smoke gate defect: a degraded query could record no rerank and later pass a baseline-only replay. The fix refuses degraded/non-neural/all-empty fused smoke and rejects absent fused replay evidence. Regression tests cover it. Neural logits remain logits; fusion scores are displayed separately.
  独立代码审核发现融合短测可通过缺失重排而误接受的问题；已拒绝降级、缺失神经结果及全空短测，并加入回归测试。神经分数及融合分数分别保留。
- A separate data audit found **65** eligible never-selected test-owned groups, after excluding all saved historical generation selections across old splits and original test question evidence. Exact/near-duplicate and source-span audits found zero violations. A seeded sample60 produced **59** accepted agent-generated questions and one exclusion before any retrieval. All59 received independent question/positive-evidence review, and exact quotes/token budgets were mechanically checked.
  独立数据审核排除所有历史生成选择和旧测试证据后，找到 65 个可用未使用组；重复及位置审核零冲突。固定抽样 60 组，形成 59 条独立 Agent 审核问题，另有一组在检索前排除；原文依据及 token 预算已检查。
- Protocol SHA-256: `cb27eccffb91e27f2e4ade2e45dbe649b2cfe26088efa0863bc58521120da6f2`; questions SHA-256: `098bf6a9e624038b816fb3aef71a1051aba302fbc4f2a509366fc78cce03df21`. Sealed inputs, full passages and review records are private runtime data outside Git. Query max29tokens, source-positive pair max216 under limits64/256.
  协议及问题哈希如上；封存输入、完整段落和审核记录留在私有运行目录，不进 Git。问题／组合长度符合既有预算。
- A follow-up independent review found two acceptance gaps: questions and their seal could change together under an unchanged plan, and a loaded corpus/tokenizer/text policy was not bound. Fixed input-binding SHA-256 `da8ecc36109b820065ec0d4eb19c8a40cf090fb98ba56e7c4189d7121783342d` now anchors the question/seal hashes and input identities. Before any request the runner checks the actual inner checkpoint, corpus/index, tokenizer, limits, CPU device and inference batch. Regression cases cover combined question/seal replacement and same-checkpoint exports with different inputs.
  后续独立审核发现问题与封存同时改写、实际语料／分词器／长度未绑定两处缺口。已用上述固定输入绑定哈希修复，请求前核对真实内部断点、语料／索引、分词器、限制、CPU 和批次，并加入回归用例。
- Independent re-review confirmed both gaps closed with no remaining actionable P1/P2 findings in that scope. The bound private archive contains13 input/evidence files plus its checksum list,120,599bytes, SHA-256 `714762e57819c03e8b2fc6d3b2b6cc6d5ad57c4d975efa9465a42dbde36da93b`. Every archived file was reread and checked. Earlier archive and evidence files are preserved.
  独立复审确认两处缺口均关闭，所审范围未发现剩余 P1/P2。新的私有包含 13 个输入／证据文件及校验表，120,599 字节，哈希如上；逐文件重新读取验 hash。旧包及证据均保留。
- CPU timing receipts now record host-visible model/cores/RAM and process peak RSS. A narrow review identified that cgroup mount-root values are not necessarily the process quota; those fields are explicitly named as mount-root values, with effective container limits unknown. This avoids claiming an allocation from host-visible resources.
  CPU 测速回执已记录可见型号／核心／内存及进程峰值 RSS。专项审核发现 cgroup 挂载根目录值不一定是进程配额；字段已明确命名为根目录值，实际容器配额记为未知，避免把宿主可见资源当作实例分配。
- SSH at the last supplied endpoint refused connection; R5 seed42 epoch3 is remote-only. No new GPU, paid API, weight download or training took place. Real-model reload, API parity, pooled top10 labels, independent quality scoring and CPU latency are **pending**, not passes.
  最近提供的 SSH 端点拒绝连接；R5 权重仅在远端。没有新 GPU、付费 API、权重下载或训练。真实重载、接口一致性、汇总前十标签、独立效果及 CPU 延迟均待完成，未标通过。

Local full suite after the binding/hardware additions: `PYTHONPATH=src:tests <bundled Python3.12> -m unittest discover -s tests -q` ran936tests with0failures/errors and331explicit skips for unavailable optional ML/serving dependencies. Loopback permissions were enabled for the socket checks. After the final quota-field clarification, targeted fusion/pipeline/release/ranking/acceptance tests ran68 with0failures/errors and11ML/serving skips. CI must supply the optional dependencies and pass with zero skips; local skips do not prove those paths.

绑定及硬件补充后的本地全量 936 项，零失败／错误，因缺少可选依赖明确跳过 331 项；socket 检查启用了本机 loopback 权限。最后配额字段说明修正后的专项 68 项，零失败／错误、11 项跳过。CI 须补齐依赖并实现零跳过，本地跳过不代表对应路径通过。

Follow [R5 integration](../docs/R5_INTEGRATION.md). The original197 test is historical exposed evidence; the new holdout measures new synthetic questions/evidencegroups within the same document snapshot and shared index. It is not a real-user study or unseen-repository benchmark.

继续步骤见接入说明。原 197 题属于已暴露历史证据；新留出集衡量同一文档快照及索引中的新合成问题／证据组，不代表真人研究或新仓库基准。
