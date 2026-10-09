# Independent version-1 preparation audit / 第一版准备独立审核

**Outcome: the local release preparation has verified supporting evidence; final-source CI, main integration and publication remain pending.** No new training or GPU run is needed to resolve these delivery tasks. This audit did not merge, push, publish, select a project license or alter a historical experiment receipt.

**结论：本地发布准备有已核验证据支撑，最终源码 CI、主分支整合及发布仍待完成。**这些交付任务无需新增训练或 GPU 运行。本审核未合并、推送、发布、选择项目许可证或修改历史实验回执。

The independent auditor reviewed source/docs, inspected compact evidence, ran the documented miniature CPU example from a clean archive, checked the built wheel's contents and identities, and visually inspected five current demonstration images. Browser operation and package execution recorded by the other release executors are attributed to their records, not presented as an independent rerun here.

独立审核读取代码／文档、核对小型证据，从干净归档运行文档中的微型 CPU 示例，检查候选 wheel 内容与标识，并查看五张本次演示图片。其他发布执行者记录的浏览器操作和安装包执行仍归属其回执，本审核不冒充另一次独立执行。

## Scope and acceptance / 范围与验收

| Item / 项目 | Status / 状态 | Actual proof or next action / 实际证据或下一步 |
| --- | --- | --- |
| Existing release scope / 既有交付范围 | PASS / 通过 | Formal BM25 default, native PyTorch pipeline and scoped reports; R5 fusion remains experimental / 正式 BM25 默认、原生训练流程及注明范围的报告；R5 融合仍属实验 |
| Historical source ancestry / 历史源码祖先 | PASS / 通过 | `b23b0f4` includes both `cc0e7f0` and `e3168ca`; docs/evidence diff from `cc0e7f0` contains no runtime Python change / 文档基线包含两次实验源码祖先，文档差异未改变运行 Python |
| Current claims versus JSON / 当前说明与 JSON | PASS / 通过 | Formal test, adaptive validation and conditional stopping bounds are distinguished; numeric checks below / 三种证据明确区分，数字核验见下文 |
| Clean-archive CPU example / 干净归档 CPU 示例 | PASS / 通过 | All three commands exited 0, no stderr, using Python 3.11.16 and standard-library fixtures / 三条命令均退出 0，无错误输出 |
| Evidence-file byte identity / 证据字节标识 | PASS / 通过 | 10/10 registered compact files match byte count and SHA-256; six private source digests independently match / 十份小证据及六份原始来源哈希匹配 |
| Candidate package identity / 候选包标识 | PASS, provisional artifact / 通过，候选产物 | Built wheel metadata and imported version literal agree at 1.0.0; final frozen-source rebuild required / 元数据与导入版本一致，冻结后仍须重建 |
| Real-corpus API/components/images / 真实语料接口／组件／图片 | PASS for recorded demonstration / 已记录演示通过 | Five HTTP assertions, four component/browser cases, five images independently inspected; 8/8 capture identities and 5/5 serving-source hashes match / 五项 HTTP、四项组件／浏览器记录及五张图片核验；八份捕获及五份服务源码哈希匹配 |
| Upstream attribution / 上游来源 | PASS for pinned terms / 固定条款核验通过 | Exact tutorial BSD-3-Clause notice and pinned MiniLM MIT declaration verified; notices added / 核验固定教程 BSD 与模型 MIT，已补来源说明 |
| Final candidate CI / 最终候选 CI | PENDING / 待完成 | Requires frozen new SHA and fresh required workflow outcomes / 须绑定新冻结提交与检查结果 |
| Integrated main and version tag / 整合主分支与标签 | PENDING / 待完成 | Dependency PR stack and main acceptance have not been performed by this auditor / 本审核未执行依赖链整合或主分支验收 |
| Publication and durable private assets / 发布与持久私有资产 | PENDING / 待完成 | Final owner decision and final downloaded-asset check; large transfer stays deferred / 最终确认及实际附件复验，大文件继续暂缓 |

## Results and interpretation / 结果与解释

The current README, MODEL_CARD, DATA_CARD and RESULTS agree with the JSON on the following material claims:

当前首页、模型／数据说明及结果报告在以下关键声明上与 JSON 一致：

- Original formal test: 197 synthetic questions, BM25 MRR@10 `0.8925187333816775`, nDCG@10 `0.7908622299298411`; frozen encoder MRR `0.20242728224961726`; full tuning MRR `0.8734690999919427`, nDCG `0.7546418769125696`. Selection remains baseline; the +0.02 gate is preserved.
  首轮正式测试含 197 个合成问题，上述指标与报告的六位小数一致；默认仍为基线，原 +0.02 门槛保留。
- Expanded validation: 202 questions and 4,571 determinate pairs; provenance methods sum to 4,571. Full new primary review is 4,345; required B review is 435 random plus 1,017 targeted = 1,452. Seven random-review issues are below the 22-item expansion threshold. Six owner confirmations are human-assisted resolutions, not independently collected human labels.
  扩展验证的题数、标签数与来源总数一致，初审和必需复审数量完整；六条所有者确认准确记为人工辅助。
- R5 fused family: per-seed MRR `0.9340759075907591`, `0.944059405940594`, `0.945049504950495`, arithmetic mean `0.9410616061606161`, compared with BM25 `0.9014163916391639`. MRR difference is `0.03964521452145214`; the paired descriptive interval is `[0.01802805280528053, 0.06335258525852588]`. Pooled nDCG difference is `0.019024267867354457`, with interval `[-0.0015269927296803795, 0.03986842937145554]` including zero.
  R5 每种子、均值及配对区间相互一致，nDCG 区间包含零；反复使用的验证集不能包装成新增独立测试。
- The 25 historical systems have complete judged top-ten coverage; top-50 pair coverage is approximately 45.25%. Later six-run rankings introduce 316 unjudged top-ten pairs, so their complete point-quality results are not claimed.
  历史 25 系统前十标签完整，前五十约 45.25%；新对照仍有 316 对未判定，因此不宣称完整点指标。
- Six later runs are three seeds per loss, all fixed epoch 3 and 387 updates. The fused upper bound agrees with the exact fraction `631/763560` = `0.0008263921630258265`; the raw upper bound agrees with `-2035/152712` = `-0.013325737335638325`. The necessary +0.005 continuation condition cannot hold under the declared fixed-label assumptions. These remain bounds, not confidence intervals or completed quality metrics.
  六次运行数、种子与固定训练步骤一致；融合和原始上限与准确分数匹配，在既定标签假设下无法达到继续门槛。界限仍不等于置信区间或完整质量指标。

Source inspection of `src/devsearch/serving/pipeline.py` confirms raw-logit ordering via `order_by_score`; live RRF fusion is not implemented. The genuine demonstration is baseline-only with no model. Original fixed-workload GPU latency is not relabeled as R5 serving speed or a production SLA.

代码确认服务使用原始 logit 排序，未实现线上 RRF；真实演示为无模型基线。原固定负载 GPU 延迟没有改称 R5 服务速度或生产 SLA。

## Reproduction and artifact checks / 复现与资产核验

The clean archive was exported from `b23b0f48a035fce7f68c923affd15b5cc94964c9`, extracted outside the checkout and given a separate temporary runtime. No model, corpus or dependency download occurred. Executed:

干净归档来自上述提交，在 checkout 外解压并使用独立临时目录，未下载模型、语料或依赖，实际执行如下：

```text
python3.11 -m devsearch.index             -> exit 0, built miniature index
python3.11 -m devsearch.retrieve          -> exit 0, documented baseline query
python3.11 -m devsearch.validate_manifest -> exit 0, valid index
```

These are miniature fixture portability checks, not formal corpus performance. All 7 files in the original GPU evidence manifest and all 3 aggregate files in the project provenance manifest match registered sizes/hashes. The private v4 evaluation/release, R5 summary, grouped completion, label coverage and result-review source digests also match the provenance registry; private source paths and records are not published in this report.

这属于微型样例便携性检查，不是正式语料效果。原 GPU 清单七份文件和项目清单三份汇总的大小／哈希匹配；六份私有来源摘要也匹配，本文不发布私有路径或记录。

The provisional wheel inspected was `devsearch-1.0.0-py3-none-any.whl`, 296,155 bytes, SHA-256 `6c3b77c96b14bf9183a0795ef98bc0e42db410aec359b184215271bb6440742b`. Its 118 entries include no heavy model/checkpoint assets. `Name=devsearch`, `Version=1.0.0`, `Requires-Python=>=3.11` match source metadata and the imported version literal. [Packaging verification](release-packaging-verification-2026-10-09.md) records actual isolated install and seven rejection tests; the gate was read independently and checks metadata before install and imported identity after install.

已查看候选 wheel 的名称、大小、哈希及 118 项内容，不含重型资产；元数据和导入版本一致。安装包报告记录隔离安装与七项拒绝测试，独立阅读确认检查确实覆盖安装前元数据及安装后导入标识。

The candidate wheel embeds the README from its build snapshot. It must be rebuilt from the frozen release source; this provisional hash must not be reused for a changed wheel or final main build.

候选包嵌入构建快照中的首页，冻结源码后必须重建，不能把本候选哈希沿用到改变后的文件或最终主分支包。

Working-tree release changes were separately reviewed: `pyproject.toml` and `src/devsearch/__init__.py` agree at 1.0.0, the description identifies implemented behavior, and the wheel gate adds rejection of stale/mismatched package identities. The added seven tests include an actual offline install whose imported version is stale; their executed outcomes belong to the packaging executor's report. This audit did not rerun the same tests or a full suite unnecessarily. No training loop, retrieval algorithm, serving implementation or platform dependency lock was changed by those release edits.

另行审查了工作区发布修改：两处版本为 1.0.0，描述已反映实现，安装包门禁增加过期／不一致标识拒绝。七项新增测试包含实际离线安装旧导入版本，其执行结果归属打包执行者报告。本审核未重复同组测试或无必要重跑全量。发布修改未改变训练循环、检索算法、服务实现或平台依赖锁。

## Repository boundary scan / 仓库边界扫描

The baseline snapshot has 360 tracked files. All reachable local refs contain 1,253 Git objects and 777 blobs; the largest blob is 131,158 bytes. No blob is over 1 MiB, and no named blob uses a model/checkpoint/wheel/archive suffix. Pattern scanning found zero high-confidence credential tokens/private-key blocks, zero live SSH access details and zero JSON literal private-salt values. Original miniature fixtures are intentionally committed test data.

基线快照有 360 个跟踪文件；本地可达引用共 1,253 个对象、777 个 blob，最大 131,158 字节，无超过 1 MiB 或以模型／断点／wheel／归档命名的 blob。模式扫描未发现高置信凭据／私钥、真实 SSH 接入信息或 JSON 私有盐值；原创微型样例是有意提交的测试数据。

**Recorded exception:** three current historical evidence/report files contain operator local-path metadata: `reports/evidence/grouped-cpu-20261009.json`, `reports/execution-review-verification.md`, `reports/scaffold-verification.md`. Reachable history also contains this metadata in one older handoff; six historical blobs across four named files match. No secret value is reproduced here. Preserve immutable receipts; do not make a blanket claim that the entire repository/history is operator-path-free. Current compact project summaries and demo records omit those paths.

**已记录例外：**三份当前历史回执／报告仍带本机路径元数据；可达历史另有一份旧交接，共四个文件名中的六个 blob 匹配。本文不复述私密值。保留原回执，不宣称整个仓库／历史完全无操作路径；当前项目汇总和演示回执已省略该类路径。

The scan is an inventory/pattern check of the stated refs, not a new human audit of private runtime contents. The main executor must scan new release files/attachments again after freezing them.

扫描限于上述引用和模式，不构成对私有运行目录的新增人工审查；主执行者冻结新增发布文件／附件后须再核对其边界。

## Upstream terms / 上游条款

The [exact PyTorch Tutorials LICENSE](https://github.com/pytorch/tutorials/blob/11512db7cbbcc4b530ecfc63205d3fed13fa190d/LICENSE) is BSD-3-Clause; its full original notice is preserved in [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md). The 1,525-byte license text has SHA-256 `5703fcbe2d5dfcc48d486d26e4b9d2e48255c24602e1bd0270b60b8978116f86`.

准确教程 LICENSE 为 BSD 三条款，全文保留于第三方说明；大小与 SHA 如上。

The [pinned MiniLM README](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased/blob/44acabbec0ef496f6dbc93adadea57f376b7c0ec/README.md) explicitly declares MIT. The [pinned model API metadata](https://huggingface.co/api/models/microsoft/MiniLM-L12-H384-uncased/revision/44acabbec0ef496f6dbc93adadea57f376b7c0ec) confirms the revision and declaration and lists no separate LICENSE sibling. The model card's linked [Microsoft UniLM current LICENSE](https://github.com/microsoft/unilm/blob/master/LICENSE) is MIT with Microsoft Corporation attribution; it is a current supplemental source, not a claimed pinned UniLM commit.

固定模型 README 明确声明 MIT，准确版本元数据确认 SHA／声明且未列单独 LICENSE。其链接的 UniLM 当前许可证为 Microsoft Corporation 的 MIT，只作为当前补充来源，不编造固定 UniLM 提交。

Only small license/model metadata was fetched for this audit. Project-license selection and a change of repository visibility remain owner decisions. The notices do not authorize publishing private assets or imply upstream endorsement.

本审核仅获取小型许可证／模型元数据。项目许可证和仓库可见性由所有者决定，第三方说明不授权发布私有资产或暗示上游背书。

## Final integration gates / 最终整合门禁

The local docs baseline is a merge commit whose parents are `afcba198433991d504a9c8ebc3d886d6b0585420` and `cc0e7f0c6898af8fc7635c5ce9ffd4a4df8c4f89`. The cached `origin/main` is still the early scaffold commit `30927a4a6b89b53dd6c9f7a78000216970510258`; do not describe the candidate as already integrated. Verify live PR heads/bases before integrating #2 → #3 → #4 → preparation PR, preserving source ancestry and reviewing each retargeted diff.

文档基线的两个父提交如上，本地缓存主分支仍为早期骨架；不能称候选已合入。整合前核对实际 PR 头／目标，按依赖顺序执行，并保留实验祖先和复查重新设置后的差异。

The existing `scaffold-check.yml` is PR/manual only, and `non-gpu-evidence.yml` requires a manual dispatch or an allowed source-branch push with `[evidence]`. The cached main contains only the scaffold workflow. The release branch does not automatically acquire exact-lock/model evidence merely by being pushed. After integration, dispatch the required workflows at final main/ref and verify each `head_sha`, job success and zero-skip requirements; artifact retention is 90 days in the evidence upload configuration.

普通工作流当前由 PR／手动触发；无 GPU 证据需手动或允许分支带指定标记触发。本地缓存主分支仅有普通工作流，推送发布分支不能自动继承准确依赖／模型证据。整合后按最终源码触发并核对头 SHA、作业和零跳过，证据产物保留期为 90 天。

A prepared candidate and draft GitHub release can be constructed before main integration by naming the candidate SHA/ref and marking it unpublished. Before publication, rebind the release manifest and assets to the accepted final main SHA, rebuild the wheel, verify downloaded attachments and preserve essential compact receipts durably. Do not attach a final-main claim to a branch-only receipt, and do not place a self-referencing final source manifest into the source commit itself.

主分支整合前可以准备明确绑定候选 SHA／引用的未发布草稿；发布前再关联最终主分支、重建 wheel、核验实际附件并持久保存小型回执。不能把分支回执描述成主分支通过，也不将包含自身最终提交号的清单提交到自身源码。

The local screenshots reviewed show normal baseline controls, actual source content/button, empty results, invalid-filter rejection and explicit compare fallback. The refreshed demo record now attributes four real browser scenarios to the main Codex executor. Its capture manifest's eight file sizes/hashes match, and all five recorded serving-file hashes match the current source. Current README/TASKS/HANDOFF/RESULTS now mark the genuine BM25 UI as verified while retaining R5 live fusion as pending. The browser record verifies the pinned source URL displayed in the UI, not a separate external-site availability check. The source excerpts in these small screenshots retain the upstream attribution and verbatim BSD notice in THIRD_PARTY_NOTICES.

已查看截图覆盖正常基线、真实来源内容／按钮、空结果、非法筛选及明确对照降级。更新后的演示回执将四项实际浏览器操作归属于主 Codex 执行者；捕获清单八份大小／哈希及五份服务文件哈希均匹配。当前首页／任务／交接／结果已标明真实 BM25 界面通过，同时保留 R5 融合待完成。浏览器记录核验显示的固定来源链接，不冒充独立外站可用性测试；小型截图中的上游摘录由第三方说明保留来源与完整 BSD 通知。

New audit/demo textual files were scanned separately with the same secret/salt/operator-access patterns: zero matches. This final audit applies to the reviewed preparation working tree before its release commit; later final-source CI and frozen-wheel identities belong in external release assets, not retroactively in this pre-commit report.

新增审核／演示文本另以相同凭据／盐值／接入模式扫描，匹配数为零。本最终审核适用于发布提交前已审工作区；后续最终源码 CI 和冻结 wheel 标识应保存在外部发布附件，不能倒填成本文已完成的检查。
