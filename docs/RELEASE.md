# Release procedure / 发布流程

## Version 1.2.0 preparation — 2026-10-13 / 版本1.2.0准备

The owner authorized the final 966 configuration and GitHub update. Publish an immutable new release after final-source verification; keep existing tags and receipts unchanged. The preferred owner demo remains the previously accepted966 seed 42 endpoint with fixed fusion0.5/k60, with BM25 comparison/fallback and explicit candidate activation. The public small CPU example must run from a clone without private model or label assets. [Final development results](../reports/final966-replication-2026-10-13.md).

所有者授权最终966配置及GitHub更新；最终源码核验后发布新的不可变版本，保留原标签和回执。首选所有者演示继续使用已验收966 seed 42模型及固定融合0.5／60，保留BM25对照／降级和显式候选启用；公开小型CPU样例须能从克隆运行，不依赖私有模型或标签。

1. Review the integrated source and aggregate-only documentation; record the resulting exact commit. Keep private raw questions, labels, review state, credentials, runtime paths and model weights outside Git and release assets.
   审阅整合源码及仅含汇总的文档，记录准确提交；私有问题、标签、审查状态、凭据、运行路径及权重不进入Git或公开附件。
2. Run applicable CI and isolated wheel checks on that final source with version 1.2.0 and zero skipped required ML/lock checks. Verify package metadata, LICENSE/NOTICE/upstream notices and the public model-free quick start. Record actual results; do not reuse prior-head CI as current verification.
   在最终源码执行所需CI及1.2.0隔离安装检查，所需ML／依赖检查零跳过；核对元数据、许可及公开无模型样例，记录真实结果，不复用旧提交CI冒充新验收。
3. Verify candidate loading, fixed fusion and BM25 failure fallback in the installed final version with owner-held assets. Preserve the old80-check/36-audit CPU receipt as `1.1.1.dev0` evidence. Any final-version service check gets its own source/version receipt; no historical latency or GPU quality number changes identity.
   使用所有者资产在最终安装版核验候选加载、固定融合及BM25故障降级；原80项／36项审核回执保留1.1.1.dev0身份，新版本服务检查单独记录源码／版本，不改写历史延迟或GPU成绩。
4. Build source and wheel attachments plus public reports, aggregate evidence, compact exact-source CI/package receipts and SHA256SUMS. The release manifest binds final commit/version, asset bytes/hashes and historical experiment-plan identities. Avoid a self-referential source manifest containing its own future commit.
   构建源码、wheel、公开报告、汇总证据、小型准确源码回执及哈希；发布清单关联最终提交／版本、附件字节／哈希及历史方案身份，不制作包含自身未来提交的源码清单。
5. Publish `v1.2.0`, verify its tag resolves to the accepted source and actual small downloads match the manifest, and preserve the matching copy on authorized external storage. Mark publication complete only after these checks.
   发布v1.2.0后，核验标签指向通过验收的源码、实际小附件下载与清单一致，并在授权外置盘保存同版副本；完成上述检查后才标记发布完成。

Three-seed 966 GPU quality evidence belongs to its frozen private execution plan, not automatically to the publication commit. Exact new training reproduction requires owner-held frozen wrappers and inputs absent from the public clone. The one-seed 1, 932 comparison failed its declared duration rule and is retained as a controlled negative result. Historical197/59 failures remain visible. No further GPU training, new rental or paid API is part of this publication.

三种子966 GPU效果证据归属冻结私有执行方案，不自动归属发布提交；新训练精确复现需要公开克隆未附的所有者冻结包装器及输入。单一种子1, 932对照未通过时长规则，作为受控负面结果保留；历史197／59题失败仍可查看。发布不包含继续GPU训练、新租机或付费API。

Private no-card checkpoint backup runs separately. Public release success does not prove full optimizer-checkpoint archival, and the instance must remain preserved until all required tensors have complete-file hashes matched to the frozen inventory and independent acceptance.

无卡模式私有断点备份单独执行。公开发布成功不证明优化器断点完整归档；所有必要张量完成整文件哈希与冻结清单匹配并通过独立验收前，必须保留实例。

## Historical version 1.1.0 procedure / 历史版本1.1.0流程

The owner authorized a public repository and final release on2026-10-10. Original DevSearch code/documentation uses Apache-2.0; upstream terms remain in [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md). Keep the existing `v1.0.0` tag and attachments unchanged. The new release integrates PR#6, including real R5 CPU serving and completed evaluation evidence.

所有者于2026-10-10授权公开仓库及正式发布。原创代码／文档采用 Apache-2.0，上游条款见第三方说明。保留既有 `v1.0.0` 标签与附件，新版本整合 PR#6 的真实 R5 CPU 服务及已完成评测证据。

The main experiment result is repeated validation improvement: on the same202-question v4 benchmark, R5+fixed rank fusion beats BM25 in seeds 42/43/44; mean MRR0.941062 versus 0.901416. The separate59-question evaluation did not reproduce the advantage. BM25 remains the formal application default; R5 is a selectable candidate requiring `--allow-candidate`. No additional GPU training or labeling is required for this release.

主实验成果为同一202题 v4 基准上三个种子重复获得验证提升；均值0.941062，高于 BM25 0.901416。另行59题评测未复现收益。应用默认保留 BM25，R5 以显式候选模式提供。本版无需新增 GPU 训练或标注。

## Public preparation / 公开准备

1. Preserve and verify private datasets, review checkpoints and original release metadata outside the repository. Retire the old private runtime draft from the public project's release inventory only after every original asset is backed up and hash-verified.
   先在仓库外保存并核验私有数据、审查备份及原发布元数据；全部附件核验后，才从公开项目发布列表移除旧私有运行草稿。
2. Inspect Git history/refs, current source, discussion text and release assets for credentials and private review state. Synthetic negative-test strings remain test fixtures; owner paths alone are not credentials. Preserve applicable upstream notices.
   检查历史、引用、当前源码、讨论及附件中的凭据和私有审查状态。负面测试字符串保持样例身份，所有者路径本身不算凭据；保留上游说明。
3. Finalize README, architecture, scoped metrics, data/model cards, CPU example, validation-score replay and real-candidate instructions. Public source does not contain sealed private labels or model weights; full historical reproduction requires owner-provided assets.
   定稿首页、架构、注明范围的成绩、数据／模型说明、CPU 样例、验证成绩重放及真实候选操作。公开源码不包含封存私有标签和权重，完整历史复现需要所有者提供产物。

## Final acceptance and publication / 最终验收与发布

1. Review PR#6's final diff, require applicable checks on its final head, mark it ready and integrate into `main`. Freeze the resulting main SHA; do not move the published historical tag.
   审阅 PR#6 最终差异及对应检查，转为可审阅状态并整合 `main`，冻结主分支 SHA，不移动历史标签。
2. Run `scaffold-check.yml` and `non-gpu-evidence.yml` at that exact final ref. Check each run's `head_sha`; all required jobs succeed, ML and exact-lock suites have zero skips. Existing GPU/model-quality receipts keep their executed source identities.
   在准确最终引用运行两工作流，核对 `head_sha`，所需作业均通过，ML／准确依赖测试零跳过；历史 GPU／效果回执保留实际执行源码归属。
3. Build the wheel from the frozen main source. Verify version 1.1.0, installed metadata and included LICENSE/NOTICE/upstream notices, then check isolated baseline use. Verify the existing frozen real model can be loaded in the installed release; this is a serving check, not new training or a new quality result.
   从冻结主分支构建 wheel，核对1.1.0元数据与许可文件，并检查隔离基线运行；在安装后的版本核验既有冻结真实模型加载，属于服务检查，不是新训练或新效果成绩。
4. Attach source archive, wheel, reports/demo, compact CI receipts, release manifest and SHA256SUMS. The manifest records final source, version, applicable workflow IDs, asset names/bytes/hashes and historical experiment identities. It is a release attachment, not a source file containing its own future commit.
   发布源码、安装包、报告／演示、小型 CI 回执、发布清单及哈希。清单绑定最终源码、版本、工作流、附件及历史实验标识，作为发布附件保存，避免源码自指提交号。
5. Publish `v1.1.0`, then verify the tag resolves to frozen main, anonymous repository visibility, license detection and actual downloaded small assets. Store a matching release copy on the external drive.
   发布后核验标签指向冻结主分支、匿名访问、许可证识别及实际下载的小附件；外置盘保存同版副本。

No new paid API or hardware is part of publication. CPU fixture, sealed-input arithmetic replay, real inference and historical GPU measurements have separate evidence scopes. Full optimizer-checkpoint archival is tracked privately and is not implied by a successful public release.

发布不新增付费 API 或硬件。CPU 样例、封存输入算术重放、真实推理与历史 GPU 实测分别说明范围；完整优化器断点归档单独记录，不因公开发布成功而视为已完成。
