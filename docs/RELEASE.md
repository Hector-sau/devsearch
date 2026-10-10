# Release procedure / 发布流程

## Version 1.1.0 / 版本 1.1.0

The owner authorized a public repository and final release on2026-10-10. Original DevSearch code/documentation uses Apache-2.0; upstream terms remain in [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md). Keep the existing `v1.0.0` tag and attachments unchanged. The new release integrates PR#6, including real R5 CPU serving and completed evaluation evidence.

所有者于2026-10-10授权公开仓库及正式发布。原创代码／文档采用 Apache-2.0，上游条款见第三方说明。保留既有 `v1.0.0` 标签与附件，新版本整合 PR#6 的真实 R5 CPU 服务及已完成评测证据。

The main experiment result is repeated validation improvement: on the same202-question v4 benchmark, R5+fixed rank fusion beats BM25 in seeds42/43/44; mean MRR0.941062 versus0.901416. The separate59-question evaluation did not reproduce the advantage. BM25 remains the formal application default; R5 is a selectable candidate requiring `--allow-candidate`. No additional GPU training or labeling is required for this release.

主实验成果为同一202题 v4 基准上三个种子重复获得验证提升；均值0.941062，高于 BM250.901416。另行59题评测未复现收益。应用默认保留 BM25，R5 以显式候选模式提供。本版无需新增 GPU 训练或标注。

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
3. Build the wheel from the frozen main source. Verify version1.1.0, installed metadata and included LICENSE/NOTICE/upstream notices, then check isolated baseline use. Verify the existing frozen real model can be loaded in the installed release; this is a serving check, not new training or a new quality result.
   从冻结主分支构建 wheel，核对1.1.0元数据与许可文件，并检查隔离基线运行；在安装后的版本核验既有冻结真实模型加载，属于服务检查，不是新训练或新效果成绩。
4. Attach source archive, wheel, reports/demo, compact CI receipts, release manifest and SHA256SUMS. The manifest records final source, version, applicable workflow IDs, asset names/bytes/hashes and historical experiment identities. It is a release attachment, not a source file containing its own future commit.
   发布源码、安装包、报告／演示、小型 CI 回执、发布清单及哈希。清单绑定最终源码、版本、工作流、附件及历史实验标识，作为发布附件保存，避免源码自指提交号。
5. Publish `v1.1.0`, then verify the tag resolves to frozen main, anonymous repository visibility, license detection and actual downloaded small assets. Store a matching release copy on the external drive.
   发布后核验标签指向冻结主分支、匿名访问、许可证识别及实际下载的小附件；外置盘保存同版副本。

No new paid API or hardware is part of publication. CPU fixture, sealed-input arithmetic replay, real inference and historical GPU measurements have separate evidence scopes. Full optimizer-checkpoint archival is tracked privately and is not implied by a successful public release.

发布不新增付费 API 或硬件。CPU 样例、封存输入算术重放、真实推理与历史 GPU 实测分别说明范围；完整优化器断点归档单独记录，不因公开发布成功而视为已完成。
