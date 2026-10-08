# DevSearch cloud continuation / 云端继续授权

The human user explicitly updated the workspace authorization on 2026-10-08: “让它云端继续完成也可以，后续下载回来，它 不可以都在 GitHub 完成吗”. Claude is authorized to continue this DevSearch project in its `Pytroch` cloud runtime. This supersedes only the earlier local/GitHub-only workspace restriction in the fixed handoff document. The frozen data/review protocol and other scope limits remain in force.

用户于 2026-10-08 明确更新授权：允许 Claude 在 Pytroch 云端继续 DevSearch，后续下载成果。这替代固定交接文档中的“仅本地／GitHub”地点限制；冻结协议和其他范围限制继续有效。

Claude is the executor for the remaining non-GPU work. GitHub is the durable source/artifact location: code/docs in the private repository; private checkpoints, judgments, salts and dataset packages as private Release assets, outside Git history. Cloud files may be temporary working copies. Preserve a downloadable checkpoint after each small wave and before quota/session stops. If the session policy prevents Release uploads, attach the checkpoint, SHA-256, manifest and restore guide for Codex to transfer through the user's existing GitHub access; never bypass the policy or leave the only copy in the container.

Claude 是剩余无需 GPU 工作的执行者。GitHub 保存长期成果：代码文档进私有仓库，私有检查点、判定、盐值和数据包放私有 Release，不进 Git 历史。云端允许临时工作副本；每轮少量批次及额度／会话停止前保存可下载检查点。若会话政策禁止 Release 上传，则附上归档、SHA-256、清单和恢复说明，由 Codex 通过已有用户权限上传，不绕过限制。

Resume from cp01 + cp02 in Release `406041484`, tag `devsearch-runtime-checkpoint-20261007`. Code `df4445498361b3d67487192f56d0768e17cec647`. Primary complete 9,852; second sample 924 with 120 accepted (`S1-0001`, `S1-0009`, `S1-0017`), 804 remaining; at least 29 adjudications; final audit incomplete. Verify actual hashes and counts before dispatch; do not reuse the old cloud runtime's earlier 2,760-item counts as the new baseline. Do not repeat accepted judgments.

从上述 Release 的 cp01＋cp02 恢复；核验准确代码、哈希及数量。首轮 9,852 全部完成；复审 924 中完成 120，余 804；裁决至少 29。不要用旧云目录的 2,760 数量作为新起点，不重复已接受判定。

Complete: remaining independent second review → protocol-required expansion (>10% unrounded stratum disagreement) → independent adjudication → audit exit0/all9,852 final → audited v2 package → fresh retrieval/rebuild/seal verification → real-model CPU crash/resume and installed-wheel/locked portability checks → final exact model/v2 pins, corrected acceptance/docs, relevant tests and zero-skip ML CI on the final code/config commit → reproducible GPU handoff commands and private backups. Small CPU diagnostic training is authorized; formal GPU campaign and premature formal test scoring are not part of this continuation.

完成顺序：余下独立复审、协议扩审、独立裁决、完整审计、v2 打包、新目录恢复及封存核验、真实模型 CPU 故障恢复与 wheel／锁定迁移、最终固定配置及 CI、GPU 指令和私有备份。允许小型 CPU 诊断训练，不提前跑正式测试评分或正式 GPU 实验。

Use existing included session quota only. Preserve actual agent/model/rubric provenance, keep uncertain usage unknown, and report agent review as agent review. No paid API purchase, extra-usage activation, GPU rental, PR merge or public publication. At quota exhaustion, checkpoint and report the exact remaining work; no extra spending. The prior detailed handoff remains the technical reference, with this workspace authorization taking precedence.

只使用现有会话包含的额度；如实记录来源，不虚构用量或真人审查。不购买 API、不开额外付费用量、不租 GPU、不合并 PR、不公开发布。额度耗尽时保存并报告剩余工作；旧交接技术步骤继续适用，地点授权以本文为准。
