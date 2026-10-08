# DevSearch non-GPU handoff / 无需 GPU 工作交接

Updated 2026-10-08, Asia/Kuala_Lumpur. The user stopped further judging to save tokens and will assign another AI. All Codex judging agents have stopped at completed-batch checkpoints.

2026-10-08 更新。用户为节省 token 改为交接给另一个 AI；本轮所有审查 agent 已在完整批次检查点停止。

**Mandatory workspace rule: edit and preserve work only on the user's local machine or in the user-accessible private GitHub repository. Do not edit in the AI provider's cloud environment, and do not leave the only copy of judgments, runtime state or artifacts there.** GitHub Actions in this repository can provide visible CPU checks and saved artifacts within existing limits; do not enable paid services. Preserve the current Mac's storage preference: no large ML environment/model downloads on its internal disk without an authorized mounted external runtime.

**强制工作地点：只在用户本地或用户／协调者可访问的私有 GitHub 工作与保存。禁止在 AI 服务商自己的云端环境编辑，也不能把判定、运行状态或产物的唯一副本留在那里。** 本仓库 GitHub Actions 可在现有限额内用于可查看的 CPU 检查及产物，不开启付费服务；当前 Mac 不安装大环境或下载权重，除非已有获授权、实际挂载的外置运行盘。

## 1. Exact status / 准确状态

Repository: `https://github.com/Hector-sau/devsearch` (**private**). Implementation branch `claude/youthful-turing-uc1lkv`, verified code `df4445498361b3d67487192f56d0768e17cec647`. PR #2 is open and **unmerged**. CI on that code passed all four jobs: `https://github.com/Hector-sau/devsearch/actions/runs/37592058531`. Do not assume the default `main` has this implementation. This handoff is also saved on `codex/non-gpu-handoff-20261008`; that branch adds documentation to the same code snapshot.

私有仓库、实现分支及准确提交如上；PR #2 未合并，既有 CI 四项通过。不要假定默认 main 已含全部实现；交接分支只在上述代码上增加文档。

| Gate / 条件 | Verified status / 已核验状态 |
|---|---|
| Primary review / 首轮 | **9,852/9,852**, 9,823 resolved, **29 unsure**, no missing primary / 全部已导入，29 未决，无缺失首轮 |
| Second sample / 抽样复审 | Frozen sample **924**; **120 imported**, **804 remaining** / 冻结样本 924，已导入 120，余 804 |
| Accepted second batches / 已接受复审批次 | **S1-0001, S1-0009, S1-0017**, each 40 / 每批 40，其余 21 批未完成 |
| Expansion / 扩审 | Not decided until sample complete / 抽样完整后由协议决定 |
| Adjudication / 裁决 | **At least 29**; disputes/expansion may add more / 至少 29，争议或扩审可能增加 |
| Final bank / 最终判定库 | **0 final**, audit exit **3** expected; not sealed / 最终为零，按预期退出 3，未封存 |
| Real MiniLM / 真实模型 | Pinned weights verified remotely; **10/10 CPU checks**, export/reload parity passed; receipts now local and in cp02 / 真实权重及 CPU 检查、重载通过，回执已迁出原云环境 |
| Real-model crash/resume and installed wheel / 真实模型恢复及 wheel | Still pending / 尚未完成 |
| Audited v2 and final GPU pins / 审计 v2 及 GPU 固定配置 | Still pending; config uses v1 + resolve-main-once / 尚未完成，现配置仍旧 |
| GPU / GPU | Not rented or run; no ranking-quality result / 未租未跑，无真实排序训练成绩 |

Never repeat the accepted primary pass or the three accepted S1 batches. Primary provenance is mixed Claude/Codex, all **agent judgments**, never human. A-0070–0079 were recovered via checksums/schema/actual import; original Claude provenance retained, usage **unknown**. Codex per-task usage is unavailable and must not be invented.

不要重做已接受首轮或这三批复审。首轮来源混合 Claude／Codex，均为 Agent 判定，不能描述成人工。残留十批经校验恢复，用量未知；Codex 逐任务用量无法取得，不能编造。

## 2. Recoverable artifacts / 可恢复资料

Private draft Release ID **406041484**, tag `devsearch-runtime-checkpoint-20261007`: `https://github.com/Hector-sau/devsearch/releases/tag/untagged-56b4a184a6ea848cd500`.

资料保存在上述私有 draft Release，原 cp01 和新增 cp02 都保留，私有判定与盐值没有提交到 Git。

| Asset / 资产 | ID | Bytes / 大小 | SHA-256 |
|---|---:|---:|---|
| `devsearch-runtime-checkpoint-20261007-cp01.tar.gz` | 619452486 | 10,355,394 | `60c60ec033632f635e99fb505a426b1317a727f3d4cf3bc77063048e44b4a3bf` |
| `devsearch-runtime-checkpoint-20261008-cp02.tar.gz` | 620429525 | 1,143,132 | `e78c28686ef4e8a0f9ea00d9a12c0f7ba3c5a86c1e98cb01a5ebeaad3e8c10e5` |

**cp02 is a delta and requires cp01.** cp01 has frozen requests/protocol, all primary batch inputs, private package source/drafts/salt, real tokenizer and v1 package. cp02 has all accepted responses (9,852 primary + 120 secondary), all 24 S1 inputs, 168 new primary raw outputs, three secondary raw outputs, provenance/import/audit reports and real-model CPU receipts. cp01 has 724 manifest entries; cp02 has 470, plus each manifest itself. Both have separate checksum, manifest and restore-guide assets.

**cp02 是增量包，必须配合 cp01。** cp01 包含冻结输入、私有源数据及真实分词器；cp02 包含全部已接受响应、全部 S1 输入、新原始判定、溯源审计和真实模型回执。两个包分别有校验、清单及恢复说明附件。

Same Mac: use existing local files; no repeat download is necessary:

若在同一 Mac，直接使用本地文件，不必重新下载：

- Mutable review: `/Users/macbookair/Documents/AI foundation/devsearch-runtime/formal-review-work-20261008/`.
- Checkpoints: `/Users/macbookair/Documents/AI foundation/devsearch-runtime/checkpoints/cp01/` and `cp02/`.
- CPU receipts: `/Users/macbookair/Documents/AI foundation/devsearch-runtime/real-model-cpu-evidence-20261008/restored/receipts/`.
- Small CLI source: `/private/tmp/devsearch-cp01-source-79mr1qlc/src/`; prefer a durable exact checkout for long work.
- Existing Python: `/Users/macbookair/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`.

Another local host: authenticate through existing authorized `gh`, download all cp01/cp02 assets, verify archive hashes and every manifest entry, unpack separately, copy cp01 formal-review to a new mutable folder, then overlay cp02 formal-review. Check existing A-0001–0069 responses are byte-identical before overlay. **Do not concatenate/re-import the response files.** Run actual progress/audit and confirm the table above. Draft assets are owner-access verified, not proven accessible through an arbitrary read-only token. A Git deploy key does not authenticate Release downloads.

另一台本地主机：用已有授权 gh 下载，核验外包和内部全部清单，分别解包，复制 cp01 后叠加 cp02。先核查重叠响应相同，**不能再次拼接或导入**。重算进度并与上表一致；不要假定只读 token 或 Git deploy key 能下载 draft 附件。

```bash
gh release download devsearch-runtime-checkpoint-20261007 \
  --repo Hector-sau/devsearch --dir LOCAL_NEW_EMPTY_FOLDER
```

## 3. Resume exactly / 准确续做

Use Python **3.11+**, exact source `df44454`, and `F` as the mutable formal-review folder. `SRC` points to that checkout's `src/`; commands below require no new ML installation. Keep merged outputs **outside** `responses/`.

使用 Python 3.11 以上、准确源码及可变工作副本。以下审查命令不需新装 ML 环境；合并文件须放在 responses 之外。

```bash
export PYTHONPATH="$SRC"
"$PY" -m devsearch.review progress \
  --request "$F/input/review-request.jsonl" --batches-dir "$F/batches/A" \
  --responses-dir "$F/responses" --reviewer-id judge-primary --pass A \
  --output "$F/reports/progress-restored.json"
"$PY" -m devsearch.review audit \
  --request "$F/input/review-request.jsonl" --responses-dir "$F/responses" \
  --data-config "$F/input/data-protocol.toml" --hidden "$F/input/hidden-provenance.jsonl" \
  --output "$F/reports/audit-latest.json"
# Expected now: audit exit3, needs second804, unresolved primary29, final0.
```

1. Finish the **21 remaining S1 batch files** in `$F/batches/S1/`, excluding the three accepted names. New independent agents get only `input/judge-candidate-v1-compact.md` and their compact blinded batch; no earlier labels, hidden ranks, salt, or coordinator history. Each reads every item and writes genuine four-field judgments. No heuristic/oracle/simulated labels. Use bounded batches; save and validate after each. Avoid another architecture-research round.

   完成余下 21 个 S1 文件，排除已完成三批。新独立 agent 只看原规则及盲化批次，逐项真实判定，不生成模拟或算法标签。分批保存校验，不再消耗 token 重做架构调研。

2. Import each new S1 output with reviewer **`judge-secondary-codex`**, type **`agent`**, role **`primary`** (secondary is also a primary-role record), and **truthful current model/task/rubric revision**, not a fabricated Codex identity if the successor is another system. Preserve the same rubric hash `77af86d3e513c54c3a22d3efc40193ef7b0642e767a74d9472ba9162b55aa173`. Prevent duplicate `(item_id, reviewer_id, role)` across all files. Verify exact IDs/order, quote<=300 characters, note<=20 words, CLI outcome and progress. `accept.py` can exit 0 despite invalid/missing; `already` only means a path exists.

   复审仍以 primary 角色导入，沿用复审 campaign 身份，但如实记录当前 AI 来源，不冒充 Codex。校验全局重复、条目、引文、备注及实际导入结果，不能只相信辅助脚本退出码。

```bash
"$PY" -m devsearch.review import \
  --request "$F/input/review-request.jsonl" --judgments RAW_JSONL \
  --reviewer-id judge-secondary-codex --reviewer-type agent --role primary \
  --revision ACTUAL_RUBRIC_MODEL_TASK_REVISION --output "$F/responses/NEW_BATCH.jsonl"
```

3. Re-audit after the full sample. Keep protocol revision 5 unchanged: positive25%, negative8%, minimum30 per stratum, at least one per question, seed20261007. A stratum's **unrounded disagreement strictly >10%** triggers full second review; unresolved secondary labels count as disagreements. Generate missing expansion batches in a new existing empty directory via `review batches --items audit-latest.json --queue second-review`. Do not change thresholds, sample or candidates for nicer results.

   抽样完整后重新审计，严格按冻结协议扩审，不为效果改变阈值或候选；未决复审也计分歧。

4. Independently adjudicate all unresolved/disputed items, using `judge-adjudicator-v1-compact.md`, a distinct reviewer and role `adjudicator`; no same-item judge reuse. Adjudication must not bypass primary/second-review/expansion gates. At most one resolved adjudicator settles each item. Require audit **exit0**, `final=items=9852`, `escalation_decided=true`, empty required-second/adjudication queues and no error states. Only then generate the merged responses outside `responses/`.

   独立第三方裁决所有未决或争议，不绕过覆盖门禁。审计退出 0、全部 9,852 最终确定、队列为空后，才能生成合并响应。

5. Build audited **v2** through existing `gpu.py package --review-responses MERGED` using cp01's private revision-5 `dataset/pkg-src/data.toml` and `sources.toml`, `realdata/index.toml`, `realdata/drafts`, and real pinned tokenizer. Preserve every hash and stage-1 seal. Verify fresh authenticated retrieval and reconstruction. Keep private salt/questions/labels/corpus/models outside Git; use private Release assets, not commits. Do not run formal test scoring before validation-only selection.

   用私有第五版源配置及完整响应构建 v2，保存所有哈希和初始封存，在全新目录验证实际获取与重建。私有内容用 Release 附件，不进 Git；验证集选模前不跑正式测试评分。

6. Finish real-model CPU crash/resume and installed-wheel checks locally on permitted storage or in visible repository Actions with saved receipts. Existing receipts already prove actual pinned weights, finite forward/backward/update, and separate-process export/reload; do not claim they prove crash recovery or a wheel outside checkout. MiniLM commit **`44acabbec0ef496f6dbc93adadea57f376b7c0ec`**, `.bin` bytes **133483028**, SHA **`bcc275fe3e183a68e629c55355be77ef51c2ee780990173ca8f8edc65152fae7`**. CPU checker is `scripts/check_model_cpu.py --assets DIR --require-pretrained --output FILE`. Fetched `.bin` files need registration/conversion first; `fetch-model` output must be nonexistent or empty.

   在允许存储的本地或可查看的本仓库 Actions 完成真实模型恢复及 wheel 验证；既有回执不能替代这些检查。固定模型及准确命令如上，下载目录和已登记资产不可混淆。

7. Implementation AI pins exact model revision and audited v2 path/hash/retrieval in `configs/gpu.toml`, updates stale `docs/08`, quickstart and acceptance report, runs relevant tests and CI on that final code/config commit with required ML dependencies and **zero skips**, and retains separate exact-lock/offline-wheel evidence (current CI dependencies are unpinned). Keep PR #2 unmerged. Deliver final source/input/seal identities and `check → prepare → diagnose → run → status` commands plus resume/backup instructions.

   实现 AI 最终固定模型与 v2、更新过时说明，在同一最终代码配置提交验证 CI 和锁定环境回执，交付启动续跑备份命令；PR #2 保持未合并。

Renting is appropriate only after these non-GPU gates have evidence. Target CUDA driver/hardware, memory calibration and actual training/evaluation remain for the GPU host. No fabricated human tests, ranking improvements, résumé outcomes or token usage.

上述无需 GPU 条件都有证据后再租机。真实硬件、显存校准及训练评测留给 GPU；不编造真人测试、排序提升、简历成果或用量。

## 4. Successor's first reply / 接手者第一条回执

Report only: workspace type/path; exact source commit; checkpoint hashes verified; actual primary/S1/final counts; next missing S1 batch; and any concrete access/storage blocker. Then continue the remaining batch work under the user's scope. No paid API, GPU rental, PR merge, public publication or provider-cloud edits.

先只报告：本地／GitHub 工作地点、源码提交、检查点校验、实际首审复审及最终数量、下一缺失批次、真实访问或存储阻碍，然后续做。不新增付费 API，不租 GPU、不合并、不开公开发布、不在服务商云环境编辑。
