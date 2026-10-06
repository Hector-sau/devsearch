# Pre-GPU prerequisites and target verification checklists / 租用 GPU 前的前置条件与目标机验收清单

Written 2026-10-06 after the review of `e387671`. **Nothing in sections 4 and 5 has been executed on a target machine.** Each step names
the evidence it produces and the gate it feeds (docs/11). CUDA and MPS are verified separately: finishing CUDA never waits for MPS, and
evidence from one never counts for the other.

本文写于 2026-10-06（`e387671` 审核之后）。**第 4、5 节的步骤均未在目标机执行。**每步注明产出的证据与对应门禁（第 11 号文档）。CUDA 与 MPS 分别验收：CUDA 的完成不等待 MPS，一方的证据也不能算作另一方的。

**Update 2026-10-06:** sections 1, 2, 3 (batches, import, audit) and the CUDA checklist steps 2-8 are now automated by
`scripts/gpu.py` ([GPU_QUICKSTART.md](GPU_QUICKSTART.md)); the dataset package freezes the membership values listed in section 2, and
the tokenizer comparison and membership audit stop the script instead of sealing on any difference. The steps below remain the
definition of what the script must prove; none has run on a GPU yet.

**2026-10-06 更新：**第 1、2、3 节（批次、导入、审计）及 CUDA 清单第 2 至 8 步已由 `scripts/gpu.py` 自动执行；数据包冻结第 2 节所列成员值，分词器比较与成员审计如有差异即停止、不封存。以下步骤仍是脚本必须证明的内容，尚未在 GPU 上运行。

## 1. Pin MiniLM and its tokenizer, then prove the tokenization / 固定 MiniLM 与分词器并证明编码一致

1. On a machine that can reach `huggingface.co` and its file hosts, download `microsoft/MiniLM-L12-H384-uncased` at a full 40-hex commit: `config.json`, `vocab.txt`, every tokenizer file the repository has (`tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `tokenizer.json`), and the weights. The project loads `model.safetensors` only; if the repository publishes only `pytorch_model.bin`, convert it once in a separate, recorded step (`torch.load(..., weights_only=True)`, then `safetensors` save) and keep both hashes.
2. `python -m devsearch.bundle register-assets --directory D --repo microsoft/MiniLM-L12-H384-uncased --revision <40-hex> --license MIT` writes `assets-manifest.json` with every file's hash; `pinned` is true only for a 40-hex revision.
3. `python scripts/compare_tokenizers.py --candidate D --reference <stand-in vocabulary> --texts <corpus>/chunks.jsonl --texts <drafts>/queries-train.jsonl --texts <drafts>/queries-validation.jsonl --texts <drafts>/queries-test.jsonl --output tokenizer-comparison.json` must exit 0. It compares the **complete effective configuration** (normalizer, pre-tokenizer, post-processor, WordPiece settings, vocabulary hash, special and added tokens, pair special-token count) and the **actual encodings** of every passage and question, alone and as pairs, against (a) the project tokenizer built from the reference directory, (b) Transformers' independent pure-Python BERT tokenizer configured from the candidate's own `tokenizer_config.json`, and (c) `AutoTokenizer` loaded from the pinned folder. The project tokenizer now refuses any declared setting it cannot reproduce (no basic tokenization, `never_split`, non-default special tokens, added tokens), so a configuration difference cannot be silently ignored.
4. Rebuild from `prepare` onward with the pinned directory, before any sealing. Chunk ids depend on document, offsets, text and chunker version, not on the tokenizer's identity: **if the comparison is identical, the rebuilt corpus must have the same `corpus_id`** (`corpus-4367f673…` for the current build) and every downstream artifact (groups, splits, questions, pairs, candidates, review pool) stays valid. If it differs, the data are rebuilt and section 2's revision rule applies.

The stand-in check already ran: the project tokenizer and the independent pure-Python reference agree on all 4,259 passages and 914 questions, alone and as pairs (evidence report 2.10). That proves the implementation, not the pinned files.

在能访问 `huggingface.co` 的机器上按 40 位提交下载全部配置、分词文件和权重（仅有 `.bin` 时单独转换为 safetensors 并记录两份哈希），用 `register-assets` 登记，再运行 `compare_tokenizers.py`：比较完整的有效分词配置以及全部段落与问题（单条与成对）的实际编码，对照参考目录、独立的纯 Python BERT 分词器和固定目录的 `AutoTokenizer`，必须退出 0。项目分词器对无法复现的声明设置一律拒绝。编码一致时，重建后的 `corpus_id` 必须不变，下游产物全部有效；否则按第 2 节记录协议修订后重建。

## 2. Freeze the dataset / 冻结数据集

- The questions stay labelled `origin: synthetic` with one `agent` review each; no step may relabel them as human-reviewed. They are suitable for a first engineering evaluation.
- **Test membership is frozen** at the 197 test questions committed by the evaluation manifest `0c2eb66f…` (test groups hash recorded there). A question may leave the test set only when a rebuild that the tokenizer requires changes its passage (it is then not carried over by exact chunk id and text). Every such change is written down as a protocol revision with the reason and the before/after commitments. Questions are never replaced, added or removed to change reranking gains, and no reranker test result is looked at before the seal.
- If the pinned tokenizer encodes differently, the audited assignment can move many groups between splits (the reader fix of 2026-10-05 moved 489 of 1,178 selected passages). That is a protocol revision, recorded before any new labels are made.

问题保持“合成、Agent 审查”标记。测试集成员冻结为评测清单 `0c2eb66f…` 承诺的 197 题；只有分词器确需重建、导致段落变化时才可移出，并须记录协议修订、原因及前后承诺哈希。不得为提高重排收益换题、增题或删题，封存前不看任何重排测试结果。

## 3. Candidate review: pilot, then scale / 候选审查：先试点，再扩大

1. Pilot (done, evidence report 2.10): `scripts/realdata/review_pilot.py sample` draws a stratified sample from the blinded request (known positives, other candidates ranked 1-10 and 11-50; strata only in a private file), two independent judge runs use `scripts/realdata/prompts/judge-candidate-v1.md`, `convert` turns their lines into bank response records checked by the bank's own parser, and `report` measures agreement per stratum and the cost per judgment, then extrapolates to the 9,852 items. Pilot judgments stay outside the evaluation tree and are not part of the bank.
2. Scale-up (not started; tooling: `python -m devsearch.review batches|import|audit`, driven by `gpu.py prepare`): two primary passes over all items with the same prompt revision, then adjudication of every disagreement and unresolved item by a separate adjudicator (a different prompt revision, or a human), then `prepare_data --stage bank --reviews <all responses>`. Coverage must be complete before sealing; partial judging cannot seal. The bank reports agent and human judgments separately.

试点已完成（证据报告 2.10）：分层抽样、两名独立判定、按分层统计一致性与单条成本，再外推到 9,852 项；试点判定不进入审查库。扩大规模（未开始）：全部条目两轮主判定、分歧与未决项单独裁决，再运行 `bank`；必须全覆盖才能封存。

## 4. CUDA checklist / CUDA 验收清单

| # | Step | Evidence | Gate |
| --- | --- | --- | --- |
| 1 | Linux x86_64 host with an NVIDIA driver that supports the CUDA build of torch 2.14.1 in `locks/nvidia-cuda-cp312.lock`; Python 3.12; `TMPDIR` and `DEVSEARCH_RUNTIME_ROOT` on the runtime volume | host description | – |
| 2 | Wheelhouse built on a connected machine: `scripts/bootstrap_env.py download --lock locks/nvidia-cuda-cp312.lock --dest W` (about 3.0 GB), copied to the host | download log, lock hash | G02 |
| 3 | `python3.12 scripts/bootstrap_env.py install …` then `scripts/record_receipt.py implementation` there: the whole suite, zero skips | implementation receipt | `implementation_verified` |
| 4 | `python3.12 scripts/record_receipt.py portable --lock locks/nvidia-cuda-cp312.lock --wheelhouse W --python python3.12 …`; check first that `unshare -rn true` works for the user (some hosts forbid unprivileged namespaces; the receipt then says `offline: false` and the gate fails) | portable receipt bound to the lock and Linux/x86_64 | `portable_bundle_verified` |
| 5 | Pinned assets (section 1) and the frozen, sealed data (sections 2-3) in a bundle: `bundle assemble --bundle B --lock locks/nvidia-cuda-cp312.lock` | bundle manifest naming the lock | `data_ready` |
| 6 | `doctor --profile nvidia-cuda --host host.toml --bundle B`: training per recipe, inference timing and save/load/export/reload on the GPU, every measurement successful and finite | doctor report, backend fingerprint | `backend_verified[nvidia-cuda]` |
| 7 | `plan --receipts <dir>`: a formal plan needs all four gates passed and the portable receipt for this bundle's lock and platform | execution plan | `accelerator_run_ready` |
| 8 | `run`, `status`; backup to an independent device | final report, release, backup receipt | G08-G10 |

## 5. MPS checklist / MPS 验收清单

| # | Step | Evidence | Gate |
| --- | --- | --- | --- |
| 1 | Apple silicon Mac (arm64), macOS 14 or newer (the lock's wheels are tagged for macOS 11-14 arm64), Python 3.12 | host description | – |
| 2 | Wheelhouse for `locks/apple-mps-cp312.lock` (about 191 MB); note it pins torch 2.11.0, not the 2.14.1 tested so far, so step 3 is the first test of that version | download log | G02 |
| 3 | `scripts/record_receipt.py implementation` on the Mac: the whole suite, zero skips | implementation receipt | `implementation_verified` |
| 4 | `scripts/record_receipt.py portable --lock locks/apple-mps-cp312.lock …`: on macOS the install and the campaign run under `sandbox-exec` with a profile that denies `network*`, and the script first **proves** the isolation with a connection attempt that must fail. This path is written and unit-tested for its construction, but has **never run on a Mac**: the first run is itself the verification. If `sandbox-exec` is unavailable the receipt says `offline: false` and the gate fails | portable receipt bound to the lock and Darwin/arm64 | `portable_bundle_verified` |
| 5 | Bundle with `--lock locks/apple-mps-cp312.lock` | bundle manifest | `data_ready` |
| 6 | `doctor --profile apple-mps …`, then `plan`, `run`, `status`, backup, exactly as for CUDA | as in section 4 | `backend_verified[apple-mps]`, G08-G10 |

## 6. What the final report must answer / 最终报告必须回答的问题

BM25 is the baseline, not a failure: 0.851 MRR@10 on the synthetic validation questions. The final report states, for the frozen test set, the
BM25 and selected-reranker metrics with paired bootstrap intervals, p50 and p95 latency of both modes measured on the target device, and the
decision the predeclared selection and latency policy takes: whether the relevance gain is large enough to justify the added latency. "No,
BM25 stays the default" is a valid, reportable result and is published the same way as "yes".

BM25 的 MRR 0.851 是基线，不是失败。最终报告须给出冻结测试集上 BM25 与选定重排模型的指标及配对自助区间、目标设备上两种模式的 p50／p95 延迟，以及预先声明的选择与延迟策略得出的结论：相关性改善是否值得增加的延迟。“否，默认仍为 BM25”同样是有效结果，照常发布。
