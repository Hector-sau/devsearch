# GPU quickstart: rented NVIDIA server / 租用 NVIDIA 服务器快速上手

One entry point, `scripts/gpu.py`, prepares and runs DevSearch on a rented Linux NVIDIA server. Every large item (Python
environment, wheelhouse, model, source snapshot, data, checkpoints, logs, backup) goes under one **runtime root on the server's
persistent volume**. Nothing is downloaded to your Mac and nothing large is committed to Git. CUDA is the release target; MPS is
optional, separate and unverified, and never blocks this path.

**Status:** this script is verified on CPU with small, original fixtures only (see `reports/gpu-release-acceptance.md`). No step
has run on a GPU, with the real MiniLM files, or with the full review bank. The first run on the server is itself the first
verification. A ranking gain is not claimed anywhere.

`scripts/gpu.py` 是在租用的 Linux NVIDIA 服务器上准备并运行 DevSearch 的唯一入口。所有大文件（环境、离线依赖包、模型、源码快照、数据、断点、日志、备份）都放在服务器**持久卷上的运行根目录**下；不向 Mac 下载，不向 Git 提交大文件。CUDA 是发布目标，MPS 可选、单独验收、尚未验证，不阻塞本流程。**现状：**仅用小型原创样例在 CPU 上验证；尚未在 GPU、真实 MiniLM 文件或完整审查库上运行。服务器上的第一次运行就是第一次验证；不声称任何排序提升。

## 1. Server requirements / 服务器要求

| Item | Requirement | Checked by |
| --- | --- | --- |
| OS / CPU | Linux x86_64, glibc 2.28+ | `check` |
| GPU | NVIDIA, at least 16 GiB memory (`min_gpu_memory_gib`) | `check` (`nvidia-smi`) |
| Driver | supports **CUDA 13.0** or newer, i.e. driver **580+**: the lock pins torch 2.14.1 built for CUDA 13 | `check` (`nvidia-smi` "CUDA Version") |
| Python | `python3.12` with `venv` and `ensurepip` (Ubuntu: `sudo apt install python3.12 python3.12-venv`) | `check` |
| Tools | `git` | `check` |
| Storage | persistent volume, at least 60 GiB free (estimate in section 6) | `check` (not tmpfs; overlay warns) |
| Permissions | unprivileged user namespaces (`unshare -rn true` works): the portable receipt proves an offline install with them | `check` |
| Network (prepare only) | HTTPS to `pypi.org`, `files.pythonhosted.org`, `huggingface.co` and its file CDN, `github.com` | `check` |
| Credentials | none for the public model; `HF_TOKEN` only if a mirror requires it | `check` reports |

## 2. Commands / 命令

```bash
# on the server
git clone https://github.com/hector-sau/devsearch.git && cd devsearch
git checkout <the reviewed commit>
mkdir -p /workspace/devsearch-runtime/inbox                  # the runtime root on the persistent volume (edit configs/gpu.toml if elsewhere)
# from your machine: copy the dataset package (section 4) into the inbox
#   scp devsearch-dataset-v1.tar.gz server:/workspace/devsearch-runtime/inbox/
python3 scripts/gpu.py check    --config configs/gpu.toml    # exit 0, or PREREQUISITE lines naming what is missing
python3 scripts/gpu.py prepare  --config configs/gpu.toml    # environment, receipts, model, sources, data; stops at the review bank (exit 3) until judgments exist
python3 scripts/gpu.py diagnose --config configs/gpu.toml    # small real-model training + inference + injected crash + resume, export, reload, backup
python3 scripts/gpu.py run      --config configs/gpu.toml    # the declared campaign (needs prepare complete and diagnose passed)
python3 scripts/gpu.py status   --config configs/gpu.toml
```

Use `tmux` or `nohup` for long stages; every stage can simply be rerun after a disconnect. Exit codes: 0 done; 1 invalid or
refused (configuration, integrity, frozen identity); 2 usage; 3 a named prerequisite is missing (`PREREQUISITE <name>: ...`);
4 a step failed or the campaign is incomplete; 5 the campaign awaits review; 6 the time limit stopped the campaign.

长时间阶段请在 `tmux` 或 `nohup` 下运行；断线后重复同一命令即可继续。退出码见上。

## 3. What each stage does / 各阶段内容

**`check`** — platform, driver CUDA version, GPU memory, Python 3.12 with venv, git, runtime root writable and persistent, free
space, user namespaces, reachability of the hosts the remaining steps need, the dataset package's SHA-256. Changes nothing.

**`prepare`** (resumable; each finished step is recorded under `<root>/state/` and skipped next time):

1. `environment`: builds the project wheel from a copy of the sources, downloads every artifact of `locks/nvidia-cuda-cp312.lock`
   with `pip download --require-hashes` into `<root>/wheelhouse/` (about 3 GB), and installs a **new** environment
   `<root>/envs/<code key>/` offline from it (`scripts/bootstrap_env.py`). A different checkout gets a new environment.
2. `receipts`: `scripts/record_receipt.py implementation` (the whole test suite in that environment, zero skips) and `portable`
   (a fresh offline install inside a network namespace whose isolation is proven, then the CPU rehearsal campaign). A formal plan
   needs both for this exact code.
3. `model`: resolves `microsoft/MiniLM-L12-H384-uncased` to a 40-hex commit (`revision = "resolve-main-once"` does this once and
   freezes it; or put a commit in `configs/gpu.toml`), downloads `config.json`, the tokenizer files and the weights **at that
   commit**, verifies every file against the hub's LFS SHA-256 or git blob SHA-1 (a mismatch deletes the file and stops), converts
   `pytorch_model.bin` to `model.safetensors` once if the repository has no safetensors (both hashes recorded), writes
   `<root>/assets/model-provenance.json` and registers the set (`bundle register-assets`, `pinned: true`).
4. `dataset`: verifies the package's SHA-256 against `configs/gpu.toml`, unpacks it with path filtering and checks every inner file.
5. `tokenizer`: `scripts/compare_tokenizers.py` compares the pinned tokenizer's complete configuration and its encodings of every
   passage and question (alone and paired) with the reference the questions were built with, plus two independent
   implementations. Any difference stops with `PREREQUISITE protocol-revision`: the data would have to be rebuilt under a recorded
   protocol revision; nothing is relabelled or replaced.
6. `data`: fetches `pytorch/tutorials@11512db7…` by exact commit, rebuilds the corpus with the pinned tokenizer, groups, splits,
   pairs, candidates and the review pool, then audits them against the frozen values in the package (corpus id, index id, test
   questions, qrels and groups, review items, training files). Any difference stops with `PREREQUISITE protocol-revision`.
7. `review`: writes judge batches for two independent passes (`<root>/review/batches/A`, `B`, 40 items each, blinded) and audits
   the imported judgments. **Until every item has two agreeing primary judgments or an adjudication, it stops with
   `PREREQUISITE review-bank` and nothing is sealed** (section 5).
8. `bundle`: `prepare_data --stage bank` (the initial test seal), then the bundle with the frozen `configs/protocol.toml`, the
   pinned assets and the lock.

**`diagnose`** — a synthetic diagnostic tree trained with the **pinned MiniLM** on the GPU (formal lengths 256/64): doctor
(training, inference timing, save/load/export/reload on the device), plan, a run killed on purpose inside `train:full:42`
(`DEVSEARCH_FAULT`), `run --resume`, `status`. It passes only if the crash happened, the resumed run completed and every artifact
re-verifies. Its numbers are plumbing evidence, never ranking results.

**`run`** — refuses (exit 3) without a complete `prepare`, pretrained (not diagnostic) assets, this code's receipts and a passed
`diagnose`. Then: host binding with the deadline (now + `time_limit_hours`), doctor, a formal plan (all four gates), and the
campaign: baseline validation, the training matrix (frozen and full recipes, seed 42, 3 epochs, FP32), validation-only selection
with the predeclared gain and latency gates, the sealed test evaluation (only after selection is committed), export, reload
verification and backup. A rerun resumes from the last committed stage.

`check` 不改动任何东西；`prepare` 分步可续：新建锁定环境、两张回执、按固定提交下载并校验模型、校验并解包数据包、比较分词器、按固定提交重建数据并核对冻结值、生成两轮审查批次并审计覆盖、封存并组装运行包。`diagnose` 用固定的 MiniLM 在 GPU 上跑合成诊断流程，并故意中断后恢复。`run` 在前置条件不全时以退出码 3 拒绝；否则按预先声明的矩阵训练、仅用验证集选择、封存后测试评估、导出、重载验证与备份。

## 4. Inputs you provide / 需要您提供的输入

- **Dataset package** `devsearch-dataset-v1.tar.gz` (1.2 MB; SHA-256 pinned in `configs/gpu.toml`). It holds the source
  configuration, the data configuration with the **private review salt**, the index configuration, the synthetic question and qrel
  drafts, the reference tokenizer vocabulary and the reference passage texts, plus `package.json` with the frozen values. It is not
  in Git (it contains the test questions and the salt). Copy it to `<root>/inbox/`, or host it privately and set `dataset.url`.
  It can be rebuilt byte-for-byte with `scripts/gpu.py package` from the drafts.
- **Review judgments** (section 5).
- Optionally `HF_TOKEN` in the environment, only for a mirror that needs it. No other credential is read.

数据包含私有盐值与测试问题，不进 Git；请复制到 `<root>/inbox/`（或私下托管并设置 `dataset.url`）。另需审查判定（第 5 节）；公开模型无需令牌。

## 5. The review bank / 审查库

The declared protocol needs, for each of the 9,852 blinded (question, passage) items, two independent primary judgments that agree,
or an adjudication. `prepare` writes the batches and prints the exact import command. Judgment lines are
`{"item_id", "relevance": 1 | 0 | "unsure", "evidence_quote", "note"}` (prompt: `scripts/realdata/prompts/judge-candidate-v1.md`).

```bash
PY=$(python3 scripts/gpu.py env-python --config configs/gpu.toml)
R=/workspace/devsearch-runtime
$PY -m devsearch.review import --request $R/evaluation/review/review-request.jsonl --judgments judged-A/*.jsonl \
    --reviewer-id judge-a --reviewer-type agent --revision judge-candidate-v1 --role primary --output $R/review/responses/A.jsonl
# same for pass B with another reviewer id; then
python3 scripts/gpu.py prepare --config configs/gpu.toml     # audits; lists the adjudication queue if anything disagrees
$PY -m devsearch.review batches --request $R/evaluation/review/review-request.jsonl --output-dir $R/review/batches/adjudication \
    --pass ADJ --items $R/review/audit-<latest>.json
# judge those, import with --role adjudicator, rerun prepare
```

Agent judgments stay labelled `agent`; human judgments `human` (no `--revision`). A missing or unresolved item keeps the audit
incomplete (exit 3), and `prepare_data --stage bank` refuses it too: sealing and formal evaluation stay blocked. The measured pilot
cost is in `reports/accelerator-readiness-evidence.md`; scaling it up is a paid-usage decision and has not been started.

协议要求 9,852 个盲审条目每条有两份一致的独立主判定或一份裁决。导入命令见上；缺失或未决条目会使审计不完整（退出码 3），封存与正式评测保持阻塞。扩大审查涉及付费用量，尚未启动。

## 6. Storage estimate / 存储估算

Estimates from file sizes known today, not measured on a GPU host:

| Item | Size |
| --- | --- |
| wheelhouse (`<root>/wheelhouse/`) | about 3 GB (measured for this lock) |
| environment (`<root>/envs/<key>/`) | about 6–8 GB installed |
| portable receipt's temporary second environment | about 6–8 GB while it runs (in `TMPDIR` = `<root>/tmp`) |
| model download + registered copy + bundle copy | about 0.5 GB (MiniLM-L12-H384, 33M parameters) |
| source snapshot `pytorch/tutorials` | about 0.2 GB |
| data, evaluation, review batches | about 0.1 GB |
| diagnose tree and campaign checkpoints, releases, backup | about 3–6 GB |
| **total** | **about 20–25 GB**; keep 60 GiB free |

## 7. Restart and time limit / 重启与时限

- Any stage: rerun the same command. Finished steps are skipped; downloads keep verified files and resume partial ones; an
  environment or receipt directory left by an interrupted step is moved aside (`*.failed-*`, `*.incomplete-*`), never reused.
- `run` after an interruption resumes the campaign from its last committed stage (completed epochs are kept).
- Time limit: the campaign deadline is fixed when the plan is written (`time_limit_hours`). At it the campaign stops cleanly with
  exit 6, keeping the last complete epoch. To continue: `python3 scripts/gpu.py run --config configs/gpu.toml --extend-hours 6`
  (writes a new host binding with the later deadline through `plan --revise-binding`, then resumes).
- Never edit `<root>/state/frozen.json`: model, dataset, backend, lock and protocol are frozen there; a configuration that differs
  is refused (exit 1). For a different model or dataset, use a new runtime root.
- Only one stage at a time per runtime root (a lock file enforces it).

任何阶段重复执行即可继续；已完成的步骤跳过，下载续传且只保留校验通过的文件。到达时限时以退出码 6 干净停止并保留最后完整一轮；用 `--extend-hours` 延长后继续。不要修改 `frozen.json`；换模型或数据请用新的运行根目录。

## 8. Outputs / 输出位置

| Path under the runtime root | Content |
| --- | --- |
| `logs/gpu.log`, `logs/*.log` | one line per step; one full log per command (command line, exit status, stdout, stderr) |
| `state/*.json` | step records, `frozen.json`, latest `check` |
| `assets/model-provenance.json`, `assets/model/` | model files, hashes, commit, conversion record, `assets-manifest.json` |
| `data/membership-audit.json`, `data/tokenizer-comparison-*.json` | the frozen-membership audit and the tokenizer comparison |
| `receipts/<code key>/` | implementation and portable receipts |
| `review/` | batches, imported responses, audits, merged responses |
| `bundle/`, `evaluation/` | the sealed training-side bundle and the separate test side |
| `diagnose/<time>/` | the diagnostic tree, its plan, run and status (diagnostic only) |
| `campaign/` | `host.toml`, doctor report, `plan/execution-plan.json`, `run/` (run state, stages, final report, per-query results, release) |
| `backup/` (or `backup.dir`) | the campaign's backup copy and receipt |

Copy `campaign/run` stage outputs (final report, `report.md`, release) and `backup/` off the server before the rental ends; then
paste the short summary (`status` output, final report's headline numbers, receipts' hashes) into `reports/` in a pull request.
Large files stay out of Git.

## 9. Cleanup / 清理

After the results and the backup are copied off the server, these can be deleted: `wheelhouse/`, `tools/`, `build/`, `dist/`,
`envs/*.failed-*`, `receipts/*.incomplete-*`, `assets/download/` (keep `model-provenance.json`), `diagnose/`, `tmp/`. Keep
`state/`, `logs/`, `campaign/`, `bundle/`, `evaluation/`, `review/` and the backup until the results are reviewed. Deleting the
whole runtime root is the clean reset; a new `prepare` then starts from scratch.

结果与备份拷离服务器后，可删除上列可再生目录；结果审阅前保留状态、日志、实验、运行包、评测、审查与备份。删除整个运行根目录即完全重置。

## 10. CPU fixture check (no GPU) / 无 GPU 的样例检查

`tests/test_gpu.py` runs the whole sequence on CPU with an original fixture (local git source, WordPiece fixture model, a package
built by `gpu.py package`, scripted fixture judgments) under the `cpu-diagnostic` profile and `environment.mode = "current"`. It
proves the orchestration, the stops and the recovery; it proves nothing about CUDA, MiniLM or ranking quality.
