# GPU quickstart: rented NVIDIA server / 租用 NVIDIA 服务器快速上手

One entry point, `scripts/gpu.py`, prepares and runs DevSearch on a rented Linux NVIDIA server. Every large item (Python
environment, wheelhouse, model, source snapshot, data, checkpoints, logs, backup) goes under one **runtime root on the server's
persistent volume**. Nothing is downloaded to your Mac and nothing large is committed to Git. CUDA is the release target; MPS is
optional, separate and unverified, and never blocks this path.

**Status:** the pinned real model passed CPU checks and crash/resume, and the complete agent review bank is packaged (see
`reports/gpu-release-acceptance.md`). The rented GPU passed a base-image CUDA tensor sanity check only; the project's locked
CUDA installation, diagnosis and formal campaign still need verification. A ranking gain is not claimed anywhere.

`scripts/gpu.py` 是租用服务器上的统一入口，大文件保存在服务器持久卷。CUDA 是发布目标，MPS 单独验收。**现状：**固定真实模型已通过 CPU 检查和恢复演练，完整 Agent 审查库已打包；租用 GPU 仅通过基础镜像的微型 CUDA 运算，项目锁定环境、诊断和正式训练尚待实机验证。不声称排序提升。

## 0. Before renting: what must already be done / 租机前必须完成的事项

The rental should be spent on installation, hardware diagnosis, training, evaluation and export only. These items do not need a GPU
and must be finished first; `gpu.py` refuses to seal or run while any of them is missing.

| # | Prerequisite (no GPU needed) | Evidence that it is done | Status on 2026-10-08 |
| --- | --- | --- | --- |
| P1 | Code release candidate accepted | PR #2 reviewed; CI green on its head (zero skips) | candidate `22db6f4`: CI and the non-GPU evidence run green (`reports/gpu-release-acceptance.md`); **awaiting your approval** |
| P2 | Pinned MiniLM obtained and verified: hashes against the hub, loads, tokenizer compatible, CPU forward/backward (`scripts/check_model_cpu.py`) | `assets/model-provenance.json`, `assets/model-cpu-check.json`, `data/tokenizer-comparison-*.json` | **done on CPU**: `44acabbe…` pinned in `configs/gpu.toml`; CPU check 10/10; crash and resume with it through the controller (`scripts/check_real_model_resume.py`) |
| P3 | Data rebuilt with that tokenizer; frozen test membership audited | `data/membership-audit.json` identical | **done**: rebuilt with the real tokenizer; the 14 frozen membership values equal v1's |
| P4 | Review bank complete under protocol revision 5: one primary pass over all 9,852 items, the seeded stratified second review (and any escalation), adjudication (docs/15) | `devsearch.review audit --data-config` complete; initial seal | **done**: audit exit 0, 9,852/9,852 final (agent judgments only) |
| P5 | Run package v2 built with the bank and sealed hashes, uploaded privately, SHA-256 pinned in `configs/gpu.toml` | `gpu.py package --review-responses …` output; fresh-root `prepare` reproduces the seal | **done**: `06e7ae62…`, initial seal `9a59905c…`; byte-reproducible; a fresh root rebuilt it from a fresh download (with `gpu.py package`; the server's own `prepare` reproduces it again on the host) |

## 0.1 Pre-rental checklist for the server / 服务器租用前检查表

Check these with the provider before paying (each is also checked by `gpu.py check` on the machine):

- [ ] **Driver compatibility:** NVIDIA driver **580 or newer** (`nvidia-smi` header shows "CUDA Version: 13.0" or higher). The lock pins
      torch 2.14.1 built for CUDA 13; an older driver (535, 550, 570) cannot run it. Linux x86_64, glibc 2.28+.
- [ ] **GPU memory:** 16 GiB or more.
- [ ] **Persistent storage:** a volume that survives a stop/restart (not tmpfs; ask whether the container overlay persists), **60 GiB
      free** for the runtime root (estimate in section 6).
- [ ] **Python 3.12** with `venv` and `ensurepip` (`python3.12 -m venv --help` works), plus `git`.
- [ ] **Portability evidence:** either local user namespaces (`unshare -rn true` exits 0), or explicitly verified Linux CI evidence
      imported using section 0.2. A denied namespace never silently falls back or counts as target-host isolation.
- [ ] **Network during `prepare`:** outbound HTTPS to `pypi.org`, `files.pythonhosted.org`, `github.com`, `huggingface.co` and its file
      hosts (`cas-bridge.xethub.hf.co`, `cdn-lfs*.hf.co`). If Hugging Face is blocked there too, use section 10 before renting.
- [ ] **Access:** SSH with a terminal multiplexer (`tmux`), and read access to this private repository (section 2) and to the run package
      (section 4).

租机前确认驱动、显存、持久盘、Python、网络与 SSH。便携验证需本机支持 `unshare -rn true`，或按下一节显式导入同版本 Linux CI 证据。

## 0.2 Containers without network namespaces / 不支持网络命名空间的容器

Keep `receipts.source = "local"` for the original workflow. For a restricted Linux/x86_64 rental, the operator can explicitly
choose `source = "ci"`. The CI job proves offline portability; the rental still installs the exact tested wheel and dependency
set, checks package origins and `pip check`, runs the entire implementation suite with zero skips, then runs the real-model CUDA
diagnosis and fresh doctor/calibration. No target-host OS network isolation is claimed. Ordinary pip offline flags are not proof
of OS isolation. The training/evaluation protocol and four acceptance gates do not change.

原有方式默认 `source = "local"`。受限 Linux 容器可明确选择 `source = "ci"`：CI 证明离线可移植性，本机仍安装完全相同的包、检查依赖来源、运行零跳过测试及真实 CUDA 诊断。不会声称本机网络被系统隔离，也不改变训练评测协议或四项门禁。

After regular CI and **both** evidence jobs pass on the exact deployment commit, on your authenticated operator machine:

```bash
git checkout <exact-code-commit>
python3.12 scripts/ci_receipts.py fetch --run-id <successful-evidence-run> \
  --commit <exact-code-commit> --output /absolute/path/to/verified-ci
```

`fetch` uses authenticated `gh api`, verifies repository/workflow/head commit, successful jobs, artifact SHA-256, positive-control
isolation, receipt seals, dependency lock and all deployment files. It outputs `import_sha256`. The artifact includes the **same
project wheel tested by CI**, the dependency-wheel inventory and two receipts, without downloading the large dependency wheelhouse
to the Mac. Transfer this small directory and the exact checkout/archive to the rental; keep GitHub credentials on the operator
machine. In the separate host config:

```toml
[receipts]
record = true
source = "ci"
import_dir = "/persistent/verified-ci"
import_sha256 = "<fetch output: SHA-256 of verified-origin.json>"
```

Run the normal `check → prepare → diagnose → run → status` sequence. Import errors are explicit prerequisites, never a reason to
disable the gate. The pin is operator-authorized provenance, not a digital signature: obtain it from the authenticated fetch, not
from an untrusted attachment. The export binds the full commit and tracked deployment files, including scripts/tests/workflows;
editing them requires new CI evidence. A changed import pin requires a new runtime. Resume revalidates source, imported files,
prepared receipts and installed dependencies. Old local runtimes without the new config fields retain local-recording semantics.

在已登录 GitHub 的操作机执行 `fetch`，复制输出目录和准确版本源码到租机，再填上述配置。小型证据包包含 CI 测试过的安装包；大依赖由服务器下载并对比哈希。按通常顺序执行。校验值来自可信下载步骤，不是数字签名；不能从不可信附件照抄。修改脚本、测试或工作流须重新生成 CI 证据；恢复时会重新核验。

## 1. Server requirements / 服务器要求

| Item | Requirement | Checked by |
| --- | --- | --- |
| OS / CPU | Linux x86_64, glibc 2.28+ | `check` |
| GPU | NVIDIA, at least 16 GiB memory (`min_gpu_memory_gib`) | `check` (`nvidia-smi`) |
| Driver | supports **CUDA 13.0** or newer, i.e. driver **580+**: the lock pins torch 2.14.1 built for CUDA 13 | `check` (`nvidia-smi` "CUDA Version") |
| Python | `python3.12` with `venv` and `ensurepip` (Ubuntu: `sudo apt install python3.12 python3.12-venv`) | `check` |
| Tools | `git` | `check` |
| Storage | persistent volume, at least 60 GiB free (estimate in section 6) | `check` (not tmpfs; overlay warns) |
| Portability proof | local `unshare -rn`, or explicitly pinned Linux CI import (section 0.2) | `check` |
| Network (prepare only) | HTTPS to `pypi.org`, `files.pythonhosted.org`, `huggingface.co` and its file CDN, `github.com` | `check` |
| Credentials | none for the public model; `HF_TOKEN` only if a mirror requires it | `check` reports |

## 2. Commands / 命令

The repository is **private**. The server needs read access to it, and nothing else from GitHub except public `pytorch/tutorials`.
Recommended: a **read-only deploy key** (scoped to this one repository, revocable, no personal token on the server):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/devsearch_deploy -N "" -C "devsearch-gpu-server"
cat ~/.ssh/devsearch_deploy.pub     # GitHub: repository Settings -> Deploy keys -> Add deploy key; leave "Allow write access" UNCHECKED
GIT_SSH_COMMAND="ssh -i ~/.ssh/devsearch_deploy -o IdentitiesOnly=yes" git clone git@github.com:Hector-sau/devsearch.git
```

Alternative: a fine-grained personal access token limited to this repository with *Contents: Read-only* and a short expiry. Do not put the
token in the clone URL (it would stay in `.git/config` and the shell history); use
`git -c credential.helper='!f() { echo username=x-access-token; echo password=$DEVSEARCH_GH_TOKEN; }; f' clone https://github.com/Hector-sau/devsearch.git`
with `DEVSEARCH_GH_TOKEN` exported for that one command. Either way: remove the key or revoke the token when the rental ends. `gpu.py` itself
never reads a GitHub credential and writes none; its only optional credential is `HF_TOKEN` (section 1).

```bash
# on the server
cd devsearch && git checkout <the reviewed commit>          # the PR's head commit; check with `git rev-parse HEAD`
mkdir -p /workspace/devsearch-runtime/inbox                  # the runtime root on the persistent volume (edit configs/gpu.toml if elsewhere)
# the audited dataset package (section 4): either copy it into the inbox ...
#   gh release download devsearch-runtime-checkpoint-20261007 --repo Hector-sau/devsearch --pattern 'devsearch-dataset-v2.*'   # on your machine
#   sha256sum -c devsearch-dataset-v2.tar.gz.sha256 && scp devsearch-dataset-v2.tar.gz server:/workspace/devsearch-runtime/inbox/
# ... or export DEVSEARCH_PACKAGE_TOKEN (read access to this private repo's releases) and `prepare` fetches dataset.url itself
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
3. `model`: uses the pinned commit `44acabbec0ef496f6dbc93adadea57f376b7c0ec` of `microsoft/MiniLM-L12-H384-uncased` from `configs/gpu.toml`
   (`revision = "resolve-main-once"` would resolve main once and freeze it; the shipped configuration no longer needs that), downloads `config.json`, the tokenizer files and the weights **at that
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
7. `review`: uses the review bank shipped in the package (normal case) or the judgments imported on the server, and audits them under
   the dataset's declared protocol (`data.toml [review]`; revision 5: one primary pass, a seeded stratified second review, adjudication).
   **Until every item is final under that protocol, it stops with `PREREQUISITE review-bank` and nothing is sealed** (section 5).
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

| # | Input | Where it comes from | When needed | Without it |
| --- | --- | --- | --- | --- |
| 1 | **Read access to the private repository** (deploy key or token, section 2) | you, on GitHub | before cloning | cannot clone |
| 2 | **Audited dataset package** `devsearch-dataset-v2.tar.gz`, 1,645,198 bytes, SHA-256 `06e7ae62c3c78cc14d06407005dc91117d869cd73b8e28c45491744e9c488b1f` (initial seal `9a59905c685cec354ed4f3eca39dcf3ee54685f54f6ce6c87d2ac98c7660e124`) | private draft-release asset 620811833 of this repository (not in Git: it holds the test questions, the private review salt and the complete review bank); rebuilt byte-for-byte with `gpu.py package` from a fresh download | `prepare`, copied to `<root>/inbox/` or fetched from `dataset.url` with `DEVSEARCH_PACKAGE_TOKEN` | `PREREQUISITE dataset-package` / `dataset-credentials` |
| 3 | **Runtime root** on the persistent volume, created first; edit `runtime.root` (and `runtime.mount`, `backup.dir` if you have a second volume) in `configs/gpu.toml` | the server | every stage | `PREREQUISITE runtime-root` |
| 4 | **Review judgments** under protocol revision 5: shipped complete in v2 (9,852/9,852 final: primary pass, 924-item stratified sample plus 553 escalated second reviews, 284 adjudications; agent judgments, never human) | existing session quota (agents) and/or people; no paid API without a new approval | `prepare` step `review`, before sealing | `PREREQUISITE review-bank`; nothing is sealed, `run` cannot start |
| 5 | Network from the server to PyPI, Hugging Face (or the section 10 procedure), GitHub | the provider | `prepare` | `PREREQUISITE network-*` |
| 6 | `HF_TOKEN` | only if the model host requires it (the public model does not) | `prepare` step `model` | `PREREQUISITE model-credentials` |
| 7 | `time_limit_hours` (default 24) and `min_free_gib` (default 60) | you, in `configs/gpu.toml` | `run` | defaults apply |

**Private, server-downloadable package.** Keep the package out of Git. Upload it once as an asset of a release in this private repository
(GitHub: Releases, draft or published, attach the file) and give the server the asset's API URL and a read-only token:

```toml
[dataset]
package = "inbox/devsearch-dataset-v2.tar.gz"
sha256 = "<the SHA-256 printed by gpu.py package>"
url = "https://api.github.com/repos/Hector-sau/devsearch/releases/assets/<asset id>"
token_env = "DEVSEARCH_PACKAGE_TOKEN"     # export it on the server; a fine-grained token with Contents: Read-only on this repository
```

`prepare` downloads with `Authorization: Bearer` and `Accept: application/octet-stream`, refuses a missing or rejected token as
`PREREQUISITE dataset-credentials`, and uses the file only if its SHA-256 equals the pin. A package built with `--review-responses` also
carries the complete review bank and the expected initial-seal hash; a fresh server then seals without any review step and must reproduce
that exact seal. Copying the file to `<root>/inbox/` with `scp` remains the alternative.

No other credential is read; none is written to any file or log. The dataset package's SHA-256 is pinned in `configs/gpu.toml`; a different
file is refused.

数据包含私有盐值与测试问题，不进 Git；请复制到 `<root>/inbox/`。输入清单见上表；缺失任何一项都会以带名称的 `PREREQUISITE` 停止，不会替代或伪造。

## 5. The review bank / 审查库

The bank normally arrives inside the run package (P5): `prepare` copies it to `review/responses/`, audits it under the dataset's
declared protocol and checks that the seal it produces equals the one frozen in the package. Judging on the server is only needed when
no complete bank was shipped. The protocol (docs/02 revision 5, docs/15) needs one complete primary pass over the 9,852 blinded items
by the declared `primary_reviewer`, the seeded stratified second review the audit lists as `second_review_queue` (and any escalated
stratum), and an independent adjudication of everything in `adjudication_queue`. Judgment lines are
`{"item_id", "relevance": 1 | 0 | "unsure", "evidence_quote", "note"}` (prompt: `scripts/realdata/prompts/judge-candidate-v1.md`).

```bash
G="python3 scripts/gpu.py cmd --config configs/gpu.toml --"; R=/workspace/devsearch-runtime
$G review import --request $R/evaluation/review/review-request.jsonl --judgments judged-A/*.jsonl \
    --reviewer-id judge-primary --reviewer-type agent --revision judge-candidate-v1 --role primary --output $R/review/responses/A.jsonl
python3 scripts/gpu.py prepare --config configs/gpu.toml     # audits; lists the second-review and adjudication queues
$G review batches --request $R/evaluation/review/review-request.jsonl --output-dir $R/review/batches/S1 \
    --pass S1 --items $R/review/audit-<latest>.json --queue second-review
# judge those with another reviewer id (--role primary), import, rerun prepare; then the adjudication queue:
$G review batches --request $R/evaluation/review/review-request.jsonl --output-dir $R/review/batches/ADJ1 \
    --pass ADJ1 --items $R/review/audit-<latest>.json
# judge those, import with --role adjudicator, rerun prepare
```

Agent judgments stay labelled `agent`; human judgments `human` (no `--revision`). A missing or unresolved item keeps the audit
incomplete (exit 3), and `prepare_data --stage bank` refuses it too: sealing and formal evaluation stay blocked. Batches, import,
the second review, adjudication, resumable `progress` (its `--budget-tokens` is an estimate for choosing batches, not a spending cap)
and the pilot/formal separation are in [docs/15-review-workflow.md](15-review-workflow.md).

审查库通常随运行包交付；服务器 `prepare` 按数据集声明的协议（第五版：一遍主审、分层随机复核、裁决）审计并核对封存值。缺失或未决条目会使审计不完整（退出码 3），封存与正式评测保持阻塞。

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
- `run` after an interruption resumes the campaign from its last committed stage (completed epochs are kept). Rehearsed on CPU with a
  `SIGKILL` of everything mid-training: the next `gpu.py run` took over the dead owner's campaign lock and finished.
- If only your SSH session ends, the campaign process **keeps running** (it has its own session). `gpu.py run` then stops with
  `PREREQUISITE campaign-lock` instead of starting a second campaign; watch it with `gpu.py status`, and rerun `gpu.py run` once it is done.
  Use `tmux` or `nohup` for `gpu.py` itself so the final status lines are not lost.
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

## 10. If the server cannot reach Hugging Face / 服务器无法访问 Hugging Face

`prepare` downloads the model itself. If `check` reports `PREREQUISITE network-model-host`, do the download on **any other authorized
machine** (a laptop is fine; the files are about 130 MB, the project needs nothing but Python 3.11+ for this) and ship the directory:

```bash
# on a machine that can reach huggingface.co (no environment needed; standard library only)
python3 scripts/gpu.py fetch-model --revision main --output ./minilm-fetch          # resolves main ONCE and prints the 40-hex commit
#   or, with the commit already chosen: --revision <40-hex>
scp -r ./minilm-fetch server:/workspace/devsearch-runtime/inbox/minilm
```

`fetch-model` verifies every file against the hub's LFS SHA-256 or git blob SHA-1 and writes `fetch-manifest.json` (commit, hashes). On the
server set, in `configs/gpu.toml`:

```toml
[model]
source = "local"
local_dir = "/workspace/devsearch-runtime/inbox/minilm"
revision = "<the printed 40-hex commit>"
```

Before copying, check the files on that machine's CPU (needs the ML packages): `python3 scripts/check_model_cpu.py --assets <registered dir> --require-pretrained`, or let `prepare` do it (it runs the same check after registering, and stores `assets/model-cpu-check.json`).

`prepare` re-checks every file against the recorded hashes (a changed file or a different commit is refused), converts `pytorch_model.bin`
to `model.safetensors` once if needed, and registers the set exactly as for a direct download. A mirror that speaks the Hub API can be used
instead with `endpoint = "https://<mirror>"`. Without `fetch-manifest.json`, a local directory is hashed as found and the provenance says the
hub's hashes were **not** verified. This container cannot reach Hugging Face, so the procedure is tested against a local fake hub only.

## 11. CPU fixture check (no GPU) / 无 GPU 的样例检查

`tests/test_gpu.py` runs the whole sequence on CPU with an original fixture (local git source, WordPiece fixture model, a package
built by `gpu.py package`, scripted fixture judgments) under the `cpu-diagnostic` profile and `environment.mode = "current"`. It
proves the orchestration, the stops and the recovery; it proves nothing about CUDA, MiniLM or ranking quality.

### CPU diagnostic rehearsal, exact commands / CPU 诊断演练的精确命令

For checking the procedure without a GPU (labelled **CPU diagnostic**; random tiny encoder; the real data are rebuilt from the pinned
sources, so it needs `git` and GitHub access, and an interpreter with the ML packages: `pip install torch numpy transformers tokenizers safetensors`):

```bash
cd devsearch && mkdir -p /workspace/rehearsal-runtime/inbox
PYML=/path/to/python-with-the-ML-packages            # in this mode gpu.py runs everything with the interpreter that starts it
cp /path/to/devsearch-dataset-v1.tar.gz /workspace/rehearsal-runtime/inbox/
$PYML scripts/gpu.py standin-model --package /workspace/rehearsal-runtime/inbox/devsearch-dataset-v1.tar.gz --output /workspace/rehearsal-runtime/standin-model
C=configs/gpu-cpu-rehearsal.toml
$PYML scripts/gpu.py check    --config $C
$PYML scripts/gpu.py prepare  --config $C     # exit 3, PREREQUISITE review-bank: no judgments yet; nothing is sealed
$PYML scripts/gpu.py run      --config $C     # exit 3, PREREQUISITE prepare: formal evaluation cannot start
$PYML scripts/gpu.py rehearsal-judge --config $C   # CPU rehearsal only: two SCRIPTED judgment passes (not a review); refused on a GPU profile
# (to test the gate first, import only pass A by hand: `$PYML scripts/gpu.py cmd --config $C -- review import ...`, docs/15-review-workflow.md)
$PYML scripts/gpu.py prepare --config $C && $PYML scripts/gpu.py diagnose --config $C && $PYML scripts/gpu.py run --config $C
```
