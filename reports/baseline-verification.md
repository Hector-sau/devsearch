# Baseline verification / 基线验证记录

Date: 2026-10-05. Scope: the first implementation slice, **T01 (offline CPU foundation) and the fixture-level part of T04 (model-free BM25)**: strict configuration, corpus/source-map/index artifacts, deterministic IDs, code identity, the `lexical-v1` tokenizer, BM25 with filters/zero-match/ties, and the `index`, `retrieve --mode baseline` and `validate_manifest` commands. All data are the tiny original fixtures in `tests/fixtures/mini_corpus` (6 documents, 20 chunks, placeholder `fixture-*` sources). These checks establish that the **code behaves as specified on fixtures**. They establish no retrieval quality on real documentation, no training, no MPS/hardware result and no performance figure; no number below may be quoted as a project metric.

日期：2026-10-05。范围：首个实现切片，**T01（离线 CPU 基础）及 T04 中样例级部分（无模型 BM25）**：严格配置、语料／来源映射／索引产物、确定性 ID、代码身份、`lexical-v1` 分词器、带筛选／无匹配／同分处理的 BM25，以及 `index`、`retrieve --mode baseline`、`validate_manifest` 命令。全部数据为 `tests/fixtures/mini_corpus` 中的微型原创样例（6 篇文档、20 个段落、占位的 `fixture-*` 来源）。这些检查只证明**代码在样例上符合规格**，不证明真实文档上的检索效果、训练、MPS／硬件结果或性能数字；下列数字均不得引用为项目指标。

## Environment / 环境

| Item / 项 | Value / 值 |
| --- | --- |
| Host / 主机 | Authorized cloud development container, Linux x86_64 (not the target Mac mini) / 获授权的云端开发容器，Linux x86_64（不是目标 Mac mini） |
| Interpreters / 解释器 | System Python 3.11.15 and 3.12.3 (3.11 is the declared minimum) / 系统 Python 3.11.15 与 3.12.3（3.11 为声明的最低版本） |
| Dependencies / 依赖 | None beyond the standard library; setuptools 68.1.2 and wheel 0.42.0 used for the offline wheel build / 仅标准库；离线构建 wheel 用 setuptools 68.1.2、wheel 0.42.0 |
| Network / 网络 | Not used by any check; the audit-hook test fails on any socket or subprocess in the baseline commands / 所有检查均未使用；审计钩子测试在基线命令出现 socket 或子进程时失败 |
| Storage / 存储 | Temporary directories and the session scratch directory; no `DEVSEARCH_RUNTIME_ROOT` volume, no model/data download / 临时目录和会话暂存目录；无 `DEVSEARCH_RUNTIME_ROOT` 卷，无模型或数据下载 |
| Code identity / 代码身份 | `source_snapshot_sha256` = `f5d46b9f956020968029d67c36e9c0e7fbe73922af967ce885217793b29f382c` over 33 `src/devsearch/**/*.py` files (`devsearch.artifacts.identity.code_identity()`); the Git commit is the branch head that adds this record / 33 个 `src/devsearch/**/*.py` 文件的快照哈希；Git 提交为加入本记录的分支头 |

## Checks run / 已执行检查

| # | Command / 命令 | Result / 结果 | Limit / 边界 |
| --- | --- | --- | --- |
| 1 | `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3.11 -m unittest discover -s tests` | **Pass**: 125 tests, 0 failures, 0 skips, 5.5 s | Fixture-level; Linux only / 样例级，仅 Linux |
| 2 | Same command with `python3.12` | **Pass**: 125 tests, 0 failures, 0 skips, 6.1 s | Python 3.13+ not tested / 未测 3.13 及以上 |
| 3 | Mutation check: 22 deliberately planted defects (sign-gated eligibility, reversed tie-break, ascending sort, ignored/inverted filters, shifted IDF, skipped chunk-slice / chunk-ID / manifest-hash / containment / postings-sum checks, `http` URLs, dropped identifier components, stopwords kept, skipped tokenizer-fingerprint check, overwriting a different index, `index_id` ignoring BM25 parameters, unenforced `top_k`, empty query allowed, BM25 ranks copied into rerank fields, bool accepted as int, synthetic text reported as source) applied one at a time to a scratch copy of `src/` and run against the in-process suites | **22 of 22 caught**, 0 survived. A separate direct computation confirmed that recomputing statistics over the filtered subset would change the scores the filter test compares (e.g. 3.697 vs 1.612) | One-off script in the session scratch directory, not committed; shows the suite can fail, not that it is complete / 一次性脚本，未提交；说明测试能失败，不代表完备 |
| 4 | `pip wheel . --no-deps --no-build-isolation --no-index` in a clean copy of the non-test sources | **Built** `devsearch-0.0.0-py3-none-any.whl`, 46,411 bytes, SHA-256 `a0c2b5b788799340494416f2d342f8e73cc036824d6d4172c31ec3fe8c2c574a` (37 entries, no tests/fixtures/data; zip timestamps make the hash non-reproducible between builds) | Hash identifies the checked artifact only / 哈希只标识被检查的产物 |
| 5 | `python scripts/check_wheel.py <wheel>` with 3.11 and with 3.12 | **Pass** on both: isolated venv, `--no-index --no-deps` install, `-I` mode, working directory outside the checkout; imports without torch/transformers/numpy/fastapi/streamlit; `index` built a 20-chunk index, `retrieve` returned `Checkpoints > Loading` first with `docs/checkpoints.md` lines 8–11, a no-match query returned `empty`, `validate_manifest` reported `valid`; `retrieve --mode rerank` and the seven stub commands exited 2 | Proves package portability and CPU baseline behavior on fixtures; fixture and configs are passed as data files from the checkout / 证明包的可移植性和样例上的 CPU 基线行为 |
| 6 | README quick start, run verbatim with a scratch directory as `DEVSEARCH_RUNTIME_ROOT` | **Pass**: `built`, five results led by `Checkpoints > Loading`, `valid` | Illustration only: ranking order on 20 fixture chunks is not a quality measure / 仅为演示 |
| 7 | GitHub Actions workflow `CPU baseline checks` (`scaffold-check.yml`), run [37307966474](https://github.com/Hector-sau/devsearch/actions/runs/37307966474) dispatched manually on head `fda9eea` | **Pass**: both matrix jobs (Python 3.11, 3.12) concluded `success`; every step succeeded: unit tests, `build` wheel, installed-wheel check | Linux `ubuntu-latest` only; I read the job/step conclusions, not the full logs; it is the same CPU fixture-level check as rows 1–5, not model or MPS evidence / 仅 Linux；读取的是作业与步骤结论而非完整日志；与第 1–5 行同属 CPU 样例级检查，不是模型或 MPS 证据 |
| – | MiniLM / real tokenizer / MPS forward-backward / real documentation / latency or throughput | **Not run, not implemented** | These remain pending; a skipped hardware check is not a pass / 仍待执行，跳过不算通过 |

### What the tests cover / 测试覆盖

125 tests: artifacts and paths 17, BM25 23, CLI end-to-end 16, config 5, corpus 21, index store 18, scaffold boundaries 6, search/response 12, tokenizer 7. They include hand-computed BM25 scores against an independent restatement of the formula; eligibility independent of score sign (an exact-zero score and all-negative scores under the `robertson` IDF variant); no zero-padding for non-overlapping queries; global statistics unchanged by filters; deterministic `chunk_id` tie-breaking; identifier queries (`load_state_dict`, `AutoModelForSequenceClassification`, `<unk>`, `torch.no_grad()`); invalid/contradictory/unknown filters rejected while a valid filter with no lexical match returns `empty`; strict rejection of unknown/missing fields, bad hashes, wrong or duplicate IDs, dangling references, chunk text differing from its slice, path traversal and symlink escapes, unsafe URLs, NaN, duplicate JSON keys; byte-identical rebuilds, idempotent re-index, refusal to overwrite a different index, tamper detection and a deep rebuild comparison; that a missing output directory or runtime root is never created; and that the baseline commands open no socket and spawn no process.

## Contract changes / 契约变化

Code, owning documentation and tests change together (docs/02, 03, 04, 06, 08, 09, `configs/README.md`, README). / 代码、所属文档与测试一并修改。

- **Config format is TOML** (stdlib `tomllib`), replacing the placeholder `.yaml` file names throughout docs/03 and `configs/README.md`; reason: zero dependencies and an offline-verifiable wheel. Decision-log row added in docs/04. / 配置格式改为 TOML，取代占位的 `.yaml`。
- **Corpus artifact v1, deterministic `chunk_id`, source-map v1** (line and notebook-cell forms, synthetic marking), canonical record order, HTTPS-or-null `source_url`: specified in docs/02 §2 (the earlier text left the source-map schema and ID recipe open). / 语料产物 v1、确定性 `chunk_id`、来源映射 v1：写入 docs/02 §2。
- **`lexical-v1` tokenizer and BM25 definition** (Lucene or Robertson IDF, binary query weights, score rounding, tie-break), **filter semantics**, **baseline response subset** and `candidate_set_id`: docs/04. / 分词与 BM25 定义、筛选语义、基线响应子集及 `candidate_set_id`：docs/04。
- **CLI**: `index`, `retrieve --mode baseline` and `validate_manifest` implemented; added optional `--runtime-root` (default `DEVSEARCH_RUNTIME_ROOT`, then the manifest directory); exit status 0/1/2 defined; `retrieve` results are never overwritten. The seven other commands keep their fail-closed stubs (`_cli.STUB_COMMANDS`). / 三个命令已实现；新增可选 `--runtime-root`；定义退出状态 0/1/2；其余七个命令保持占位。
- **Package layering rule** is now an enforced test (replacing the temporary stdlib-only assertion). / 包分层规则现由测试强制。
- **CI**: workflow now also runs on pull requests and covers Python 3.11 and 3.12. / 工作流新增 pull request 触发并覆盖 3.11 与 3.12。

## Deviations from the instructions / 与指示的偏差

- **Branch name:** work is on `claude/youthful-turing-uc1lkv`, the branch this session was assigned, not a `codex/` branch (AGENTS.md allows another convention chosen by the user). / 工作在会话指定的分支，而非 `codex/` 分支。
- **No pull request has been opened**; the branch is pushed for review. / 未创建 pull request，分支已推送待审。
- BM25 `k1=1.2`, `b=0.75`, the stopword list and "no stemming" are **untuned starting choices**; their effect on real documentation is unmeasured (T04). / 这些均为未调优的起始选择，对真实文档的效果未测量。

## Known limitations / 已知局限

- No stemming: `resume` does not match `resuming`. Whether to add it is a validation-set decision on the real corpus.
- `max_query_chars` is a character guard; the token-based `max_query_tokens` needs the pinned real tokenizer (T03).
- Fixture text is not real documentation; license, revision and URL fields are placeholders (`.invalid` domain). Nothing here reflects PyTorch or Transformers content.
- Linux, Python 3.11/3.12 only; macOS/Windows and the target Mac mini are unchecked. Index postings are plain JSON, sized for fixtures; the real corpus (T03/T04) may require measuring load time and a denser format.
- The index is rebuilt in memory on load (`Bm25Index`), and the filter mask is computed per query (O(chunks)); both are unprofiled and unneeded at fixture scale.

## Task status (docs/06 §5 schema) / 任务状态

```json
[
  {
    "task_id": "T01",
    "engineering_status": "running",
    "learning_evidence_status": "not_applicable",
    "input_manifest_ids": ["tests/fixtures/mini_corpus (fixture only)"],
    "output_manifest_ids": ["source_snapshot_sha256 f5d46b9f956020968029d67c36e9c0e7fbe73922af967ce885217793b29f382c"],
    "check_results": "Pass: 125 unit tests on Python 3.11 and 3.12; mutation check 22/22; installed-wheel check on both. GitHub Actions run 37307966474: both jobs succeeded.",
    "open_issues": ["Real-source gates belong to T03", "Dependency lock unresolved (no dependencies yet)", "Training/serving schemas and validators still to come with their tasks"],
    "next_action": "Land this slice after review, then start T03 with a verified DEVSEARCH_RUNTIME_ROOT"
  },
  {
    "task_id": "T04",
    "engineering_status": "running",
    "learning_evidence_status": "not_applicable",
    "input_manifest_ids": ["tests/fixtures/mini_corpus (fixture only)"],
    "output_manifest_ids": ["fixture index built on demand; no real-corpus index exists"],
    "check_results": "Pass at fixture level: filters, identifier queries, zero-match, source mapping and links, deterministic ties. Not run: real sources, 30-50 development queries, lexical-miss analysis.",
    "open_issues": ["Needs T03 real sources and the pinned real CPU tokenizer", "Development queries and `evaluate --mode baseline` not started"],
    "next_action": "After T03: build the real index, write 30-50 reviewed development queries, implement baseline evaluation"
  }
]
```

## GitHub workflow status / GitHub 工作流状态

Run [37307966474](https://github.com/Hector-sau/devsearch/actions/runs/37307966474) (`workflow_dispatch`, branch `claude/youthful-turing-uc1lkv`, head `fda9eea`, started 2026-10-05 12:12 UTC, about 30 s) finished with both `cpu-baseline (3.11)` and `cpu-baseline (3.12)` = `success`. This is the first time the workflow has run on GitHub; it also confirms that the previously untested action versions in the file (`actions/checkout@v6`, `actions/setup-python@v5`) resolve. The `pull_request` trigger has not been exercised because no pull request exists yet. This record was committed after that run, on top of `fda9eea`; the later commit changes only documentation.

运行 [37307966474](https://github.com/Hector-sau/devsearch/actions/runs/37307966474)（`workflow_dispatch`，分支 `claude/youthful-turing-uc1lkv`，头提交 `fda9eea`，2026-10-05 12:12 UTC 启动，约 30 秒）结束，`cpu-baseline (3.11)` 与 `cpu-baseline (3.12)` 均为 `success`。这是该工作流首次在 GitHub 上运行，也确认了文件中此前未经测试的 action 版本（`actions/checkout@v6`、`actions/setup-python@v5`）可正常解析。因尚无 pull request，`pull_request` 触发器未被实际执行。本记录在该次运行之后提交于 `fda9eea` 之上，后续提交只改动文档。
