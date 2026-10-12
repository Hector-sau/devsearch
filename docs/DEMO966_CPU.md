# 966-update local CPU demo / 966 次更新本地 CPU 演示

**2026-10-12 completed:** 966-update seed42 plus fixed rank fusion is the owner-authorized local demo default. The second actual CPU acceptance attempt passed all 80 checks, the independent actual-receipt audit passed all 36 checks, and the external manager/API/UI activation passed real API and browser checks. Owner controls are preserved externally with file-level SHA-256 verification. [Acceptance summary](../reports/evidence/demo966-cpu-20261012/acceptance-summary.json) · [actual activation summary](../reports/evidence/demo966-cpu-20261012/demo-activation-summary.json).

**2026-10-12 已完成：**966 次更新的 seed42 加固定排名融合已成为所有者授权的本地演示默认。第二次真实 CPU 验收通过全部 80 项检查，独立实际回执审核通过全部 36 项检查，外置管理器／接口／界面已通过真实接口及浏览器激活检查。所有者工具已保存外置盘，逐文件 SHA-256 核验通过；准确实测范围见上述验收及实际激活回执。

This is a local demo integration, not a new public release or a replacement for historical evidence. Published `v1.1.0` still describes its evaluated BM25 default and original R5 candidate. The local service package is provisional `1.1.1.dev0`, installed separately; the immutable model release retains `candidate` status and requires explicit `--allow-candidate` authorization.

这是本地演示接入，不是新的公开发布，也不替代历史证据。已发布 `v1.1.0` 仍记录已评测的 BM25 默认及原 R5 候选。本地服务包为单独安装的暂定版 `1.1.1.dev0`；不可变模型发布包保留 `candidate` 状态，须显式使用 `--allow-candidate`。

## Model and evaluation scope / 模型与评测范围

The model is **seed42, endpoint2, 966 optimizer updates**. It is distinct from original R5 seed42/epoch3 and from the original R5 three-seed family. The 483- and 966-update comparisons currently use one training seed each; bootstrap uncertainty over questions does not measure uncertainty over training seeds. Do not copy the original R5 three-seed mean onto the new 966 model.

模型是 **seed42、第二个端点、966 次优化器更新**，与原 R5 的 seed42／第三轮以及原 R5 三种子系列区分。目前 483 与 966 次更新对照各使用一个训练种子；问题 Bootstrap 区间不测量训练种子变动的不确定性，不能把原 R5 三种子均值归给新 966 模型。

The expanded comparison preserves the original 202-question benchmark separately and evaluates 47 accepted new synthetic development questions together with the earlier 34-question development set. All compared systems share the same questions, natural candidates and graded labels. These are agent-authored questions and agent-reviewed labels inside the existing indexed corpus, not human-user testing or unseen repositories. Historical test and 59-question holdout results remain unchanged; neither set is used to tune this demo or to measure its interface latency.

扩展对照单独保留原 202 题基准，并用 47 个新增合成开发问题及此前 34 个开发问题评测。各系统使用相同问题、自然候选和分级标签。这是既有索引语料内的 Agent 问题及 Agent 标签，不是真人测试或新仓库泛化。历史测试与 59 题留出集结果保持不变，不用它们调参或测量本演示接口延迟。

| Serving identity / 服务身份 | Fixed value / 固定值 |
| --- | --- |
| Model / 模型 | `model-80c0fb327a6479b6614633195368b72b` |
| Native checkpoint SHA-256 / 原生断点 | `eb80768745196aaaeabb4975a8c6358a0fca3a693962e71487d8c7ea8133a03f` |
| Candidate release / 候选发布 | `release-03090398414e5e3c4a860a435edd883c` |
| Retrieval / 召回 | Natural BM25 top 50; no positive injection / 自然前 50，不插入正例 |
| Neural scoring / 神经评分 | MiniLM cross-encoder, CPU FP32, batch 8, four Torch threads / CPU FP32，批次 8，四线程 |
| Encoding / 编码 | Pair 256 / query 64 tokens; no silent truncation / 组合 256／问题 64 token，不静默截断 |
| Fixed fusion / 固定融合 | `(1-0.5)/(60+bm25_rank) + 0.5/(60+neural_rank)`; one-based ranks, chunk-ID tie-break / 排名从 1 开始，同分按段落 ID |
| API/UI / 接口界面 | Loopback `127.0.0.1:8820` / `127.0.0.1:8821` / 仅本机 |
| Request budget / 请求时限 | 2.0 seconds plus 1.0 second HTTP grace / 2 秒及 HTTP 额外 1 秒 |

The API's omitted `mode` uses `rerank`; the dedicated demo UI initially selects `rerank`. Explicit `baseline` returns BM25, and `compare` shows BM25 and fused rankings from the same retrieved pool. Raw logits and fusion ordering scores remain separately labeled.

API 未填写 `mode` 时使用 `rerank`；专用演示界面初始选择也是 `rerank`。显式 `baseline` 返回 BM25，`compare` 展示同一召回池的 BM25 和融合排序；原始 logit 与融合排序分数分别标注。

## Owner-held inputs and controls / 所有者资产与启动工具

A public source clone does not include trained weights, private evaluation questions, labels, corpus snapshots or personal runtime paths. The owner-held local controls include `build_release.py`, `launch_demo.py`, `demo_ui.py`, `manage_demo.py`, configuration files and optional `Start Demo.command` / `Stop Demo.command`. They require the verified external runtime, installed CPU dependencies and accepted model/corpus/index bundle; no download or training happens during search.

公开源码克隆不包含训练权重、私有评测问题、标签、语料快照或个人运行路径。所有者本地工具包括上述构建、启动、界面、管理脚本及配置，可选双击启动／停止文件。它们要求已核验的外置运行目录、已安装 CPU 依赖及通过验收的模型／语料／索引包；检索时不下载或训练。

Use the existing CPU environment's Python. Replace the placeholders with owner-held locations; keep paths quoted.

使用已有 CPU 环境的 Python，将占位符换成所有者路径，并保留路径引号。

```bash
"<CPU environment>/bin/python" "<owner controls>/launch_demo.py" check \
  --config "<owner controls>/config/demo.json"

"<CPU environment>/bin/python" "<owner controls>/manage_demo.py" start \
  --config "<owner controls>/config/demo.json" \
  --acceptance "<accepted activation receipt>.json"

"<CPU environment>/bin/python" "<owner controls>/manage_demo.py" status \
  --config "<owner controls>/config/demo.json"

"<CPU environment>/bin/python" "<owner controls>/manage_demo.py" stop \
  --config "<owner controls>/config/demo.json"
```

`check` loads and verifies without binding a port. `start` first verifies the exact release, installed service source and acceptance receipt; then starts the API, checks exact release readiness, and starts the UI. Logs and PID state stay in the external runtime. `status` is read-only. `stop` signals only the recorded process whose PID, creation time and full command still match, then preserves every artifact. If another program occupies the ports, startup refuses to replace it. Stop all demo controls before requesting external-drive ejection.

`check` 加载核验但不监听端口。`start` 先验证发布包、已安装服务源码和验收回执，再启动 API，检查准确发布就绪后启动界面。日志和 PID 状态保存在外置运行目录。`status` 只读；`stop` 仅向 PID、创建时间和完整命令仍匹配的已记录进程发信号，保留所有产物。其他程序占用端口时拒绝替换；退出外置盘前先停止演示工具。

The owner controls copy contains 32 files / 527,219 bytes, with every copied file verified and recorded in the owner-held preservation receipt. The active-demo pointer binds the accepted release, service source and acceptance receipt. The completed external-manager startup and actual API/browser checks are separately bound by the activation receipt; recorded process birth and command identities were verified.

所有者工具副本包含 32 个文件／527,219 字节，每个文件的复制核验记录在所有者持有的保存回执中。活动演示指针绑定已验收发布包、服务源码和验收回执。外置管理器实际启动及真实接口／浏览器检查另由激活回执绑定，已核对所记录进程的创建时间和命令身份。

### Replaying the copied acceptance harness / 重跑已复制验收脚本

Keep the accepted `accept_cpu_http.py` bytes unchanged. After relocation, its default sibling inputs path still names the original workspace layout. Pass the preserved evaluation root explicitly, and choose a new output directory inside the verified runtime. Existing attempt receipts remain immutable. Use the accepted installed service package in `PYTHONPATH` and the existing CPU interpreter.

保留已接受 `accept_cpu_http.py` 的原始字节。脚本迁移后，默认同级输入路径仍指向原工作区结构；必须显式传入已保存评测根目录，并在核验后的运行根目录内选择新的输出目录，保留已有运行回执不可变。`PYTHONPATH` 指向已接受的安装服务包，解释器使用已有 CPU 环境。

```bash
PYTHONPATH="<runtime>/environments/package-demo966-20261012" \
"<CPU environment>/bin/python" "<owner controls>/accept_cpu_http.py" \
  --runtime-root "<runtime>" \
  --release "<runtime>/runtime/demo966-fusion-candidate-cpu-20261012" \
  --config "<owner controls>/config/serve-cpu.toml" \
  --inputs "<runtime>/experiments/expanded-evaluation-20261012/evaluation" \
  --output "<runtime>/runtime/<new acceptance attempt directory>"
```

The mixed-schema helper tests also depend on their declared input fixture location. An external controls copy with the old default sibling path is not evidence that those tests ran there; record explicit input bindings and the actual executed command. The completed original tests keep their executed source/input identities.

混合结构辅助测试同样依赖声明的输入样例位置。外置工具副本仍使用旧默认同级路径时，不能据此宣称测试在外置副本中运行过；应记录显式输入绑定和实际命令。已完成的原测试保留实际执行源码／输入身份。

## Failure behavior and acceptance / 异常行为与验收

- A deadline timeout, including a deadline crossed by the final model batch, discards all neural results and returns complete BM25 results with an explicit timeout warning. The timeout alone does not latch rerank readiness off; a later ordinary request may succeed.
  超时包括最后一批评分越过时限的情况：丢弃全部神经结果，带明确超时提示返回完整 BM25。单纯超时不锁死重排就绪，后续正常请求可以恢复。
- A model exception or nonfinite scores discard partial output, return labeled BM25 fallback and latch rerank readiness off. An explicit successful self-check or restart is required before neural scoring becomes ready again.
  模型异常或非有限分数丢弃部分输出，返回标注清楚的 BM25，并将重排就绪锁为关闭；显式成功自检或重启后才恢复。
- Corrupt model/index/corpus hashes, failed parity or failed startup smoke block startup. A damaged model candidate is never silently activated as a baseline release.
  模型／索引／语料哈希损坏、一致性或启动重放失败会阻止启动，不静默把损坏模型候选改为基线发布。
- Measured latency covers the real loopback API path under the declared warm sequential workload. Cold load, internet transit and concurrent production load are separate scopes. A historical 622 ms R5 result must not be reused as a 966 measurement.
  延迟按声明的预热串行负载测量真实本机接口，冷加载、互联网传输和并发生产负载属于其他范围；不能把历史 R5 的 622 毫秒结果当作 966 实测值。

Engineering acceptance and development ranking improvement are separate evidence. Activation requires the CPU receipt to pass; even after activation, report the single-seed and development-set limitations. No additional GPU training, rental or paid API is authorized by this integration.

工程验收与开发集排序提升是不同证据。激活要求 CPU 回执通过；激活后也保留单一种子及开发集范围说明。本次接入不授权新增 GPU 训练、租机或付费 API。

## Recorded CPU engineering result / 已记录 CPU 工程结果

The completed second attempt used Apple M4, 16 GiB visible host RAM, CPU FP32, four Torch threads and inference batch 8. After ten excluded warmups, 150 sequential real TCP requests retrieved the natural 50 pool and returned top 10 results. There were zero HTTP errors and zero degraded measured requests. The client-wall nearest-rank p95 point target was 2000 ms.

第二次完成运行使用 Apple M4、可见宿主机内存 16 GiB、CPU FP32、四个 Torch 线程和推理批次 8。排除十次预热后，150 次串行真实 TCP 请求召回自然前 50 并返回前 10；实测请求无 HTTP 错误、无降级。客户端完整耗时按最近排名计算 p95，点值目标为 2000 毫秒。

| Attempt / 运行 | Measured requests / 测量请求 | Median / 中位数 | p95 | Overall evidence status / 整体验收状态 |
| --- | ---: | ---: | ---: | --- |
| First / 第一次 | 150 | 1396.083 ms | 1577.089 ms | Latency sample valid; full run failed on mixed saved-schema handling / 延迟样本有效，完整运行在混合保存结构处理处失败 |
| Second / 第二次 | 150 | 581.042 ms | 635.736 ms | 80/80 checks pass; maximum 840.199 ms / 80 项检查通过，最大 840.199 毫秒 |

Both latency samples remain in the evidence. Their difference is not a demonstrated speedup or a stable SLA; the first run's failed overall acceptance is not hidden. Cold process-to-TCP startup in the second attempt was 58.978 seconds, including artifact verification, recorded smoke and model parity; `start_service` alone was 49.972 seconds. Observed process peak RSS was about 1.036 GB. Cold startup is excluded from the warm request table.

两次延迟样本均保留为证据，二者差异不代表已证明提速或稳定 SLA，不隐藏第一次整体验收失败。第二次从进程启动到 TCP 监听为 58.978 秒，包含产物核验、已记录重放及模型一致性；其中 `start_service` 为 49.972 秒。进程实测峰值 RSS 约为 1.036 GB；预热请求表不包含冷启动。

All 81 saved development pools and exact fused orders matched the real API. Maximum absolute score gaps were BM25=0, logits=0.0000151694 and fusion=0.0000515358, all below the declared 0.0001 tolerance. Saved passage-text hashes exist for the earlier 34 questions / 1700 candidates; the newer 47 pools have no separate saved passage hashes and instead remain bound by the immutable release's corpus hashes. Candidate-set identifier recipes differ between mining and serving, so parity checks complete ordered candidate IDs, ranks and scores rather than requiring those identifier strings to be identical.

81 个保存的开发候选池及准确融合顺序均与真实接口一致；最大分数绝对差为 BM25=0、logit=0.0000151694、融合=0.0000515358，均小于预定 0.0001 容差。此前 34 题／1700 个候选有保存的段落文本哈希；新增 47 题的候选池没有单独保存段落哈希，继续由不可变发布包的语料哈希绑定。采样与服务的候选集 ID 生成规则不同，因此核对完整有序候选 ID、排名和分数，不要求两种候选集 ID 字符串相同。

Five actual injected failure scenarios passed: model exception, nonfinite output, cooperative timeout, deadline crossed by the final scoring batch, and an occupied worker beyond the HTTP deadline. Each discarded inappropriate partial results and recovered under its declared rule. Missing or corrupt required heads refused startup; a separate explicitly model-free BM25 release passed. These are engineering fault injections and agent-operated measurements, not human traffic or new ranking-quality evaluations.

五项真实故障注入通过：模型异常、非有限输出、协作超时、最后评分批次越过时限，以及工作线程超过 HTTP 时限仍占用。各项均丢弃不应发布的部分结果，并按声明规则恢复。必需分类头缺失或损坏时拒绝启动；单独显式无模型的 BM25 发布包通过。这是工程故障注入及 Agent 测量，不是真人流量，也不是新的排序效果评测。

The independent actual-receipt audit passes 36 checks. Its full-file SHA-256 is `0627dce19c1c63f67cfed66ce8aff636b957fc531afac81d84555bd857d85604`; the sanitized acceptance summary binds that audit to the actual run. Separate manager/API/UI activation has also completed.

独立实际回执审核通过 36 项检查，完整文件 SHA-256 如上；脱敏验收概要将该审核与实际运行绑定。独立于该审核的管理器／接口／界面激活也已完成。

## Actual browser demonstration / 实际浏览器演示

The external controls started the exact accepted CPU API and Streamlit UI. Both recorded process identities matched. The browser initially selected `rerank`, displayed the single-seed42 scope, and successfully searched an authored demonstration question. `compare` displayed both ranked lists side by side; explicit `baseline` displayed the BM25 list with reranking marked as not run. Live omitted-mode, baseline and compare API requests also passed, with exact shared BM25 records and fused order. The UI was restored to `rerank` after verification.

外置工具启动了准确的已验收 CPU 接口和 Streamlit 界面，两项记录进程身份均匹配。浏览器初始选择 `rerank`，显示单一种子 42 的范围，并成功检索专门编写的演示问题。`compare` 并排展示两组排序；显式 `baseline` 展示 BM25，注明未执行重排。真实接口未指定模式、基线和对照请求也通过，两模式的 BM25 完整记录及融合顺序准确一致。核验后已将界面恢复为 `rerank`。

| Actual state / 实际状态 | Recorded capture / 截图 |
| --- | --- |
| Initial default and successful fused search / 初始默认及成功融合检索 | [demo-default.jpg](../reports/evidence/demo966-cpu-20261012/demo-default.jpg) |
| Same-pool comparison / 同一候选池对照 | [demo-compare.jpg](../reports/evidence/demo966-cpu-20261012/demo-compare.jpg) |
| Explicit BM25 mode / 显式 BM25 模式 | [demo-baseline.jpg](../reports/evidence/demo966-cpu-20261012/demo-baseline.jpg) |

The [activation summary](../reports/evidence/demo966-cpu-20261012/demo-activation-summary.json) binds these agent-operated browser/API checks to the accepted model and release. The screenshots do not establish human usability success or add new ranking-quality evidence. Published `v1.1.0` and all historical receipts remain unchanged; no additional GPU training occurred.

上述激活概要将 Agent 操作的浏览器／接口检查与已验收模型及发布包绑定。截图不证明真人可用性成功，也不增加排序质量证据。已发布 `v1.1.0` 和历史回执保持不变，本轮未追加 GPU 训练。
