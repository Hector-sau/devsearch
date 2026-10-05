# Machine handoff and execution / 机器交接与执行

**Latest decision, 2026-10-05:** the target is undecided; retain both a proposed 24 GB Apple/MPS profile and a Linux/NVIDIA CUDA profile. Actual chip/GPU, software, access and throughput remain unverified. The earlier Mac-only plan below is historical; apply its storage/access/backup principles to either host and follow [10](10-accelerator-execution-contract.md) for current device/runtime acceptance. New hardware costs remain the user's decision.

**2026-10-05 最新决定：**目标机器未定，保留拟定 24 GB Apple／MPS 与 Linux／NVIDIA CUDA 两种方案，实际芯片显卡软件访问和吞吐均待核实。下文仅 Mac 计划属于历史记录，存储访问备份原则适用于两种主机，当前设备运行验收见第 10 号文档；新增硬件费用仍由用户决定。

Decision recorded on 2026-10-05: the user will arrange a Mac mini M6 with 24 GB memory. The assistant will transfer project materials and handle implementation and execution after access is available. The machine has not yet been acquired or connected. This document records the plan, not completed deployment.

2026-10-05 决策：用户自行安排 24 GB 内存的 Mac mini M6；可访问后，助手负责项目资料迁移、开发和运行。目前机器尚未取得或接入，本文记录计划，不代表完成部署。

## Ownership / 分工

| User / 用户 | Assistant / 助手 |
| --- | --- |
| Arrange the machine, payment and any account login / 安排机器、付款及必要账号登录 | Transfer project files; recreate and verify the runtime / 迁移项目文件，重建并验证环境 |
| Provide usable access and the availability or rental end time when ready / 准备好后提供可用访问方式和使用期限或租期结束时间 | Select compatible dependencies and configure paths, caches and device use / 选择兼容依赖，配置路径、缓存与设备 |
| Connect the external SSD if the Mac is physically available / 实体机器到手时连接外置 SSD | Acquire public source material, preserve licenses and versions, clean and split data / 获取公开资料，保留许可版本，清洗并划分数据 |
| Review focused ambiguous examples and try the final demo / 集中审查少量歧义样例并体验演示 | Implement BM25, PyTorch training, checkpoints, evaluation, API and UI / 实现 BM25、PyTorch 训练、断点、评测、接口与界面 |
| Explain and independently modify a small part before claiming personal mastery / 陈述个人掌握前，独立解释和修改一小部分 | Diagnose failures, tune within resources, package results and prepare interview explanations / 排错、按资源调参、整理结果和面试讲解 |

No rental listing is required now. Once the machine is ready, provide its access method; if rented, include the end time so work and backups can be bounded. Passwords and private keys should be configured through the access mechanism, not pasted into chat. Routine dependency choices, batches, retries and checkpoints do not require repeated user decisions.

目前无需租赁链接。机器准备好后提供访问方式；如果有租期，附上结束时间，以便限定运行和备份窗口。密码和私钥通过接入机制配置，不粘贴到聊天。常规依赖选择、batch、重试及断点无需反复让用户决定。

## Compute and storage / 算力与存储

The user-provided target is a Mac mini M6 with 24 GB memory; the actual machine, installed software, and MiniLM compatibility remain unverified until access. Record the observed hardware at handoff and size experiments from measured throughput and memory. Unified memory is shared with the system, not dedicated GPU memory. The proposed configuration alone does not establish a training duration or maximum dataset size.

用户提供的目标配置为 24 GB Mac mini M6；实际机器、已安装软件及 MiniLM 兼容性，均待接入后核实。交接时记录实际硬件，依据实测吞吐与内存确定实验规模。统一内存与系统共享，不等于独立显存；不能仅凭拟定配置承诺训练时长或最大数据量。

| Delivery / 机器形式 | Runtime / 运行位置 | Backup / 备份 |
| --- | --- | --- |
| Physically available Mac mini / 本地实体 Mac mini | Prefer the SSD connected to that Mac, after mount and write checks / 优先该 Mac 实际连接且通过挂载写入检查的外置盘 | A verified separate copy of final artifacts / 最终产物保留经校验的独立副本 |
| Remotely hosted Mac mini / 远程托管 Mac mini | A verified project directory on that host's available SSD/storage / 该主机可用 SSD 或存储上的经检查项目目录 | Transfer artifacts back to the current Mac's external drive when mounted / 当前 Mac 外置盘挂载后，将产物传回 |

Use a host-specific `DEVSEARCH_RUNTIME_ROOT`; the preferred physical-drive path is `/Volumes/MAC专用/AI-Lab/pytorch/devsearch`. The current Mac's volume is not automatically mounted on a remote machine. Verify free space, permissions and the real mount before large downloads. Keep the environment, Hugging Face cache, datasets, indexes, checkpoints and logs under the chosen runtime root; do not silently redirect them to the current MacBook's internal disk.

使用每台主机单独配置的 `DEVSEARCH_RUNTIME_ROOT`；实体外置盘首选路径为 `/Volumes/MAC专用/AI-Lab/pytorch/devsearch`。当前 Mac 的卷不会自动挂载到远程机器。大规模下载前验证空间、权限与真实挂载。环境、Hugging Face 缓存、数据、索引、断点和日志均放选定运行目录，不静默改存当前 MacBook 内置盘。

A missing external drive on the current Mac does not block remote work when the remote runtime has verified durable storage within the known availability window. Record the pending off-host backup instead of claiming redundancy. Do not move heavy assets to the current Mac's internal disk as an automatic substitute. Confirm the transfer destination before the measured return deadline; a threatened sole copy requires a concrete resolution before access expires.

只要远端运行目录在已知使用窗口内具备经验证的持久存储，当前 Mac 未挂载外置盘不阻塞远端工作。明确记录异机备份待完成，不声称已有冗余副本；也不自动将大体积产物改存当前 Mac 内置盘。须在实测得出的归还截止点前确认接收位置；如果唯一副本面临过期丢失，必须在访问失效前落实具体处理。

## Private demo access / 私下访问演示

Proposed defaults are API `127.0.0.1:8000` and UI `127.0.0.1:8501` on the execution host. Both ports are configurable and must be checked for availability. The remote Streamlit process calls the remote API on the same host; the user's browser only needs the UI. The interface never loads another model.

拟定默认端口为执行主机上的 API `127.0.0.1:8000` 与 UI `127.0.0.1:8501`，均可配置，并须检查端口可用性。远端 Streamlit 进程调用同机远端 API；用户浏览器只需访问界面，界面不另行加载模型。

For a physical machine, open its local UI in its browser. For a remote machine with an already configured SSH access mechanism, the following is a proposed command on the user's Mac; replace the host alias and ports with verified values. It does not install a tool or establish credentials.

实体机器可在其浏览器打开本地界面。远端机器已有配置好的 SSH 接入机制时，可在用户 Mac 上使用以下拟定命令；主机别名和端口替换为已核实值。这不是安装工具或配置凭据的操作。

```sh
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8501:127.0.0.1:8501 <configured-host>
```

Open `http://127.0.0.1:8501` in the user's browser. Add a separate loopback forward for port 8000 only when direct API diagnosis is needed; ordinary UI use does not require it. No public listening address is needed. Record API/UI start, health check and stop commands in the runbook. M4 checks the actual browser-to-forwarded-UI-to-remote-API route, filtering and source links. Stop the forwarding process when the session ends; service shutdown follows its runbook. If existing access uses remote desktop instead, verify the same local UI there.

用户浏览器打开 `http://127.0.0.1:8501`。只有直接排查 API 时才另外转发 8000 端口，日常界面无需它；不需要公网监听地址。在运行手册记录 API／UI 启动、健康检查和停止命令。M4 验证实际“用户浏览器→转发界面→远端 API”通路、筛选与来源链接。会话结束时停止转发进程，服务按运行手册关闭；如已有接入方式为远程桌面，则在远程桌面验证本地界面。

## Backup and return window / 备份与归还窗口

Test export early: after the first diagnostic checkpoint, export a small self-contained bundle and reload fixed prediction examples in a fresh process from the export path. When a backup destination is available, repeat the hash check and reload there. Record a same-host export check separately from an independent backup; the former does not protect against loss of the host.

尽早验证导出：第一个诊断模型保存后，导出小型自包含包，在新进程中从导出目录重载并运行固定预测样例。备份位置可用后，在接收端再次校验哈希并重载。同机导出检查与独立备份分别记录，前者不能防止整机丢失。

At every completed epoch and milestone, sync changed critical artifacts to the available verified backup destination and record hashes, time and outcome. Include source/configuration changes, labels/splits/manifests, metrics/reports, tokenizer/model assets, the latest complete-epoch resumable checkpoint, and the best-validation checkpoint. Download caches need not be copied if exact sources can be reconstructed. Verify fresh loading after checkpoint-format or environment changes and at final handoff; checksum each normal sync. A pending or failed transfer remains visible and never permits deletion of the only valid copy.

每个完整 epoch 和里程碑结束时，将变更的关键产物同步到可用且经验证的备份位置，记录哈希、时间与结果。包括源码／配置变更、标签／划分／清单、指标／报告、tokenizer／模型资产、最新完整 epoch 的可续训断点，以及验证集最佳断点。可依据精确来源重建的下载缓存不必全部复制。检查点格式或环境变化后，以及最终交接时验证重载；日常同步校验哈希。待完成或失败的传输保持可见，不能因此删除唯一有效副本。

Retain the latest complete-epoch checkpoint and the best checkpoint until their replacements are verified. A periodic mid-epoch snapshot may assist diagnosis, but v1 recovery reloads the last complete-epoch state and replays the interrupted epoch; it does not claim exact continuation from that snapshot. Before epoch one, preserve an initialized resumable state or an equivalent verified deterministic restart recipe. Bound epoch size and run duration so the replay cost fits the resource window.

最新完整 epoch 断点与最佳断点，在替代产物验证前均保留。周期性的 epoch 中途快照可用于排错，但首版恢复加载最近完整 epoch 状态，并重跑中断的 epoch，不宣称从中途快照精确续接。首个 epoch 前保留初始化可恢复状态，或等效且经验证的确定性重启方案。限制 epoch 规模和运行时间，使重跑成本在资源窗口内可接受。

Measure transfer throughput with the early bundle and estimate a backup reserve from remaining artifact size, checksum/reload time, and a recorded contingency margin. Set a latest training stop time before a known rental end; do not start an epoch that cannot finish, checkpoint and export within that window. If a destination is unavailable near this cutoff, stop expansion, preserve durable remote copies, and resolve the concrete access/transfer issue before expiry. Extensions or a new paid destination need the user's decision; no renewal or deletion of the sole copy is automatic.

利用早期导出包测量传输吞吐，按剩余产物体积、校验／重载时间及明确记录的余量，估算备份预留窗口。已知租期结束时间时设置最晚停止训练时间；无法在窗口内完成、保存和导出的 epoch 不启动。接近截止点仍无接收位置时，停止扩量、保留远端持久副本，并在到期前解决具体访问或传输问题。续租或新增付费存储须由用户决定，不自动续费，也不自动删除唯一副本。

## Execution and acceptance / 执行与验收

1. Inspect the actual chip, memory, OS, access, storage and rental window. Recreate the Python environment on the target machine rather than copying an old virtual environment. Transfer only project assets, preserving source revisions and hashes.
   核实实际芯片、内存、系统、访问、存储及租期。在目标机器重建 Python 环境，不复制旧虚拟环境；仅迁移项目资产，保留来源版本与哈希。
2. Check MPS availability and actual MiniLM forward/backward/optimizer steps, then benchmark 50–100 complete steps. Start conservatively with FP32 and a small batch. Record throughput, memory pressure and checkpoint recovery before choosing larger runs. Train and serve on the accelerator at different times.
   检查 MPS 及 MiniLM 实际前向、反向和参数更新，再测量 50～100 个完整 step。先用 FP32 和小 batch；记录吞吐、内存压力与断点恢复，再确定扩大运行的规模。加速器上的训练和服务错时进行。
3. Follow M0–M5 in the architecture: usable BM25 search, training comparisons, one controlled domain experiment, integration, sealed evaluation and delivery. Prefer direct public-source downloads on the target host where that avoids redundant transfer. Preserve acquisition and label provenance.
   按架构 M0～M5 推进：可用 BM25 检索、训练对照、一项受控领域实验、集成、封存评测和交付。适合时直接在目标主机下载公开来源，减少重复传输；保留数据获取与标签来源。
4. Save bounded runs and complete-epoch restartable checkpoints, with stop times derived from the known availability window and the measured backup reserve above. Do not depend on an interactive shell remaining connected. Recovery and detached execution must be verified; continuous assistant monitoring is not assumed.
   配置有界运行与完整 epoch 的可恢复断点，按已知使用窗口和上述实测备份预留时间设定停止时间。任务不依赖交互终端持续连接；需验证脱离终端执行及恢复，不能假定助手始终在线监控。
5. Before the machine is returned, export source, dependency lock, model/tokenizer, data/index manifests, required licensed assets, configurations, metrics and reports. Verify hashes and a fresh load with fixed prediction examples on the destination; recreate its environment there. Resume acceptance checks optimizer state on the selected device. Do not delete the only copy of deliverables.
   退还机器前导出源码、依赖锁、模型与 tokenizer、数据和索引清单、必要且许可允许的资产、配置、指标及报告。校验哈希，并在目标位置重建环境后重载固定样本预测；续训验收检查优化器状态所在设备。不删除交付物的唯一副本。

Record `engineering_status` and `learning_evidence_status` separately. User exercises do not block otherwise authorized engineering milestones. Engineering completion means an independently loadable project, verified search and training recovery, an evaluation report and a working demo. Personal PyTorch mastery claims remain pending until the user's independent explanation and modification are verified. Human-review and simulated-evaluation labels remain accurate. A failed improvement hypothesis is reported honestly. Rental extensions, additional paid services and public publication remain user decisions; ordinary technical execution is delegated.

分别记录 `engineering_status` 与 `learning_evidence_status`，用户练习不阻塞其他已获授权的工程阶段。工程完成意味着项目可独立加载、检索与训练恢复经过验证，并交付评测报告和可运行演示。本人掌握 PyTorch 的陈述，须等用户独立解释与修改经过验证后才成立。人工审查与模拟评测如实标记；没有改善的实验如实报告。续租、新增付费服务和公开发布仍由用户决定，日常技术执行已交给助手。
