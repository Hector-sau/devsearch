# Local 966 CPU demo evidence / 本地966 CPU演示证据

The owner authorized the fixed 966-update seed42 endpoint plus rank fusion 0.5/k60 as the local demo default after CPU model-loading, HTTP latency and failure-fallback acceptance. The installed service is `1.1.1.dev0`; published `v1.1.0` and historical evaluation decisions remain unchanged. [Operator guide](../../../docs/DEMO966_CPU.md).

所有者授权固定的 966 次更新 seed42 模型及融合 0.5／60，在 CPU 模型加载、HTTP 延迟和故障降级验收后作为本地演示默认。已安装服务为 `1.1.1.dev0`；已发布 `v1.1.0` 及历史评测决策保持不变。操作文档见上述链接。

| Evidence / 证据 | File / 文件 | Scope / 范围 |
| --- | --- | --- |
| Development quality / 开发集效果 | [development-comparison-summary.json](development-comparison-summary.json) | All seven systems and five cohorts, with question-group intervals; single training seed42 / 七系统五分组及问题组区间，仅一个训练种子 |
| CPU service acceptance / CPU服务验收 | [acceptance-summary.json](acceptance-summary.json) | 80 actual checks and 36 independent checks; source/model/receipt hashes / 80项实测及36项独立检查，代码模型回执哈希 |
| Actual local activation / 实际本地激活 | [demo-activation-summary.json](demo-activation-summary.json) | Default/no-mode, baseline and compare API requests against the owned running service / 实际服务中的默认、基线及对照请求 |
| Browser behavior / 浏览器行为 | [Default](demo-default.jpg) · [Compare](demo-compare.jpg) · [Baseline](demo-baseline.jpg) | Actual agent-operated browser, authored interface query, no human-user study or relevance scoring / Agent操作真实浏览器，界面演示问题，无真人研究或相关性计分 |

The successful CPU run has 150 serial real-loopback-TCP requests, ten excluded warmups, natural50 candidates and requested top10 results: median **581.042 ms**, p95 **635.736 ms**, maximum **840.199 ms**, zero errors/degradations. The earlier run's same timed workload had p95 **1577.089 ms**, but full acceptance stopped on a harness metadata-schema error; its raw receipts are preserved. Both timed phases pass the declared 2000 ms demo target. These observations are not a production SLA or proof of invariant performance. Cold process-to-TCP readiness is separately **58.978 s**, including artifact verification and startup replay.

成功 CPU 运行包含 150 次串行真实本机 TCP 请求、十次不计入统计的预热、自然前50候选及请求前10结果：中位数 **581.042毫秒**、p95 **635.736毫秒**、最大 **840.199毫秒**，零错误降级。较早运行的同一计时负载 p95 为 **1577.089毫秒**，完整验收在脚本元数据兼容错误处停止，原始回执保留。两次计时阶段均通过声明的 2000毫秒演示目标，不能视为生产 SLA 或恒定性能证明。冷进程到接口就绪另外记录为 **58.978秒**，含产物核验及启动重放。

All81 complete candidate pools and fused orders agree with frozen CPU predictions; numeric gaps are within the predeclared `1e-4` tolerance. Direct frozen passage hashes exist for the original34 pools (1700 pairs), while release/corpus hashes bind the full corpus. Five controlled faults prove complete BM25 fallback or explicit busy/deadline errors, readiness transitions and recovery; missing/corrupt required weights refuse startup. A separate model-free baseline also starts and serves successfully. Private full HTTP traces and runtime paths stay outside Git.

全部81题完整候选池及融合顺序与固定 CPU 预测一致，数值误差位于预先声明的 `1e-4` 容差内。直接冻结段落哈希覆盖原34题候选池（1700对），发布／语料哈希关联完整语料。五类受控故障验证完整 BM25 降级或明确忙碌／超时错误、就绪变化及恢复；必需权重缺失或损坏时拒绝启动，单独无模型基线也可启动服务。私有完整 HTTP 记录及运行路径保存在 Git 外。

The 483- and 966-update endpoints have only training seed42. Question-bootstrap intervals do not replace matched seed43/44 training. No additional GPU training, rental, paid API, test/holdout tuning or change to sealed grades occurred during this integration.

483 与966次更新模型只有训练种子42；问题 Bootstrap 区间不能替代相同配置的种子43／44训练。本次接入未追加GPU训练、租机、付费API、测试／留出集调参，未改变封存标签。
