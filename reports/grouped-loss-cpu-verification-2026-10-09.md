# Grouped training CPU verification / 分组训练 CPU 验证

**Outcome:** both optional native-PyTorch objectives are implemented and pass targeted offline CPU and installed-wheel tests. No new model-quality result or GPU-readiness claim is made. Changes are local and uncommitted on top of `e3168ca6fef9b64ea0688ea059332c729125e30f`.

**结论：**两种可选原生 PyTorch 损失已实现，通过离线 CPU 专项测试及安装包测试。本次没有新增排序效果结果，也未宣称 GPU 就绪；改动保存在上述提交之上的本地工作区，尚未提交。

## Change and rationale / 改动与原因

- Preserve query/pair IDs and verified train ownership; run one forward per five-record query, with three hard and one random negative. This makes the BCE versus pairwise comparison match exposures and updates.
  保留问题／样本编号及已验证的训练归属，每题五条记录仅一次前向，包含三难一随机负例，使 BCE 与排序损失对照的曝光和更新次数一致。
- Add explicit `group_bce_weight4` and `pairwise_logistic_scale_4_over_5`; normalize a short final update by its actual query count. Fix the selected checkpoint to epoch three.
  增加两种显式损失，尾批按实际题数归一化，模型固定选择第三轮。
- Bind objectives, source hashes and grouped protocol versions into checkpoint identity; reject changed-input resume. Keep default flat BCE serialization and behavior.
  断点绑定目标、源哈希和分组协议版本，拒绝改变输入后续训；保持原平铺 BCE 序列化及行为。
- Allow a separately hash-pinned reviewed pair manifest for grouped jobs, verified against the prepared corpus, index and training query/qrel hashes. R6 reviewed pairs live outside the original prepared bundle; reusing that bundle's default pairs would select the wrong experiment inputs.
  分组任务可显式使用独立且固定哈希的审核训练对清单，并核对语料、索引、训练问题及 qrels。R6 审核训练对位于原数据包之外，直接使用原包默认训练对会选错实验输入。

See the [owning specification](../docs/16-grouped-loss-experiment.md). No BM25 serving, promotion gate, frozen v1 labels or real held-out test input changed.

详见[所属规格](../docs/16-grouped-loss-experiment.md)。BM25 服务、发布门槛、冻结 v1 标签及真实留出测试输入均未改动。

## Actual checks / 实际验证

Environment: existing local macOS Python 3.11.16, PyTorch 2.13.0 and Transformers 4.57.6. Used tiny randomly initialized synthetic BERT fixtures on CPU; no pretrained-model download or dependency download. This is **not** the server's exact CUDA lock environment.

环境为本地已有 macOS Python 3.11.16、PyTorch 2.13.0、Transformers 4.57.6；使用 CPU 微型随机 BERT 合成样例，没有下载预训练模型或依赖。此环境**不是**服务器精确 CUDA 锁环境。

| Check / 检查 | Observed result / 实测结果 |
| --- | --- |
| `test_training_grouped test_grouped_operator test_training test_cli_ops` | 73 tests, zero failures/errors/skips; 7.439 s / 73 项全通过、零跳过 |
| Installed wheel baseline outside source checkout / 安装包在源码外运行基线 | Passed / 通过 |
| Installed wheel grouped ML tests / 安装包分组 ML 测试 | 18 tests, zero failures/errors/skips; 1.445 s / 18 项全通过、零跳过 |
| In-memory mutations: wrong tail divisor and validation-based early selection / 内存变异：错误尾批分母及提前选轮次 | Both caught by their regression tests / 两项均被对应回归测试检出 |
| Full suite on local host / 本机全量测试 | 866 tests: 780 passed, 1 failure, 18 errors, 67 skips / **不是全量通过** |
| Reproduce failed cases on unchanged `e3168ca` / 原提交复现失败项 | Same 19 test IDs fail; no new failing ID observed / 相同 19 项失败，未观察到新增失败项 |
| `git diff --check` | Exit 0 / 退出 0 |

The 18 errors involve sandbox-denied localhost listeners or portable-receipt fixtures on this host. The one failure is an existing independent-tokenizer comparison incompatibility. Skips: 65 need the complete serving extra (Streamlit is absent), one platform process-start-time check and one tokenizer-reference availability check. None is counted as a pass. A clean full-suite result under the existing supported lock remains required before release.

18 项错误涉及沙盒拒绝本地监听及本机便携回执样例；一项失败是已有的独立分词器比较兼容问题。跳过的 65 项缺完整服务依赖（未装 Streamlit），另有一项平台进程启动时间检查和一项参考分词器可用性检查；均不算通过。发布前仍需在现有支持的锁环境中取得全量通过结果。

One initial reference test used `logaddexp` versus the implementation's `softplus`; their rounding differed in nearly cancelling pairwise bias gradients, which Adam amplified (maximum head-bias difference 4.4e-6). The accumulation reference now independently assembles batches, objective and updates using the same elementary kernel, and matches exactly on the observed run. A separate float64 scalar finite-difference test checks loss algebra; the wrong-divisor mutation proves the accumulation test detects its intended defect. The production loss was not changed to hide this difference.

初始参考测试使用 `logaddexp`，实现使用 `softplus`，两者在近乎相消的成对偏置梯度上存在舍入差异，经 Adam 放大后 head bias 最大差 4.4e-6。累积参考现独立构造批次、目标及更新，但采用相同基础核，此次实测参数完全一致；另有 float64 标量数值梯度验证公式，错误分母变异证明累积测试能检出目标缺陷。未为隐藏差异而改动实际损失。

## Reproduction and identities / 复现与身份

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:tests \
  /opt/homebrew/bin/python3.11 -m unittest -v \
  test_training_grouped test_grouped_operator test_training test_cli_ops

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:tests \
  /opt/homebrew/bin/python3.11 -m unittest discover -s tests -v
```

- Package source snapshot: `c98015ac3cea1566c6f665c28e8cdb24051616b795a817f5f4ff9a8f8d76b338` (111 Python source files).
- Built wheel: `dae1623a9ee2dfe4935e88a8774dd99cdd5e78446f840f5021685fda1198a050` (302,068 bytes).
- Detailed local commands/logs and receipts: `devsearch-runtime/autodl-results-e3168ca/next-cycle-20261008/training/verification/` in the parent workspace; summary: `reports/evidence/grouped-cpu-20261009.json`.

以上分别为源码快照、安装包哈希及本地完整日志位置；安装包仅约 302 KB。详细证据不包含模型权重或私人判定内容。

## Remaining work / 剩余工作

1. Continue independent B review after quota becomes available (150 / 4,345 accepted; 4,195 pending). At the recorded snapshot, Claude's local A review has 1,550 accepted. A completion alone cannot seal v2; disagreements and unsure items still require independent adjudication.
   B 审额度恢复后继续（已接受 150 对，余 4,195 对）。当前快照中 Claude 本地 A 审已接受 1,550 对。只完成 A 不能封存 v2，还须独立处理分歧及不确定项。
2. Reevaluate all frozen systems on the same sealed full-v2 labels before deciding whether the optional GPU comparison is worthwhile. Thirty human cards remain pending and optional under the recorded addendum; agent judgments remain agent judgments.
   完整 v2 封存后统一重评全部冻结系统，再判断可选 GPU 对照是否值得。30 张人工卡按补充协议仍为可选且待确认，Agent 判定保持该来源标记。
3. If proceeding, verify real R6 artifacts, freeze the six configs and bounded serial runner, check the exact existing CUDA lock, and perform the real-model smoke. The loop/CLI implementation is tested; that actual-input campaign and GPU smoke were not run here.
   若继续，验证真实 R6 产物，冻结六个配置及限时串行执行器，验证既有 CUDA 锁并做真实模型短测。循环／CLI 已测试，真实输入整轮运行和 GPU 短测本次未执行。

No SSH session, GPU startup, new rental, paid API, real test scoring or large transfer was initiated. The 60 existing server checkpoints were left untouched. The server's current power state was not independently verified.

本次未发起 SSH、GPU 开机、新租机、付费 API、真实测试集评分或大文件回传。服务器既有 60 个断点未被触碰；未独立确认服务器当前电源状态。
