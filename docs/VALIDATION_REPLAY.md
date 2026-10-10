# Frozen validation replay / 冻结验证结果重放

This CPU command reproduces the published validation-v4 ranking arithmetic from the owner's frozen inputs. It uses the repository's ranking metrics and performs no training, model scoring, network request or new labeling.
本命令在 CPU 上利用所有者保存的冻结输入，复现已发布的 validation-v4 排序指标。它使用仓库中的指标实现，不进行训练、模型打分、网络请求或新标注。

The public repository includes the script, regression tests and [aggregate replay evidence](../reports/evidence/public-replay-20261010/README.md). It does **not** include the private questions, judgments or ranking orders. A public clone can run the [small fixture application](REPRODUCIBILITY.md); the complete experiment requires the owner-held inputs listed below.
公开仓库包含脚本、回归测试及[汇总重放证据](../reports/evidence/public-replay-20261010/README.md)，**不包含**私有问题、判定及排序序列。公开克隆可以运行[小型示例应用](REPRODUCIBILITY.md)；完整实验需要下列所有者保存的输入。

## Inputs and byte identities / 输入与文件身份

Files can be moved to any local directory. `--release` must point to the directory containing the two sealed release files. The four files total 8,238,786 bytes, about 7.86 MiB; GPU weights are unnecessary for this replay.
文件可以迁移至任意本地目录。`--release` 必须指向包含两个封存文件的目录。四个文件共 8,238,786 字节，约 7.86 MiB；本重放不需要 GPU 模型权重。

| Input / 输入 | Bytes / 字节 | SHA-256 |
| --- | ---: | --- |
| `queries-validation.jsonl` | 143,556 | `a45a64dc9753b8d84442620e8f8f134e6ad66d99409241dccc3d67dbeee54ee5` |
| `compact-ranking-orders.json` | 4,170,307 | `5e28564770f93de98f1752a94845088714d41a6cd72f1ebc245e5163078c84ba` |
| `label-release.json` | 123,079 | `bee1c2bcd08b4085c6391fc022fca5b0355bce67c0e2a8317cb1c1bcccf2e038` |
| `resolved-labels.json` | 3,801,844 | `2e998b5817613b29eb7903c165302fd042d9f2a1eee59f010ddca0bd0cb83174` |

The queries, compact rankings and sealed manifest are pinned directly in the script. The manifest binds the resolved-label hash. Editing labels and replacing their hash in the manifest cannot bypass the manifest's fixed identity. Legacy filesystem paths inside the manifest are not dereferenced.
脚本直接固定问题、精简排序序列及封存清单的哈希，清单再绑定最终标签的哈希。修改标签并替换清单中的标签哈希，仍无法绕过清单本身的固定身份。清单中的旧文件系统路径不会被访问。

## Run / 执行

Run from the repository root with Python 3.11 or 3.12. Replace the input placeholders with the paths to your preserved files; create the output directory first. The output must be a new file: existing evidence is never overwritten.
在仓库根目录使用 Python 3.11 或 3.12 执行。将输入占位路径替换为保存文件的实际位置，并先创建输出目录。输出必须使用新文件名；已有证据不会被覆盖。

```bash
mkdir -p replay-output
python scripts/replay_validation.py \
  --queries /path/to/private-inputs/queries-validation.jsonl \
  --rankings /path/to/private-inputs/compact-ranking-orders.json \
  --release /path/to/private-inputs/validation-v2-sampled202-human6-v4 \
  --output replay-output/validation-v4-aggregate.json
```

The implementation uses Python's standard library and existing pure-Python package metrics. It requires neither PyTorch nor a GPU. Success exits `0`; an input identity, structure, coverage or output gate failure exits `2` without publishing a result.
实现仅依赖 Python 标准库及仓库已有的纯 Python 指标函数，不需要 PyTorch 或 GPU。成功退出码为 `0`；输入身份、结构、覆盖率或输出门槛失败时，退出码为 `2`，且不生成结果。

## Gates and calculation / 门槛与计算

- Exactly 202 unique English validation queries and 202 unique groups; no duplicate evidence-group membership and no excluded query.
  必须为 202 个独立英文验证问题及 202 个独立组；依据组不可重复，任何问题都不能被排除。
- Exactly 4,571 unique binary query–passage labels, including at least one known positive for every query.
  必须为 4,571 个独立的二元问题—段落标签，每个问题至少有一个已知正例。
- Exactly 25 systems: BM25 and R4–R7 raw/fused orders for seeds 42, 43 and 44. Every order must be a permutation of the same 50 distinct candidates for its query.
  必须为 25 个系统：BM25，以及 R4–R7 在种子 42、43、44 下的原始及融合排序。每个问题的各系统排序必须是同一组 50 个不同候选段落的有效排列。
- Every system's top 10 must be fully labeled. Missing labels cannot silently become negative judgments.
  各系统前 10 名必须全部已标注；缺失标签不得被默认为负例。
- MRR@10, pooled nDCG@10 and known-positive recall@50 use **all** known positives, including positives that retrieval missed. Both query-macro and group-macro means are reported.
  MRR@10、池内 nDCG@10 及已知正例 recall@50 使用**全部**已知正例，包含检索未找到的正例；同时报告按问题及按组平均的结果。
- R5 seed scores are averaged **inside each group** before the paired group bootstrap: 10,000 repetitions, seed `20261008`, 95% descriptive intervals. The three seeds do not become three independent copies of the benchmark.
  配对组 bootstrap 前，先在**各组内部**平均 R5 三个种子的得分：重复 10,000 次，随机种子 `20261008`，报告 95% 描述性区间；三个种子不能变成三份独立验证样本。

## Recorded result / 已记录结果

The actual CPU replay checked 202 queries, 202 groups, 4,571 pairs, 498 positive pairs and 25 systems, with zero excluded queries and 100% labeled top-10 coverage. Only 45.25% of top-50 positions are labeled, so recall refers to the **known** positives. Query-macro and group-macro values coincide because this benchmark has one question per group.
实际 CPU 重放校验了 202 个问题、202 个组、4,571 个标签对、498 个正例对及 25 个系统，无问题排除，前 10 名标注覆盖率为 100%。前 50 名只有 45.25% 的位置已标注，因此 recall 针对**已知**正例。该验证集每组只有一个问题，所以按问题及按组平均的指标相同。

| System / 系统 | MRR@10 |
| --- | ---: |
| BM25 | 0.901416 |
| R5 + RRF, seed 42 | 0.934076 |
| R5 + RRF, seed 43 | 0.944059 |
| R5 + RRF, seed 44 | 0.945050 |
| R5 + RRF, three-seed mean / 三种子均值 | 0.941062 |

The mean MRR gain is `+0.039645`, with a descriptive paired group interval `[+0.018028, +0.063353]`. The pooled nDCG gain is `+0.019024`; its interval `[-0.001527, +0.039868]` includes zero. Replay matches the existing 25-system aggregate report to floating-point tolerance; machine-specific elapsed time is informational.
平均 MRR 提升为 `+0.039645`，描述性配对组区间为 `[+0.018028, +0.063353]`。池内 nDCG 提升为 `+0.019024`，区间 `[-0.001527, +0.039868]` 包含零。重放与已有 25 系统汇总报告在浮点误差范围内一致；耗时仅记录具体运行环境的信息。

## Evidence boundary / 证据边界

This verifies arithmetic over sealed inputs and their exact byte identities. The original adjudication and review-ledger audit remain in their original evidence; this command does not repeat that audit or prove that the judgments are correct.
本步骤校验封存输入上的算术及精确文件身份。原裁决与审查账本审核仍以原始证据为准；本命令不重新审核账本，也不证明标签本身正确。

The questions are synthetic and validation was repeatedly consulted during development. These results support the stated **same-benchmark validation** finding, not a fresh test or real-user success claim. See [reproducibility](REPRODUCIBILITY.md) and [results](../reports/RESULTS.md) for the historical 197-question test and separate 59-question holdout, including the holdout regression. BM25 remains the application default.
问题由程序生成，开发期间也反复使用了验证集。这些结果支持已声明的**同一验证基准**发现，不支持新测试集或真实用户成功的结论。[复现说明](REPRODUCIBILITY.md)及[结果报告](../reports/RESULTS.md)保留了历史 197 题测试及独立 59 题留出评测，包含留出集退化结果。应用默认模式仍为 BM25。

The JSON output contains only aggregate metrics, counts, input hashes, calculation scope and timing. It exports no question text, query/group/chunk identifiers, individual labels, ranking orders or filesystem paths.
JSON 输出仅包含汇总指标、计数、输入哈希、计算范围及耗时，不导出问题文本、问题／组／段落标识、逐条标签、排序序列或文件系统路径。

## Regression checks / 回归检查

```bash
python -m unittest discover -s tests/tooling -p test_validation_replay.py -v
```

The 14 focused tests cover hand-computed ranking metrics, retrieval misses, averaging seeds before group resampling, input-byte tampering, rewritten label manifests, missing top-10 judgments, malformed permutations, duplicated memberships/labels/JSON members, privacy and refusal to overwrite prior evidence.
14 个针对性测试覆盖手工计算的排序指标、检索遗漏、先平均种子后抽样组、输入字节篡改、标签清单改写、前 10 名缺审、无效排列、重复成员／标签／JSON 字段、隐私及拒绝覆盖原证据。
