# Execution review verification / 执行评审验证记录

Date: 2026-10-05. This record separates the concurrent AI's CPU implementation, independently repeated checks and the new execution specification. No dataset acquisition, model download, training, GPU benchmark or user study occurred in this review.

日期：2026-10-05。本记录区分另一位 AI 的 CPU 实现、本次独立复查及新增执行规格。本次评审未采集真实数据、下载模型、训练、测试 GPU 性能或开展用户研究。

| Evidence / 证据 | Result / 结果 | Boundary / 边界 |
| --- | --- | --- |
| Remote snapshot / 远端快照 | Main `30927a4`; development branch `claude/youthful-turing-uc1lkv` at `ee4edea2d53e103119b5f45fe4f1278e4f478499` | Later work not automatically covered / 不自动覆盖后续工作 |
| Static branch inspection / 分支静态检查 | Three real CPU commands, TOML configuration, corpus/index schemas and tests; seven remaining operation stubs / 三个 CPU 命令、TOML 配置、语料索引结构测试及七个占位入口 | CPU baseline, not training/app completion / 基线不代表训练应用完成 |
| Independent CPU suite / 独立 CPU 测试 | 125 tests, 2.897 s, exit 0 on existing Python 3.12.14 / 125 项通过，退出 0 | Temporary archive of exact `ee4edea`, no new dependency installation; fixtures only / 精确提交临时副本，无安装，仅样例 |
| Focused defect probes / 定点缺陷探测 | F01–F03 reproduced; see document 11 / 已复现，见第 11 号文档 | Temporary fixtures only; no source modification or formal data exposure / 仅临时样例，不改源码、不接触正式数据 |
| GitHub run check / GitHub 运行核对 | [37307966474](https://github.com/Hector-sau/devsearch/actions/runs/37307966474), completed/success on `fda9eeaedc54ad6b081b862704f4fa6fc3b5b892` | Read run metadata; did not rerun CI or claim uninspected logs / 读取元数据，未重跑或声称审过全部日志 |
| Architecture review / 架构评审 | Three agent roles and a second challenge pass; findings integrated into 10–12 / 三条 Agent 评审及二轮质疑，修正纳入文档 | Agent review, not human testing or full code audit / 非人工研究或完整代码审计 |
| Document validation / 文档校验 | 22 Markdown files, 10 JSON records, three diagram/source pairs, arithmetic examples and diff whitespace checks passed / 链接代码块、JSON、三组图源、算术及空白检查通过 | Structural checks only; Mermaid was not rendered or visually inspected / 仅结构检查，未渲染视觉检查图表 |
| Device/data readiness / 设备数据就绪 | Pending / 未就绪 | No real corpus, MiniLM, MPS or CUDA result / 无真实语料模型设备结果 |

Independent run: create a temporary directory, export the pinned branch with `git archive --format=zip`, validate relative archive paths, and execute the command below with the existing interpreter. `PYTHONPATH` pointed to the temporary `src`, bytecode writes were disabled, and Hub/Transformers offline flags were set. The temporary directory was removed afterward. No work in the implementation branch was edited.

独立运行方法：临时目录导出固定分支归档，校验相对路径后用已有解释器执行下列命令；`PYTHONPATH` 指向临时源码，禁写字节码并设置模型库离线标记。执行后清理临时目录，未修改开发分支的工作。

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests
```

Interpreter: `/Users/macbookair/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`, version 3.12.14. Local path is provenance only, not a requirement for other hosts. Source/Markdown review and new run-contract checks do not turn any G01–G10 end-to-end gate into a pass.

解释器为上述本机路径、版本 3.12.14；路径仅作来源记录，不是其他主机要求。源码文档评审及新增契约检查不会将 G01～G10 任一完整门禁改为通过。

Focused probe observations: (F01) replace a fixture source URL with an invalid HTTP URL, call `write_corpus`, observe `ManifestError` and an existing final manifest, then retry with valid input and receive the immutable-manifest error; (F02) assign two fixture chunks the same evidence group and opposite train/test owners, recalculate consistent hashes via the fixture writer, and observe `load_corpus` accept both; (F03) build a valid fixture index inside the temporary runtime root and invoke baseline retrieval with an output outside that root, observing exit 0 and the output file. All probe directories were removed; these tests intentionally establish defects, not successful acceptance.

定点探测：F01 将样例来源改为非法 HTTP，写入报错后正式清单仍存在，换合法输入重试遭不可变清单拒绝；F02 将两个样例段落设为同组但 train/test 冲突归属，使用样例工具重建一致哈希后，加载仍接受两者；F03 在临时运行根内构建有效索引，将检索输出指定至根外，观察退出 0 且文件存在。探测目录已清理，这些探测证明缺陷，不代表验收通过。
