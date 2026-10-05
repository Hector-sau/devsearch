# Scaffold verification / 骨架验证记录

Date: 2026-10-05. Scope: package structure, internal interfaces and fail-closed CLI placeholders. These checks do not establish retrieval quality, model training, API behavior or hardware compatibility. T01 is partially implemented; no full M0–M5 gate is complete.

日期：2026-10-05。范围为包结构、内部接口及明确拒绝执行的 CLI 占位。检查不能证明检索效果、模型训练、接口行为或硬件兼容；T01 部分实现，M0～M5 均未完整验收。

| Check / 检查 | Result / 结果 | Limit / 边界 |
| --- | --- | --- |
| Python interpreter / 解释器 | Existing bundled Python 3.12.14 / 已有捆绑版本 | Python 3.11 minimum declared; not tested on every supported version / 声明最低 3.11，未逐版本实测 |
| Five standard-library tests / 五项标准库测试 | Passed / 通过 | Includes all ten help and rejection paths, no network/write side effects, dependency boundaries and immutable request / 覆盖十个帮助及拒绝路径、无网络写入、依赖边界和不可变请求 |
| Wheel build / wheel 构建 | Passed with existing setuptools/wheel in a temporary source copy / 临时源码副本使用已有工具通过 | No dependency lock or full ML runtime verified / 未验证依赖锁和完整 ML 环境 |
| Clean wheel installation / 干净 wheel 安装 | Passed offline in a temporary venv; import and retrieve help ran outside checkout / 临时 venv 离线安装、在 checkout 外导入及运行帮助通过 | Checks package portability only; temporary environment removed / 仅包便携性，临时环境已清理 |
| GitHub workflow / GitHub 工作流 | Prepared, not run / 已准备、未运行 | Manual trigger in private repository [Hector-sau/devsearch](https://github.com/Hector-sau/devsearch); no remote CI pass claimed / 私有仓库内手动触发，不宣称远端 CI 已通过 |
| Real sources, MiniLM, MPS, API/UI / 真实来源模型设备接口界面 | Not implemented or not tested for this project / 待实现或待项目验证 | No metrics, checkpoints or trained artifacts exist / 没有指标、断点或训练产物 |

Source check command, from the project root using Python 3.11+:

在项目根目录使用 Python 3.11+ 的源码检查命令：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -v
```

The current Mac's plain `python3` is 3.9.6; select the verified existing 3.12 interpreter instead. The test run used the bundled interpreter at `/Users/macbookair/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3`. This path is a local verification record, not a portable runtime requirement.

当前 Mac 的普通 `python3` 是 3.9.6，应选已验证的现有 3.12 解释器。测试使用上述捆绑解释器绝对路径；该路径只是本地验证记录，不是便携运行要求。

For a future wheel built with packaging tools already available in an authorized environment:

在已获授权且具备打包工具的环境生成 wheel 后，可使用：

```sh
python scripts/check_wheel.py dist/devsearch-0.0.0-py3-none-any.whl
```

The script creates and removes a temporary environment, installs only the supplied wheel with network indexes disabled, and checks import/help outside the checkout. No persistent project environment, model cache, data download or training run was created. Keep scaffold-only tests separate from future functional tests; replace a CLI's refusal assertion when implementing its real behavior.

脚本创建并清理临时环境，禁用网络索引，仅安装提供的 wheel，并在原 checkout 外检查导入和帮助。本次未建立持久项目环境、模型缓存、数据下载或训练运行。骨架测试与未来功能检查分开，命令实现后替换对应拒绝执行断言。
