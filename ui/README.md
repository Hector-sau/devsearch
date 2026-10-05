# UI boundary / 界面边界

The Streamlit application is planned, not implemented. It will call the FastAPI service and render escaped passages, code fragments and source links. It must not import the PyTorch model or training modules. The baseline, rerank and comparison views must display the effective mode and explicit fallback state returned by the API.

Streamlit 应用待实现。它将调用 FastAPI，展示转义后的段落、代码片段和来源链接，不导入 PyTorch 模型或训练模块。基线、重排和对比视图必须显示接口返回的实际模式及明确降级状态。

See [integration contracts](../docs/04-integration-and-decisions.md) for fields/errors and [machine handoff](../docs/05-machine-handoff.md) for loopback ports and SSH access. UI development can use explicit fixture responses; fixture output is never a model evaluation.

字段错误见集成契约，回环端口及 SSH 见机器交接文档。界面开发可使用明确标记的测试响应，但不能将样例输出当成模型评测。
