# CPU demo of the formal BM25 release / 正式 BM25 发布包的 CPU 演示 (2026-10-08)

Recorded by Claude (an agent), not a human test. Demo record only — **not a performance or quality result**.

| Item | Value |
| --- | --- |
| Code | `e3168ca6fef9b64ea0688ea059332c729125e30f`, exported with `git archive` into a separate demo directory |
| Release | `release-ce3160fc579a52ae2ec30d15412853eb` (evaluated, defaults to BM25, no model), from `backup/release/` of `devsearch-e3168ca-backup.tar.gz`; local SHA-256 `b8ee3cd6…0e60` equals the delivery manifest |
| Environment | Apple M4, 16 GiB, macOS 26.6.2; Python 3.11.16, FastAPI 0.141.1, uvicorn 0.52.4, httpx 0.28.1; no Streamlit; nothing installed |
| Server | `python3.11 -m devsearch.serve --config serve-demo.toml --manifest runtime/backup/release --runtime-root runtime` on `127.0.0.1:18080` (`configs/serve.toml` with only the port changed); started and stopped by the demo |
| Inputs | demonstration inputs D1–D4 from [DEMO_SCRIPT](../../../docs/DEMO_SCRIPT.md); no test-set question |

Files:
- `demo-record.json` — every scenario's command or request, exit code or HTTP status, effective mode, warnings and the top result's title,
  section and source link (passage text omitted).
- `ui-logic.txt` — output of the page's own functions (`describe_response`, `describe_error`, `call_api`) on those responses; the rendered
  page was **not** executed.

本目录记录正式 BM25 发布包在本机 CPU 上的演示（Agent 执行，非真人测试，非性能结果）：五项接口场景通过，界面仅验证逻辑函数，渲染与截图未执行。
