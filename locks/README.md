# Dependency locks / 依赖锁

One hashed lock per accelerator profile and CPython minor version, resolved for the **target** platform by `scripts/lock_profile.py` (it uses the `uv` resolver; `pip install uv`). `apple-mps-cp312.lock` targets macOS arm64; `nvidia-cuda-cp312.lock` targets Linux x86_64 with PyPI's default CUDA torch build. Both cover the runtime dependencies of the `ml` and `serve` extras; the project wheel is pinned separately by `plan --wheel-sha256`.

每个加速器配置、每个 CPython 次版本一份带哈希的依赖锁，由 `scripts/lock_profile.py` 针对**目标**平台解析（使用 `uv`，`pip install uv`）。`apple-mps-cp312.lock` 面向 macOS arm64，`nvidia-cuda-cp312.lock` 面向 Linux x86_64（PyPI 默认 CUDA 版 torch）。均覆盖 `ml` 与 `serve` 额外依赖的运行依赖；项目 wheel 另由 `plan --wheel-sha256` 固定。

```sh
# On a connected machine: fetch every hashed artifact into a wheelhouse (a hash mismatch aborts)
python scripts/bootstrap_env.py download --lock locks/apple-mps-cp312.lock --dest wheelhouse/apple-mps
# On the target (no network needed): a NEW virtual environment from the wheelhouse only, then the project wheel
python scripts/bootstrap_env.py install --lock locks/apple-mps-cp312.lock --wheelhouse wheelhouse/apple-mps --wheel dist/devsearch-0.0.0-py3-none-any.whl --venv /path/to/new/env
# Record the exact lock in the bundle so the plan binds it
python -m devsearch.bundle assemble --bundle <bundle> --lock locks/apple-mps-cp312.lock
```

**What a lock does not prove.** A lock fixes versions and artifact hashes; it does not show that torch's MPS or CUDA build runs on that OS, driver and device. The profiles stay `unverified` until `doctor` and a real run on that machine say otherwise (docs/11 G06). The versions here are what PyPI offered on the day the lock was made; the unit tests ran against the versions recorded in the evidence report, which differ for macOS (see there). After installing a lock on the target, run the whole test suite there (`python scripts/record_receipt.py implementation --output <dir>`) and keep the receipt.

**锁不能证明的内容。**锁只固定版本和产物哈希，不证明 torch 的 MPS／CUDA 版本能在该系统、驱动与设备上运行；配置在 `doctor` 及目标机真实运行给出结论前保持 `unverified`（docs/11 G06）。此处版本为制锁当日 PyPI 所提供；单元测试所用版本见证据报告（macOS 的 torch 版本与测试环境不同）。在目标机安装后应在目标机运行完整测试并保留回执。
