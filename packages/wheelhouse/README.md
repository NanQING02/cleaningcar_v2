# Offline wheelhouse

本目录用于 RK3588 板端离线安装 Python 运行依赖，适用基线：

- Ubuntu 22.04
- aarch64
- Python 3.10

安装脚本检测到本目录中的 wheel 后，会使用 `--no-index`，不会回退到公网 pip。
完整版本锁定见项目根目录 `requirements.lock`。
