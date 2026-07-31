# packages 目录说明

这个目录存放板端运行所需的 RKNN 依赖，以及可选的 RGA 相关依赖。

## 当前内容

- `rknn_toolkit_lite*.whl`
  - RKNN Lite Python 包
- `librknnrt.so`
  - RKNN 运行库
- `im2d.h`
  - RGA 头文件
- `packages.md5sum`
  - 包文件校验信息
- `wheelhouse/`
  - RK3588 Ubuntu 22.04 / aarch64 / Python 3.10 的离线 Python wheels
- 根目录 `requirements.lock`
  - 与 wheelhouse 配套的完整锁定依赖版本

当前仓库现状：

- 当前已包含 `librknnrt.so`
- 当前已包含 `im2d.h`
- 当前默认未包含 `librga.so`

因此，项目脚本虽然支持复制 RGA 相关文件，但当前仓库并未自带完整 RGA 运行库。

## 使用方式

- 首次执行 `./start_web_server.sh start` 时，内部会通过 `install_runtime_venv.sh` 优先从 wheelhouse 离线安装依赖
- wheelhouse 存在时不会回退到公网 pip；文件缺失会直接失败，避免产生不可复现的环境
- `./start_web_server.sh preflight` 会检查 MPP、FFmpeg rkmpp、NPU 驱动、RKNN 运行库和模型文件
- 板端交付时建议整个目录一起带上
