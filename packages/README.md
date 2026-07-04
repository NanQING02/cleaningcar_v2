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

当前仓库现状：

- 当前已包含 `librknnrt.so`
- 当前已包含 `im2d.h`
- 当前默认未包含 `librga.so`

因此，项目脚本虽然支持复制 RGA 相关文件，但当前仓库并未自带完整 RGA 运行库。

## 使用方式

- 首次执行 `./start_web_server.sh start` 时，内部会通过 `install_runtime_venv.sh` 优先从这里安装或复制依赖
- 板端交付时建议整个目录一起带上
