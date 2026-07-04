# models/detection 目录说明

这个目录存放当前主链路使用的检测模型。

## 当前文件

- `best.rknn`
  - 主检测模型
  - 当前项目默认检测入口

## 使用方式

- 由主流程自动加载
- 后处理模式通过 `video.fp_output_mode` 控制 `6` 或 `9`
