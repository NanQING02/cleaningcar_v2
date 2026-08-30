# models/plate 目录说明

这个目录存放当前双模型车牌链路使用的模型。

## 当前文件

- `plate_detect.rknn`
  - 车牌检测模型
- `plate_rec_color.rknn`
  - 车牌字符与颜色识别模型

## 使用方式

- 由 `cleaningcar/plate_lpr.py` 负责加载和调用
- 当前主链路只保留这套双模型方案

## 当前运行精度

2026-08-31 在 `RK3588-Office-02` 使用 RKNNLite 2.3.2 的 `verbose` 运行时信息核验：

- `plate_detect.rknn` 的输入、特征张量和主要权重均为 `FLOAT16`
- `plate_rec_color.rknn` 的输入、特征张量和主要权重均为 `FLOAT16`
- 两个模型当前都不是 INT8 运行模型

当前双模型链路每次触发时会先执行全帧车牌检测，再对检测到的车牌区域逐个执行字符/颜色识别。后续优化需要分别记录两个阶段的耗时，并用代表性车牌数据集评估 INT8 量化后的速度和精度，不能只用模型文件大小判断量化状态。
