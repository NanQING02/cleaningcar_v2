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

## 当前运行精度与历史

- 2026-08-31 在 `RK3588-Office-02` 使用 RKNNLite 2.3.2 核验时，当时的
  `plate_detect.rknn` 与 `plate_rec_color.rknn` 都是 FLOAT16。
- 2026-09-01 提交 `7cbbc2f` 将 `plate_detect.rknn` 从 4,305,909 字节替换为
  当前 2,346,737 字节的 INT8 三路 raw-head 检测模型，并增加专用解码与回归测试。
- `plate_rec_color.rknn` 此后未替换，当前仍为 FLOAT16 字符/颜色识别模型。

当前双模型链路每次触发时先执行车牌检测，再对检测到的车牌区域逐个执行字符/颜色识别；正式配置默认每两帧运行一次。量化状态应结合模型提交记录、输出schema和板端RKNN运行信息确认，不能只按文件大小推断。
