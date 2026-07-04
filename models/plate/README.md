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
