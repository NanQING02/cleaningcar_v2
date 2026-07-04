# models 目录说明

本目录只保留当前主链路实际使用的模型。

## 当前实际使用

- `detection/best.rknn`
  - 主检测模型
  - 后处理支持 `video.fp_output_mode=6|9`

- `plate/plate_detect.rknn`
  - 双模型车牌检测模型

- `plate/plate_rec_color.rknn`
  - 双模型车牌字符/颜色识别模型

- `wheel/2026.4.28CRwheel.rknn`
  - 左右车轮旁路检测模型
  - 仅在 `wheel.enabled=true` 时启用

## 当前策略

- 当前主链路只保留双模型车牌流程
- 旧单模型 `lpr.rknn` / `lprnet.rknn` 不再参与运行
- 如需保留历史模型，请放到 `future_modules/legacy_models/`

## 子目录说明

- `detection/README.md`
  - 检测模型目录说明
- `plate/README.md`
  - 车牌双模型目录说明
- `wheel/README.md`
  - 车轮旁路模型目录说明
