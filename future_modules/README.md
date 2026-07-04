# future_modules 目录说明

本目录用于存放当前不参与主运行，但需要保留的历史实现、备用模块和后续开发内容。

## 子目录

- `legacy_models/`
  - 历史模型文件
- `legacy_detection/`
  - 历史检测实现
- `legacy_lpr/`
  - 历史单模型车牌识别实现
- `unused_utils/`
  - 当前主流程未启用的工具模块
- `acceleration/`
  - 当前未启用的加速插件或实验功能

## 使用约束

- 当前主运行链路不要直接依赖这里的代码
- 当前 FP 检测后处理切换能力已经在主链路中完成，不依赖此目录

## 子目录说明

- `acceleration/README.md`
- `legacy_detection/README.md`
- `legacy_lpr/README.md`
- `legacy_models/README.md`
- `unused_utils/README.md`
