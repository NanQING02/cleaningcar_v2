# RGA 运行默认策略

## 现场结论

- RGA 不适用于本项目当前实时多路高负载链路。
- 现场压测已确认：高负载开启 RGA resize 存在严重稳定性问题，可导致板端死机。
- 项目运行链路固定使用 `cv2.resize`，不再提供配置或环境变量开启 RGA 的入口。

## 当前实现

- `run_zone_detect.py` 启动时强制设置 `CLEANINGCAR_RGA_DISABLE=1`，并清理 `CLEANINGCAR_RGA_ENABLE`。
- `config_manager.py` 会清理历史 `video.rga_enable` 字段。
- `cleaningcar.resize_accel` 不加载 RGA backend，`resize_backend_name()` 固定返回 `cv2`。

## 维护约束

- 不要在现场配置中添加 `video.rga_enable`。
- 不要通过 `CLEANINGCAR_RGA_ENABLE=1` 绕过禁用策略。
- `future_modules/acceleration/rga_resize_plugin.py` 仅保留为历史实验/问题复现代码，不接入主流程。
