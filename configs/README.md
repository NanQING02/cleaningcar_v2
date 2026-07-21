# configs 目录说明

`configs/` 用于存放项目运行配置。当前建议以 `configs/config.json` 作为正式主配置，其他配置只保留为备份或场景切换。

## 当前配置文件

- `config.json`
  - 默认主配置
- `config_绕行.json`
  - 备用 / 绕行配置

## 重点配置项

### `system.device_id`

- 当前多路运行时命名空间的主键
- 默认运行时文件会按它自动隔离
- 多路部署时，**每一份配置的 `device_id` 必须唯一**

### `system.api.url` / `system.api.wheel_photo_url`

- `system.api.url`：主事件上报接口（`POST /api/vehicle/wash-event`），完整 URL
- `system.api.wheel_photo_url`：车轮照片上报接口（`POST /api/vehicle/wheel-photo`），**留空时自动从 `system.api.url` 推导**（替换路径最后一段为 `wheel-photo`）
- 例如 `url=http://host:port/api/vehicle/wash-event` → 推导 `wheel_photo_url=http://host:port/api/vehicle/wheel-photo`
- 若两个接口走不同服务器，可显式填 `wheel_photo_url` 覆盖

### `system.wheel_photo_base_dir`

- 车轮照片落盘根目录，默认 `/data/ftp`
- `photoUrl` 上报字段是绝对路径（如 `/data/ftp/box/20260119/14/xxx.jpg`）
- `type=5` 主事件里的 `wheelResults[].photoUrl` 同样使用绝对路径，不再上传 `imageBase64`

### `video.source` / `video.source_mode`

- `video.source`：视频源地址，可以是 RTSP、摄像头索引或本地文件
- `video.source_mode`：建议优先用 `auto`
- 当输入被识别为本地文件时，会按文件源处理，默认跑完一遍后退出

### `video.hw_decode`

- 读流只使用 `FFmpeg rkmpp` 硬解
- 当前配置固定为 `video.decode_backend=ffmpeg`；旧配置里的 `auto`、`gstreamer` 或 `software` 会在加载时统一归一为 `ffmpeg`
- 不使用 GStreamer，也不切软件解码；FFmpeg rkmpp 打开失败时主链路会按读流失败处理并重连或退出

### `video.fp_output_mode`

FP 检测模型后处理模式：

- `6`
  - 默认值
  - 即使模型给出 `9` 个输出，也只使用每个尺度的 `box + class`
- `9`
  - 使用每个尺度的 `box + class + score`

### `logic.per_id_video_dir`

- 单车视频输出目录
- 留空、只填空白、或目录不可写时，会自动回退到 `video_result/per_id/`

### `logic.enable_per_id_video`

- 当前只保留单车视频留存开关 `logic.enable_per_id_video`
- 单车视频仅使用 `FFmpeg` 硬编写出；FFmpeg 硬编不可用时不保存该段单车视频
- `logic.per_id_video_source=auto` 时，`logic.no_draw=true` 默认写主路原始解码帧；`logic.no_draw=false` 时写绘制后的帧
- Web 开发者参数只保留一个录像画面开关：关闭时写 `raw` 原始帧并禁用绘制；开启时写 `annotated`，同时启用 Zone、轨迹、锚点、水雾和双模型车牌结果绘制
- `logic.per_id_type6_require_plate_candidate=false` 时，Type6 不再因缺少有效车牌候选被拦截；需要恢复旧门槛时可改为 `true`
- Web 端不再浏览这些单车录像，但后台仍会继续保存
- 旧的全局视频保存字段 `video.save_video`、`logic.enable_global_video` 已彻底删除，不再生效

### `logic.draw_plate_boxes`

- 代码缺省值为 `false`；当前主配置 `configs/config.json` 中为 `true`
- 只有同时未启用 `logic.no_draw` 且开启该项时，车牌绘制才会出现在调试帧与事件截图中

### `wheel.*`

- `wheel.enabled`
  - 是否启用左右车轮 RTSP 旁路
- 车轮旁路只使用本地模式：本地拉左右 RTSP、本地 RKNN 推理、本地生命周期锁定；不使用远程车轮检测服务
- `wheel.left_source` / `wheel.right_source`
  - 左右车轮视频源
- `wheel.target_fps`
  - 每路非活动状态节流推理频率；`wheel.event_driven=true` 且无 Zone A 活跃轨迹时，车轮 processor 暂停推理，reader 仍持续拉流
- `wheel.active_target_fps`
  - 主轨迹进入/经过 Zone A 后的车轮推理频率；`0` 表示活动窗口内不额外节流，只受模型推理耗时限制
- `wheel.classes`
  - 当前默认：`0-25`、`25-50`、`50-75`、`75-100`
  - `type=5` 上传时直接透传为 `wheelResults[].className`
- `wheel.bind_window_seconds` / `wheel.bind_pre_start_seconds` / `wheel.bind_after_end_seconds` / `wheel.bind_require_active`
  - `bind_window_seconds` 是车轮缓存保留和候选查询窗口
  - `bind_require_active=true` 时，只在主轨迹进入/经过 Zone A 后锁定车轮结果
  - `bind_pre_start_seconds` 默认 `3.0`：拒绝早于该轨迹 wheel 激活时间太多的候选，避免上一辆车晚到结果绑定到下一辆车
  - `bind_after_end_seconds` 默认 `3.0`：拒绝轨迹结束后太晚的候选
- `wheel.bind_wait_seconds` / `wheel.bind_wait_poll_seconds`
  - `type=5` 事件生成前短暂等待车轮旁支补齐结果，默认最多 `2.0s`
  - 只在最终事件触发时等待，不影响常规帧处理
- `wheel.photo_bucket_seconds` / `wheel.photo_min_score`
  - 车轮照片批量上报（`POST /api/vehicle/wheel-photo`）的桶式参数
  - 照片在车辆生命周期内缓存和落盘，稳定桶会实时加入上传队列，最终 `type=5` 事件生成前会强制 flush 未上传照片
  - `photo_bucket_seconds` 当前配置为 `0.5`：每 0.5 秒为一个桶，每桶一张代表
  - `photo_min_score` 默认 `0.3`：低于此分数的检测不入桶
  - 桶内结果：`cleanValue` 按类型多数投票（同票倾向类型最低），代表图按检测框中心离画面中心最近选择

### `system.startup_capture_dir` / `system.manual_capture_dir`

- `system.startup_capture_dir`
  - 自动启动截图目录
  - 主流程首次真正产出结果后自动写入
- `system.manual_capture_dir`
  - 手动保留截图目录
  - 对应 Web 的 `POST /snapshot/keep`

### `system.heartbeat_path` / `system.startup_flag_path` / `system.command_dir`

- `system.heartbeat_path`
  - 推理心跳文件
- `system.startup_flag_path`
  - 启动成功标志文件
- `system.command_dir`
  - 运行期命令目录，手动截图等命令会从这里消费

默认收口规则：

- 当这 3 项留空，或仍使用旧默认值时，运行时会自动改写到：
  - `/dev/shm/cleaningcar_runtime/<device_id>/heartbeat.json`
  - `/dev/shm/cleaningcar_runtime/<device_id>/started.flag`
  - `/dev/shm/cleaningcar_runtime/<device_id>/cmd/`
- 如果你显式填写了自定义路径，则仍按自定义路径运行

### `video.debug_frame_path`

- 调试画面输出文件
- Web 控制台里的“实时调试画面”读取的就是这里的文件
- 当前 Web 页面不会自动轮询，需要手动点击获取
- 调试帧会单独保存带绘制画面，不影响事件上报默认原图策略
- 当留空或仍使用旧默认值 `/dev/shm/cleaningcar_debug.jpg` 时，运行时会自动改写到：
  - `/dev/shm/cleaningcar_runtime/<device_id>/debug.jpg`

### 旧功能：运行产物清理

- 当前默认配置和 Web 配置页已不再暴露 `storage.*`
- `storage_cleanup.py` 旧实现仍保留在代码中，但运行期默认禁用
- 后续如果需要重新启用，建议单独评审清理策略、回收范围和误删风险后再恢复

## 常用启动命令

```bash
source venv-gst/bin/activate
python run_zone_detect.py --config configs/config.json
```

临时切换 FP 后处理：

```bash
python run_zone_detect.py --config configs/config.json --fp_output_mode 6
python run_zone_detect.py --config configs/config.json --fp_output_mode 9
```

启动 Web：

```bash
./start_web_server.sh start
```

说明：

- 首次执行会自动安装或修复 `venv-gst`
- 环境已就绪时会直接拉起 Web
