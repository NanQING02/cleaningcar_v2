# configs 目录说明

`configs/` 用于存放项目运行配置。当前建议以 `configs/config.json` 作为正式主配置，其他配置只保留为备份或场景切换。

## 当前配置文件

- `config.json`
  - 默认主配置
- `config_绕行.json`
  - 备用 / 绕行配置

两份正式配置都采用当前车牌基线：INT8 车牌检测 + FP 识别/颜色，`logic.plate_infer_stride=2`。

| 配置 | 车道 | 主路 Core | 车牌 Core | 车辆门控 | per-id 录像 | 录像画面 | 车轮旁路 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `config.json` | 冲洗 | 0 | 0 | `true` | 开启 | `annotated` | 开启 |
| `config_绕行.json` | 绕行 | 1 | 1 | `false` | 开启 | `annotated` | 关闭 |

其中 `annotated` 是单车录像的完整调试画面，包含车辆/车牌框、Zone、轨迹状态、方向和 H/D/L 锚点；若只需要干净画面，将对应配置的 `logic.per_id_video_source` 改为 `raw`。

## 正式路径默认值

路径仍可按部署现场修改；首次部署或恢复默认时，以两份正式配置中的地址为准：

| 项目 | `config.json`（冲洗） | `config_绕行.json`（绕行） |
| --- | --- | --- |
| 主视频源 | `rtsp://admin:***@192.168.1.111:8557/h264` | `rtsp://admin:***@192.168.1.222:8557/h264` |
| 事件 JSON | `events/config` | `events/config_绕行` |
| 事件截图 | `/data/ftp/event_captures/config` | `/data/ftp/event_captures/config_绕行` |
| 单车录像 | `/data/ftp/per_id` | `/data/ftp/per_id` |
| 运行指标 | `/dev/shm/metrics_laneA.json` | `/dev/shm/metrics_bypass.json` |

冲洗配置额外使用左右车轮源 `192.168.1.20`、`192.168.1.21`；绕行配置不启动车轮旁路。不要在文档或新配置中写入真实密码。

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
- `video.source_mode`：正式 RTSP 配置固定为 `camera`；离线录像测试固定为 `file`
- 当输入被识别为本地文件时，会按文件源处理，默认跑完一遍后退出

### `video.hw_decode`

- 正式配置固定为 `true`；RTSP/camera 强制使用 `GStreamer+mpp direct-BGR`，不提供运行时切换开关
- 两份现场配置固定 `video.decode_backend=gstreamer`。离线文件可按需要使用 `auto`/`gstreamer`/`ffmpeg`，但不得使用任何 RGA 后端
- RGA 管控（2026-08-26）：原 `ffmpeg_rga` 后端已移除，配置段会被丢弃；RTSP/camera 强制归一到 `gstreamer + direct-BGR`，离线文件的无效后端归一到 `auto`
- 若两级硬解都不可用，不再切软件解码；主链路会按读流失败处理并重连或退出

### `video.fp_output_mode`

`FP` 指主检测模型输出的历史后处理名称，不是车牌模型的量化精度开关。

- 当前车牌链路已固定为 **INT8 检测 + FP 识别/颜色**；模型选择不由该字段控制
- 主检测后处理固定为模式 `6`：只使用 `box + class`，不读取 score 分支
- 不启用模式 `9`，因为 score 分支会再次压缩置信度，干扰后续类别阈值
- `video.fp_output_mode=6` 保留在正式 JSON 中作为模型 schema 基线，不在 Web 或 CLI 开放调整

### `logic.per_id_video_dir`

- 单车视频输出目录
- 留空、只填空白、或目录不可写时，会自动回退到 `video_result/per_id/`

### `logic.enable_per_id_video`

- 当前只保留单车视频留存开关 `logic.enable_per_id_video`
- 单车视频仅使用 `FFmpeg` 硬编写出；FFmpeg 硬编不可用时不保存该段单车视频
- 单车录像画面只由 `logic.per_id_video_source` 控制：`annotated` 写完整调试帧，`raw` 写干净原始帧，`auto` 为兼容值并按 `raw` 处理
- `annotated` 模式会统一启用车辆框、车牌框/关键点、Zone、轨迹状态、方向、H/D/L 锚点、水流框和双模型车牌结果绘制，不再由多个 `debug_*` 参数分别拼装 per-id 画面
- `logic.track_lost_grace_seconds=4.0` 按视频源 FPS 换算锚点消失确认与 tracker 丢失保留帧数；25 FPS 时为100帧。连续4秒没有可用锚点后触发 `type=5`，Zone A 内消失会标记异常
- `logic.min_track_frames_for_type1=5`：车辆在 Zone A 内连续稳定5帧后触发 `type=1`；若先确认进入 Zone B，会严格按 `type1 -> type2` 顺序补齐
- `logic.water_confirm_frames=3`：`type=2` 成立后，只有在 Zone B 内连续3帧识别到水目标才触发 `type=3`
- `logic.min_zone_b_dwell_seconds_for_type4=0.5`：确认离开 Zone B 且区内停留至少0.5秒才触发 `type=4`；Zone B 离开同时使用动态边界缓冲和连续帧防抖，避免检测框变化导致锚点瞬时跳出
- `type=5` 只要求车辆已经满足 `type=2` 且锚点连续消失4秒；低置信度、缺少中间事件等问题只写入异常原因，不再阻止最终记录发送
- 车型在 `type=2` 前持续累计基础票，`type=2` 到 `type=4` 阶段的车型票额外加权；在 `type=4` 正式冻结，没有 `type=4` 时在 `type=5` 前冻结。冻结后不允许短串识别覆盖，多 tracker 生命周期交接会继承车型票和锁定状态
- `logic.event_trace_enabled=false` 默认关闭事件输入追踪；本地视频手测时可临时开启
- `logic.event_trace_dir=event_traces` 控制追踪输出根目录，相对路径按项目根目录解析
- `logic.event_trace_queue_size=4096` 控制异步JSONL队列；队列溢出数量会写入 `summary.json`
- Web 端不再浏览这些单车录像，但后台仍会继续保存
- 旧的全局视频保存字段 `video.save_video`、`logic.enable_global_video` 已彻底删除，不再生效

### 参数收口建议

per-id 录像已不再依赖分散的画面参数；画面只由 `logic.per_id_video_source` 选择。以下字段也不再是正式配置参数：

- `logic.no_draw`、`logic.draw_plate_boxes`
- `logic.debug_overlay`、`logic.debug_track_state`、`logic.debug_anchor_points`、`logic.debug_water_boxes`
- `logic.track_timeout_frames`、`logic.track_max_age`
- `logic.plate_draw_stable_only`、`logic.per_id_type6_require_plate_candidate`

旧配置中出现上述字段时，ConfigManager 会忽略或清除，不参与运行逻辑。

以下字段是正式运行应固定的基线，不建议频繁调整：

- `video.decode_backend=gstreamer`、RTSP `source_mode=camera`、主/车牌 NPU Core 分配
- `logic.plate_infer_stride=2`、`logic.per_id_video_source`、`logic.enable_per_id_video`
- `logic.zone_a_margin_*`、`logic.zone_a_*_hits`、`logic.track_lost_grace_seconds` 等已验证的空间和生命周期参数

以下旧能力已经删除，不应重新加入配置：`video.save_video`、`logic.enable_global_video`、显式 RGA 后端，以及远程车轮检测服务字段。

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

### 未来功能：运行产物清理

- 当前运行期不加载清理器
- 实现暂存于 `future_modules/storage_cleanup.py`
- 后续重新启用前，需要单独评审清理策略、回收范围和误删风险

## 常用启动命令

```bash
source venv-gst/bin/activate
python run_zone_detect.py --config configs/config.json
```

主检测后处理固定为 mode 6，不提供运行时切换参数。

启动 Web：

```bash
./start_web_server.sh start
```

说明：

- 首次执行会自动安装或修复 `venv-gst`
- 环境已就绪时会直接拉起 Web
