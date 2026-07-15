# CleaningCar v2 项目说明

CleaningCar v2 是运行在 RK3588 板端的车辆冲洗监测系统。它从主相机 RTSP 或本地视频取流，识别车辆、清洗相关目标和车牌，结合配置里的检测区、清洗区和流向规则生成事件；并可选接入左右车轮旁路摄像头，把车轮检测结果绑定到同一辆车的最终冲洗事件里。Web 端基于 FastAPI，负责配置管理、推理进程守护、日志、调试帧和手动截图。

项目主语言是 Python，注释和文档以中文为主。

## 当前实际使用的模型

- 主检测模型：`models/detection/best.rknn`
- 车牌检测模型：`models/plate/plate_detect.rknn`
- 车牌识别/颜色模型：`models/plate/plate_rec_color.rknn`
- 车轮旁路模型：`models/wheel/2026.4.28CRwheel.rknn`（仅 `wheel.enabled=true` 时使用）

旧单模型车牌链路不再参与主流程，历史实现保留在 `future_modules/legacy_lpr/`。

## 主链路速览

```text
RTSP/File -> VideoReader -> Frame Queue -> DetectWorkers (RKNN/NPU)
  -> fp_detect 后处理 -> plate_lpr 双模型车牌识别
  -> VehicleTracker(ByteTrack/Kalman) -> EventManager 状态机
  -> 事件 JSON + 截图 + 单车视频 + 上传队列
```

调用入口：

```text
run_zone_detect.py
  -> cleaningcar/cli.py
  -> cleaningcar/pipeline.py
  -> cleaningcar/worker.py
  -> cleaningcar/tracking.py + cleaningcar/events.py
```

核心入口文件：

- `run_zone_detect.py`：主检测程序启动壳层，把 CLI 转发到 `cleaningcar/cli.py`
- `config_manager.py`：读取并校验 `configs/*.json`，给 `system`/`video`/`logic` 等配置补默认值
- `zone_manager.py`：管理 Zone A、Zone B、流向向量、ROI 和方向判定

## 当前运行口径

- 当前只保留双模型车牌链路，旧单模型 LPR 不再参与主链路
- 当 `video.hw_decode=true` 时，读流顺序为：`GStreamer+mpp direct-BGR 硬解 -> FFmpeg rkmpp 硬解`；两级硬解都不可用时按读流失败处理，不再切软件解码
- 单车视频写出顺序为：`GStreamer 硬编 -> FFmpeg 硬编`；硬编不可用时不保存该段单车视频，没有软件编码兜底
- 本地文件视频默认只跑一遍，读到 EOF 后退出；只有手动勾选自动重启才会循环
- 车轮旁路独立于主相机冲洗检测运行，只在 `wheel.enabled=true` 时启用
- 主事件只有 `type=5` 会附加 `wheelResults`，且只附加该车主轨迹生命周期内已锁定的左右轮结果；轮胎图片字段为 `photoUrl` 绝对路径，不再使用 `imageBase64`
- 同一侧短时间连续命中的车轮结果会按连续簇整串归属给同一辆车，避免同一波旁路结果拆给后车
- 上传给接口的车轮图片为原图，不带调试标注
- `logic.per_id_video_dir` 留空、空白或不可写时，统一回退到 `video_result/per_id/`
- `logic.per_id_video_source=auto` 时，`logic.no_draw=true` 按主路原始解码帧写单车录像；`logic.no_draw=false` 按绘制帧写
- 全局视频保存功能已彻底删除，当前只保留 `logic.enable_per_id_video`
- Web 端不再提供按车辆 ID 的单车录像浏览，但后台仍按 `logic.enable_per_id_video` 保存
- 事件/API 截图默认优先原图；实时调试帧单独输出带绘制画面
- 运行产物清理当前默认禁用，`storage_cleanup.py` 旧实现仅保留在代码中备用

## 当前默认配置快照

以 `configs/config.json` 为准，当前仓库默认口径大致如下：

- 设备 ID：`system.device_id=RK3588-DEV`
- 视频源：RTSP，`video.source_mode=camera`
- 解码：`video.hw_decode=true`、`video.decode_backend=auto`
- 推理并发：`video.workers=1`，`video.core_mask=0`，`video.worker_core_strategy=auto`
- NPU 分配：冲洗道主检测+车牌用 core 0，绕行道主检测+车牌用 core 1，双车轮旁路用 core 2
- 板端定频：`system.performance_lock_enabled=true`，推理启动前默认尝试定频
- RGA：禁用；项目高负载场景已复现死机风险，不允许开启
- FP 后处理：`video.fp_output_mode=6`
- 画面叠加：`logic.no_draw=true`
- 车牌框绘制：`logic.draw_plate_boxes=false`
- 调试帧：`video.debug_frame_path=off`
- 车牌副链路降频：`logic.plate_infer_stride=2`
- 单车视频：`logic.enable_per_id_video=true`
- 单车视频目录：`logic.per_id_video_dir=/data/ftp/per_id`，不可写时回退到 `video_result/per_id/`
- 单车视频帧源：`logic.per_id_video_source=auto`，`logic.no_draw=true` 时走原始解码帧
- 车轮旁路：`wheel.enabled=true`，`wheel.event_driven=true`，平常只拉流不推理，Zone A 活跃轨迹触发后 `wheel.active_target_fps=0.0` 拉满推理
- 车轮照片批量上报：`system.api.wheel_photo_url`，落盘到 `system.wheel_photo_base_dir=/data/ftp`，桶式去重默认 `wheel.photo_bucket_seconds=0.5`，稳定桶实时入上传队列，最终 `type=5` 前强制 flush 未上传照片
- 检测 CSV：`video.csv=./video_result/test.csv`
- 事件截图上报格式：`system.api.capture_mode=path`
- 事件目录：`event_output_dir=events/config`
- 事件截图目录：`event_capture_dir=captures/config`

说明：

- 上述只是当前仓库默认值，运行时仍以实际配置文件和 Web 保存结果为准
- 程序优先使用 `GStreamer+mpp direct-BGR`；短测显示该路径资源占用最低。排查 RGA 问题时可临时设置 `video.gstreamer_bgr_mode=safe`，但该模式在 1080p RTSP 上帧率明显偏低。
- 涉及性能、正确性和旁路开销时，优先同时对照 `configs/config.json` 与 `cleaningcar/pipeline.py`

## 运行时命名空间

以下路径留空或仍使用旧默认值时，会自动隔离到 `/dev/shm/cleaningcar_runtime/<device_id>/` 下：

- `system.command_dir`
- `system.heartbeat_path`
- `system.startup_flag_path`
- `video.debug_frame_path`

`video.debug_frame_path=off` 表示禁用调试帧；留空表示启用命名空间默认调试帧路径。多路部署时务必保证 `system.device_id` 唯一，否则心跳、命令和调试帧可能互相覆盖。

## 快速启动

### 1. 一键启动 Web（对外交付唯一入口）

```bash
chmod +x start_web_server.sh
./start_web_server.sh start
```

推理启动前默认会尝试板端定频；如需临时关闭，可设置 `CLEANINGCAR_PERF_LOCK=0`。

`start_web_server.sh` 首次执行 `start` 或 `restart` 时会自动安装或修复 `venv-gst/`；环境已就绪时直接拉起 Web。常用动作：

```bash
./start_web_server.sh status
./start_web_server.sh restart
./start_web_server.sh stop
```

后台启动后会生成 `web_server_<port>.pid` 和 `web_server_<port>.log`，默认端口 8000。脚本直接调用 `venv-gst/bin/python`，安装完环境后不需要先激活虚拟环境。

### 2. 直接运行检测链路

```bash
source venv-gst/bin/activate
python run_zone_detect.py --config configs/config.json
python run_zone_detect.py --config configs/config.json --fp_output_mode 6
python run_zone_detect.py --config configs/config.json --fp_output_mode 9
```

### 3. 直接调试 Web

```bash
source venv-gst/bin/activate
python -m web.server --config configs/config.json --host 0.0.0.0 --port 8000
```

### 4. 甲方自行配置 systemd 开机守护（可选）

仓库不再提供自动安装脚本。如甲方需要 `systemd` 开机守护，先手工执行一次 `./start_web_server.sh start` 确认环境和 Web 正常，再自行创建 `/etc/systemd/system/cleaningcar-web.service`：

```bash
cat <<'EOF' | sudo tee /etc/systemd/system/cleaningcar-web.service >/dev/null
[Unit]
Description=CleaningCar Web Service
After=network.target

[Service]
Type=simple
User=<运行用户>
Group=<运行组>
WorkingDirectory=/path/to/cleaningcar-delivery
Environment=PYTHONUNBUFFERED=1
ExecStart=/path/to/cleaningcar-delivery/venv-gst/bin/python -m web.server --config /path/to/cleaningcar-delivery/configs/config.json --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now cleaningcar-web
systemctl status cleaningcar-web --no-pager
```

说明：

- `systemctl stop cleaningcar-web` 属于人工停止，不会自动重启
- 直接 `kill` / `kill -9` Web 主进程会被 `systemd` 视为异常退出并拉起
- `ExecStart` 请按实际交付目录、配置文件路径和运行用户替换

## 目录与文件

- `cleaningcar/`：主运行包，核心是 `pipeline.py`
- `web/`：FastAPI 管理层和推理 guardian，核心是 `server.py`、`inference.py`、`config_tiers.py`
- `configs/`：运行配置，主配置 `configs/config.json`，备用 `configs/config_绕行.json`
- `models/`：当前运行所需 RKNN 模型
- `fonts/`：中文绘制字体，板端部署必须带 `fonts/platech.ttf`
- `packages/`：板端安装 RKNN Lite 等依赖时使用的离线包
- `future_modules/`：历史或预留实现，不参与当前主链路
- `docs/`：项目说明与部署验收文档

根目录脚本：

- `start_web_server.sh`：对外交付唯一入口，支持 `start|stop|restart|status`
- `install_runtime_venv.sh`：内部运行环境安装 helper，由 `start_web_server.sh` 自动调用，甲方一般不需要手工执行

## 关键输出目录

- 事件 JSON：`events/<配置名>/`（当前主配置为 `events/config`）
- 事件截图：`captures/<配置名>/`（当前主配置为 `captures/config`）
- 启动截图：`captures/startup/`
- 手动截图：`captures/manual/`
- 单车视频：优先写入 `logic.per_id_video_dir`，不可写时回退到 `video_result/per_id/`
- 检测 CSV：`./video_result/test.csv`
- 推理日志：`logs/inference/`
- Web 日志：`web_server_<port>.log`
- Web pid：`web_server_<port>.pid`
- 上传队列：`<event_output_dir>/upload_queue.db`
- 心跳文件：由 `system.heartbeat_path` 控制
- 启动标志：由 `system.startup_flag_path` 控制
- 命令目录：由 `system.command_dir` 控制

## Web 接口速览

- `GET /config`：读取完整配置
- `GET /config/user`、`POST /config/user`：读取/保存用户参数白名单
- `POST /config/developer`：保存开发者参数白名单
- `GET /config/schema`：查看字段分层
- `POST /inference/start|stop|restart`：控制推理进程
- `GET /inference/status`、`GET /inference/health`：查看 guardian 和推理健康状态
- `GET /debug_frame_meta`、`GET /debug_frame`：手动读取最新调试帧
- `POST /snapshot/keep`：保留手动截图
- `GET /logs/*`：查看推理、事件和检测日志

多路运行时，辅助接口要带 `?key=<配置文件名>`。`POST /snapshot/keep` 是异步入队，不是同步返回图片，真正的截图结果在同目录下的 `*.done.json` / `*.failed.json`。

## 现场排查顺序

1. 先看 Web 是否启动：`./start_web_server.sh status`
2. 看推理健康：`GET /inference/health?key=<配置名>`
3. 看 guardian 状态：检查 `restart_count` 是否持续增长
4. 看心跳文件是否持续更新
5. 看 `logs/inference/` 和 `web_server_8000.log`
6. 看 `events/<配置名>` 是否生成事件 JSON
7. 看事件里的 `captureImage` 是否非空，并确认文件真实存在
8. 若中文框或标签异常，确认 `fonts/platech.ttf` 是否随包部署
9. 若车轮结果缺失，确认 `wheel.enabled`、左右 RTSP 源、车轮模型路径和旁路摄像头画面

## 易踩点

- 根目录文档和代码都以 UTF-8 保存；Windows PowerShell 默认编码可能把中文显示成乱码，读取时请显式使用 UTF-8
- `captureImage` 只表示截图文件真实落盘；对外上报是路径还是 base64 由 `system.api.capture_mode` 决定
- 本地文件视频默认跑完一遍就退出；如果 guardian 自动重启开启，看起来会像循环跑
- 全局视频保存字段已经移除；不要再使用旧的 `video.save_video` 或 `logic.enable_global_video`
- 运行产物清理实现仍在 `storage_cleanup.py`，但当前运行期开关默认禁用，不会主动删除产物
- `future_modules/` 下是历史或预留模块，默认不参与当前主链路

## 部署提醒

- 板端部署时必须一并带上 `fonts/platech.ttf`
- `requirements.txt` 已包含 `Pillow`
- 手工停 Web 请使用 `./start_web_server.sh stop`
- 若甲方自行配置 `systemd`，人工停服务请使用 `systemctl stop cleaningcar-web`
- 当前不对外开放清理策略配置项；`storage_cleanup.py` 仍保留旧实现，但运行期默认禁用

## 代码阅读路线

如果需要读代码，建议按下面顺序：

1. `run_zone_detect.py`
2. `cleaningcar/cli.py`
3. `cleaningcar/runtime_config.py`
4. `config_manager.py`
5. `cleaningcar/pipeline.py`
6. `cleaningcar/video_io.py`
7. `cleaningcar/worker.py`
8. `cleaningcar/fp_detect.py`
9. `cleaningcar/plate_lpr.py`
10. `cleaningcar/plate.py`
11. `cleaningcar/tracking.py`
12. `zone_manager.py`
13. `cleaningcar/events.py`
14. `cleaningcar/wheel.py`
15. `web/server.py`
16. `web/inference.py`

## 推荐阅读顺序

建议其他 agent 或后续接手者按下面顺序建立上下文：

1. `README.md`（本文）
2. `AGENTS.md`（AI 代码助手工作约定，与本文档同源）
3. `cleaningcar/README.md`
4. `configs/README.md`
5. `docs/README.md`
6. `docs/部署验收/运行路径说明.md`
7. `docs/部署验收/心跳与截图对接说明.md`
8. `docs/部署验收/板端验收清单.md`

补充说明：

- `docs/部署验收/` 里同时包含"当前仍有效"的部署文档和"带时间/环境前提"的专项文档
- 遇到 `甲方MPP环境与补齐要求.md`、`部署前性能评估与低风险优化建议.md` 这类文档时，要先看文首说明，再判断是不是当前场景
