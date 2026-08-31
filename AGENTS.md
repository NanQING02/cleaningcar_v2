# AGENTS.md

本文件给所有 AI 代码助手（Codex、Claude Code 等）接手本仓库时使用。项目说明以当前代码、`configs/config.json` 和仓库内实际文件为准。Claude Code 也会读取本文件作为项目入口。

## 工作约定

- 对话回复、执行说明、commit、PR 文案和任务总结使用中文。
- Windows 下使用 conda 时，如果工作目录、路径或输出包含中文/Unicode，优先直接使用目标环境的 `python.exe` 执行脚本；除非明确需要，不优先使用 `conda run`，避免 GBK/Unicode 编码问题。
- 环境检查顺序固定为：先 `conda env list`，再在候选环境里检查关键包是否可导入；只有 import 失败或版本不满足时，才考虑安装。
- 如果当前目录不是 Git 仓库，先明确说明，再跳过 commit/PR 操作，不继续尝试 Git 命令。

## RGA 管控口径（2026-08-26 定，接手必读）

2026-08-26 实机排查定论（完整实验报告见仓库外部文档 `RGA_4G死机排查_实验报告.md`）：
RK3588 整机死机 = 四要素叠加——RGA3 双忙（job 落 RGA2，无 IOMMU）× buffer 物理页 >4G ×
调用方以 **malloc 虚拟地址** 传 buffer × librga 1.10.x virAddr/im2d 错误路径 SIGSEGV
→ 内核 elf_core_dump BUG → D 状态卡死 → 整机慢性挂死。

据此本项目执行如下硬性口径：

1. **唯一允许的 RGA 用法**：GStreamer 解码端 BGR 直出（`mppvideodec format=BGR` direct 管线）。
   gst-rockchip 插件内部以 fd/DMA-BUF 调 RGA，提交失败仅丢帧（实测 223 次连续失败进程存活），不触发死机链。
2. **禁止**任何其他 RGA 用法：显式 `scale_rkrga`/ffmpeg_rga、`wrapbuffer_virtualaddr`+`imresize`、
   ctypes/c 直接调 librga 等。不走 fd 路径或未实机验证错误路径的 RGA 方案一律不允许合入。
3. 若未来确需新增 RGA 用法：必须先用实机探针验证目标 librga 版本在"RGA2 + >4G"下的错误路径
   不 crash 进程，且优先 importbuffer 句柄 + 绑定 RGA3 core；评估通过前不允许进入任何链路。
4. 板端 librga 以系统镜像自带为准（soname `librga.so.2`），项目不再向 `/usr/local/lib` 安装第二份。

## 项目概览

CleaningCar v2 是部署在 RK3588 板端的实时车辆检测与冲洗监测系统。主链路使用 RKNN NPU 做 FP 目标检测，使用双模型车牌识别链路识别车牌和颜色，可选启用左右车轮 RTSP 旁路检测。Web 端基于 FastAPI，负责配置管理、推理进程守护、日志、调试帧和手动截图。

项目主语言是 Python，注释和文档以中文为主。

## 常用命令

### 启动 Web 服务

```bash
./start_web_server.sh start
./start_web_server.sh stop
./start_web_server.sh restart
./start_web_server.sh status
```

`start_web_server.sh` 是对外交付入口，首次 `start` 或 `restart` 会自动安装或修复 `venv-gst/`，已就绪时直接启动 Web。

### 直接运行检测链路

```bash
source venv-gst/bin/activate
python run_zone_detect.py --config configs/config.json
python run_zone_detect.py --config configs/config.json --fp_output_mode 6
python run_zone_detect.py --config configs/config.json --fp_output_mode 9
```

### 直接调试 Web

```bash
source venv-gst/bin/activate
python -m web.server --config configs/config.json --host 0.0.0.0 --port 8000
```

### 运行测试

```bash
python -m pytest tests/
python -m pytest tests/test_events_plate_locking.py
python -m pytest tests/test_wheel_binding.py -k "test_name"
```

## 主链路

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

## 关键模块

- `cleaningcar/pipeline.py`：主调度层，串联视频 I/O、worker、跟踪、事件、车轮旁路、单车视频和运行信号。
- `cleaningcar/video_io.py`：视频读写与回退链路。RTSP 只允许 `GStreamer+mpp direct-BGR`，打开失败即按读流失败处理；离线文件可保留非 RGA 硬解链路。单车视频仅使用 `FFmpeg` 硬编，硬编不可用时不保存视频。
- `cleaningcar/worker.py`：RKNN worker 线程，负责检测推理、后处理、车牌识别和基础绘制。
- `cleaningcar/tracking.py`：车辆跟踪，默认 `vehicle_tracker_impl=bytetrack`，内部使用 Kalman 预测和匹配。
- `cleaningcar/anchor.py`：统一业务锚点估算；默认方向感知D正式，H用于多帧方向观测，L仅作调试对照和紧急回退，G暂时停用。
- `cleaningcar/events.py`：事件状态机、事件 JSON、截图、上报 payload；`type=5` 事件会附加已锁定的 `wheelResults`。
- `cleaningcar/event_trace.py`：默认关闭的事件层输入JSONL追踪，用于本地视频逐案例手测；不得记录RTSP密码或图片Base64。
- `cleaningcar/wheel.py`：左右车轮 RTSP 旁路检测，结果按车辆生命周期锁定，同侧短时间连续结果按簇归属给同一辆车。
- `cleaningcar/plate_lpr.py`：双模型车牌检测、矫正、识别和颜色解码。
- `cleaningcar/plate.py`：车牌文本规范化、合法性校验和锁定辅助。
- `zone_manager.py`：Zone A/B、流向向量、ROI 和方向判定。
- `config_manager.py`：JSON 配置加载、校验、默认值补齐和旧字段清理。
- `web/server.py`：FastAPI 路由、配置接口、截图接口、日志接口。
- `web/inference.py`：推理进程 guardian，负责启动、停止、健康检查和异常重启。
- `web/config_tiers.py`：Web 配置字段分层白名单。

## 当前模型

- 主检测模型：`models/detection/best.rknn`
- 车牌检测模型：`models/plate/plate_detect.rknn`
- 车牌识别/颜色模型：`models/plate/plate_rec_color.rknn`
- 车轮旁路模型：`models/wheel/2026.4.28CRwheel.rknn`

旧单模型车牌链路不再参与主流程，历史实现保留在 `future_modules/legacy_lpr/`。

## 配置要点

- 主配置：`configs/config.json`
- 备用配置：`configs/config_绕行.json`
- `system.device_id` 是运行时命名空间主键，多路部署必须唯一。
- `system.command_dir`、`system.heartbeat_path`、`system.startup_flag_path`、`video.debug_frame_path` 留空或使用旧默认值时，会自动收口到 `/dev/shm/cleaningcar_runtime/<device_id>/` 下。
- `video.debug_frame_path=off` 会禁用调试帧；留空表示使用命名空间默认调试帧路径。
- Web 配置分为用户参数和开发者参数，白名单在 `web/config_tiers.py`。

`configs/config.json` 当前快照：

- `video.source_mode=camera`
- `video.hw_decode=true`
- `video.decode_backend=gstreamer`；RTSP 运行时只允许 `GStreamer+mpp direct-BGR`，不回退到 FFmpeg 或 safe 管线
- `video.workers=1`
- `video.core_mask=0`
- `video.worker_core_strategy=auto`
- `video.fp_output_mode=6`
- `video.debug_frame_path=off`
- `system.performance_lock_enabled=true`
- RGA 管控：**只允许** GStreamer 解码端 BGR 直出（`mppvideodec format=BGR`）隐式使用 RGA（fd/DMA-BUF 路径，失败仅丢帧不死机）；其余任何显式 RGA 用法（ffmpeg_rga/scale_rkrga、rga_resize、wrapbuffer_virtualaddr/imresize 等）已于 2026-08-26 全部移除，禁止恢复
- NPU 分配：冲洗道主检测+车牌用 core 0，绕行道主检测+车牌用 core 1，双车轮旁路用 core 2
- `logic.no_draw=true`
- `logic.draw_plate_boxes=false`
- `logic.plate_infer_stride=2`
- `logic.enable_per_id_video=true`
- `logic.per_id_video_dir=/data/ftp/per_id`，不可写时回退到 `video_result/per_id/`
- `wheel.enabled=true`
- `wheel.event_driven=true`
- `wheel.target_fps=5.0`
- `system.api.capture_mode=path`

## 运行口径

- 本地文件视频默认只跑一遍，读到 EOF 后退出；只有 Web guardian 自动重启开启时才会再次拉起。
- 事件截图默认优先原图；调试帧是单独输出的带绘制画面。
- 当前只保留 per-id 单车视频，不再保留全局视频保存功能。
- Web 不再提供 per-id 单车录像浏览接口，但后台仍按 `logic.enable_per_id_video` 保存。
- 车轮旁路独立于主相机运行，仅在 `wheel.enabled=true` 且左右源/模型可用时启动。
- 运行产物清理代码仍保留，但 `storage_cleanup.py` 内运行期开关默认为禁用，当前不会主动删除产物。

## 输出目录

- 事件 JSON：由 `event_output_dir` 控制，当前主配置为 `events/config`
- 事件截图：由 `event_capture_dir` 控制，当前主配置为 `captures/config`
- 启动截图：`captures/startup/`
- 手动截图：`captures/manual/`
- 单车视频：优先 `logic.per_id_video_dir`，不可用时回退到 `video_result/per_id/`
- 推理日志：`logs/inference/`
- Web 日志：`web_server_<port>.log`

## 推荐阅读顺序

1. `README.md`
2. `cleaningcar/README.md`
3. `configs/README.md`
4. `cleaningcar/pipeline.py`
5. `cleaningcar/video_io.py`
6. `cleaningcar/worker.py`
7. `cleaningcar/tracking.py`
8. `cleaningcar/plate.py`
9. `zone_manager.py`
10. `cleaningcar/events.py`
