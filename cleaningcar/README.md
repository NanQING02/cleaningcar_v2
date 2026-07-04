# cleaningcar 模块说明

`cleaningcar/` 是项目主运行包，负责把视频输入、RKNN 推理、车牌识别、跟踪、事件、截图、运行信号和守护协作起来。

## 模块职责

- `constants.py`
  - 定义类别列表、类别阈值、车牌字符表、颜色常量和业务映射
  - 当前项目内统一以这里的车牌字符定义为准

- `runtime_config.py`
  - 读取配置并生成运行时配置
  - 将 CLI 覆盖项与默认值合并

- `video_io.py`
  - 打开视频源
  - 管理 RTSP / 文件输入
  - 负责 `FFmpeg 硬解 -> GStreamer+mpp 硬解 -> 软件解码` 与 `FFmpeg 硬编 -> GStreamer 硬编 -> FFmpeg libx264` 的统一回退
  - 负责运行时路径解析

- `vision.py`
  - 提供 IoU、点位缩放、anchor 计算等几何工具

- `fp_detect.py`
  - 负责 FP 检测模型后处理
  - 支持 `6` 输出模式和 `9` 输出模式切换

- `plate_lpr.py`
  - 双模型车牌识别主链路
  - 包含车牌检测、透视矫正、字符解码、颜色解码

- `plate.py`
  - 车牌文本规范化
  - 合法性校验
  - 车牌锁定辅助
  - 不负责旧 `lpr.rknn` 单模型推理

- `text_render.py`
  - 统一中文文本渲染层
  - 优先使用 `fonts/platech.ttf`
  - 无项目字体时按固定系统字体顺序兜底

- `tracking.py`
  - 负责车辆目标跟踪

- `events.py`
  - 负责事件状态机、事件 JSON、事件截图和事件上报
  - `type=5` 会附加 `wheelResults`
  - 只使用单车生命周期内已锁定的左右轮结果；未锁到则不附加
  - 事件截图只有真实落盘成功才写入 `captureImage`
  - 当前默认优先保存原图事件截图，而不是调试叠加图

- `runtime_signals.py`
  - 输出心跳文件、启动标志、启动截图、手动截图

- `storage_cleanup.py`
  - 保留的旧清理模块
  - 当前运行期默认禁用
  - 后续如需恢复，应单独评审回收策略和误删风险

- `monitoring.py`
  - 输出 CPU、内存、温度等运行监控信息

- `worker.py`
  - 每个 RKNN worker 线程负责：
    - 检测模型推理
    - 检测后处理
    - 双模型车牌识别
    - 基础叠框与叠字
    - 车牌框仅在 `logic.draw_plate_boxes=true` 且未启用 `logic.no_draw` 时绘制

- `wheel.py`
  - 左右车轮 RTSP 旁路检测
  - 低队列、丢旧帧、按 `wheel.target_fps` 节流推理
  - 单帧内优先选离画面中心最近的有效轮胎框
  - 生命周期内按“更居中 -> 更高分 -> 更晚命中”持续更新单车锁定轮胎结果
  - 同侧短时间连续命中的轮胎结果会整串归属给同一辆车，避免同一波结果拆给后车
  - 正式上传的车轮图片使用原图，不带调试标注

- `pipeline.py`
  - 主流程总调度层
  - 负责把读流、worker、跟踪、事件、截图、命令处理、车轮旁路和单车视频串起来
  - 当前只保留 per-id 单车视频，不再维护全局视频保存链路
  - 调试帧会单独从原始帧复制并绘制，不和事件截图共用一张图

- `cli.py`
  - 命令行入口
  - 把配置和参数组装后交给 `pipeline.py`

## 主调用关系

1. `run_zone_detect.py`
2. `cleaningcar/cli.py`
3. `cleaningcar/pipeline.py`
4. `cleaningcar/worker.py`
5. `cleaningcar/fp_detect.py` + `cleaningcar/plate_lpr.py`
6. `cleaningcar/tracking.py` + `cleaningcar/events.py`
7. `cleaningcar/runtime_signals.py` + `cleaningcar/storage_cleanup.py`

## 主链路依赖图

```mermaid
flowchart TD
    A["run_zone_detect.py"] --> B["cleaningcar/cli.py"]
    B --> C["cleaningcar/runtime_config.py"]
    B --> D["cleaningcar/pipeline.py"]

    D --> E["cleaningcar/video_io.py"]
    D --> F["cleaningcar/worker.py"]
    D --> G["cleaningcar/tracking.py"]
    D --> H["cleaningcar/events.py"]
    D --> I["cleaningcar/runtime_signals.py"]
    D --> J["cleaningcar/monitoring.py"]
    D --> K["cleaningcar/vision.py"]
    D --> L["cleaningcar/storage_cleanup.py"]
    D --> R["cleaningcar/wheel.py"]

    F --> M["cleaningcar/fp_detect.py"]
    F --> N["cleaningcar/plate_lpr.py"]
    F --> O["cleaningcar/plate.py"]
    F --> P["cleaningcar/text_render.py"]

    M --> Q["cleaningcar/constants.py"]
    N --> Q
    O --> Q
    H --> O
    H --> I
    G --> K
    N --> K
```

说明：

- `pipeline.py` 是主调度中心，负责把读流、推理、跟踪、事件和运行信号串起来
- `worker.py` 是检测与双模型车牌识别的执行层
- `plate_lpr.py` 负责双模型车牌推理，`plate.py` 只负责文本后处理和锁定
- `runtime_signals.py` 独立负责心跳、启动标志、启动截图和手动截图
- `storage_cleanup.py` 保留为旧能力占位，当前默认禁用，不作为现行主链路能力

## 当前模块级改动收口

- 车牌字符表统一到了 `constants.py`
- 中文显示新增 `text_render.py`
- 事件截图真实落盘校验收到了 `events.py`
- 运行期清理已收口为默认禁用，仅保留 `storage_cleanup.py` 旧实现
- 视频链路已收口为 FFmpeg 硬解/硬编优先，GStreamer 硬件链路次选，软件链路兜底
- 全局视频保存残留已删除，仅保留 `logic.enable_per_id_video`
- Web 不再浏览 per-id 单车录像，但 `pipeline.py` 仍保存单车录像文件
- 实时调试图与事件截图已分流：调试图带绘制，事件截图默认优先原图
- 当前主链路只保留双模型车牌流程，旧单模型 LPR 仅保留在 `future_modules/legacy_lpr/`

## 建议阅读顺序

1. `cli.py`
2. `pipeline.py`
3. `worker.py`
4. `fp_detect.py`
5. `plate_lpr.py`
6. `plate.py`
7. `events.py`
8. `runtime_signals.py`
9. `storage_cleanup.py`
10. `text_render.py`
