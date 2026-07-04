# 甲方 MPP 环境与补齐要求

本文档汇总甲方 RK3588 板端在 Rockchip MPP 多媒体环境上的差异、对项目的影响、对板商/BSP 的正式补齐要求和验收口径。可直接作为对外发出的依据。

文档时间基准：`2026-03-26`（差异对比），最后整理：`2026-06-14`。

> 阅读前说明
>
> - 本文针对某一类甲方板端环境问题，不是当前仓库所有部署场景的统一结论。
> - 当前代码在 `cleaningcar/video_io.py` 中会先尝试 `FFmpeg` 硬解，失败后再尝试 `GStreamer+mpp`，最后才回退到软件解码。
> - 单车视频写出顺序是 `FFmpeg 硬编 -> GStreamer 硬编 -> FFmpeg libx264`。
> - 因此，本文中建议显式关闭 `hw_decode` 的前提是：你已经确认目标板既缺少可用的 `ffmpeg rkmpp`，也缺少 `mppvideodec` 或 Rockchip GStreamer MPP 插件。

## 1. 当前项目对硬解链路的依赖

本项目的硬解入口依赖：

- `video.hw_decode=true`
- `ffmpeg` 提供可用的硬件解码器，优先 `*_rkmpp`
- 若 `ffmpeg` 硬解不可用，OpenCV 仍需支持 `GStreamer`，且 GStreamer 存在 `mppvideodec`
- 系统存在 `libgstrockchipmpp.so` 和 `librockchip_mpp.so`

任何一环缺失，读流就会退到软件解码；硬编同理。

## 2. 设备差异对照

| 对比项 | 实验室设备 | 甲方设备 | 判断 |
| --- | --- | --- | --- |
| 系统 | `Ubuntu 24.04.1` | `Ubuntu 20.04.6 LTS`（内核 `Linux 5.10.226`，`aarch64`） | 版本不一致 |
| GStreamer | `1.24.2` | `1.16.3` | 版本不一致 |
| `gst-inspect-1.0 mppvideodec` | 成功，能输出插件详情 | 失败，返回 `No such element or plugin 'mppvideodec'` | 甲方缺插件 |
| GStreamer Rockchip MPP 插件文件 | `/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstrockchipmpp.so` | 未找到 | 甲方缺插件文件 |
| OpenCV 的 GStreamer 支持 | 本轮输出未重复采集 | `YES (1.16.2)` | 甲方该项已满足 |
| `librockchip_mpp.so` 运行库 | 已存在 | 已存在（`librockchip-mpp1 1.5.0-1`） | 两端均具备运行库 |
| `ffmpeg -hide_banner -decoders \| grep rkmpp` | 本轮输出未体现 | 未搜到 | 甲方缺当前优先链路能力 |
| `gst-launch-1.0 ... ! mppvideodec ! ...` | 可启动 | 直接报 `no element "mppvideodec"` | 甲方 GStreamer 链路断 |
| 已装相关包 | `gstreamer1.0-rockchip1`、`librockchip-mpp1`、`rockchip-multimedia-config` | `librockchip-mpp1`、`librockchip-mpp-dev`、`librockchip-vpu0`、`gstreamer1.0-libav`、`gstreamer1.0-plugins-{base,good,bad}`、`ffmpeg` 等 | 甲方装了通用包但没装 Rockchip GStreamer 插件包 |

实验室设备已经能正常加载 `mppvideodec`，可证明备用 `GStreamer+mpp` 链路可用；甲方设备虽然装了 MPP 运行库和 OpenCV GStreamer 支持，但缺 GStreamer Rockchip MPP 插件层，同时 `ffmpeg` 也未提供 `rkmpp`。

## 3. 归因结论

1. 实验室设备存在 `libgstrockchipmpp.so`，`mppvideodec` 可被 GStreamer 正常识别，备用 `GStreamer+mpp` 链路可用。
2. 甲方设备仅安装了 `librockchip-mpp1` 等底层运行库，但未提供 Rockchip GStreamer MPP 插件文件。
3. 甲方设备的 `ffmpeg` 也未提供 `rkmpp` 编解码能力，因此当前代码无法命中 `FFmpeg` 优先硬解/硬编链路。
4. 所以甲方设备虽然"有 MPP 库"，但没有"项目当前真正可调用的硬件加速编解码能力"。
5. 这属于板端 BSP/多媒体环境交付不完整，不属于 Python 项目改代码能解决的问题。

不能直接拷实验室插件的原因：

- 实验室设备是 `Ubuntu 24.04`，甲方设备是 `Ubuntu 20.04`
- 两边 GStreamer 主版本和系统依赖不同
- 直接混装存在 ABI 不匹配、插件注册失败、运行时崩溃的风险

因此，MPP 环境问题应按系统级多媒体环境处理，而不是按 Python 项目文件处理。

## 4. 当前临时处理口径

在板商补齐环境之前，项目部署建议继续保持：

```json
"video": {
  "hw_decode": false
}
```

原因不是项目代码不支持，而是当前甲方板端同时缺少 `FFmpeg rkmpp` 与 Rockchip GStreamer MPP 插件层。本轮先按软件解码部署，不阻塞业务功能、稳定性和事件链路验收；硬件加速链路作为甲方环境增强项单独推进。

## 5. 对板商/BSP 的补齐要求

请板商针对当前甲方板端环境（`Ubuntu 20.04.6 + Linux 5.10.226 + aarch64 + RK3588 BSP + GStreamer 1.16.3 + librockchip-mpp1 1.5.0-1`）补齐与之匹配的 `FFmpeg rkmpp` 能力和 Rockchip GStreamer MPP 插件层。

### 5.1 必须提供的资源

- **`FFmpeg rkmpp` 资源**：支持 `rkmpp` 的 `ffmpeg`，安装后 `ffmpeg -hide_banner -decoders | grep rkmpp` 能搜到可用解码器；编码器至少能搜到一个项目可用的 H.264 编码器，优先 `h264_rkmpp`。
- **Rockchip GStreamer MPP 插件包**：安装后系统可识别 `mppvideodec`，GStreamer 插件目录内存在 `libgstrockchipmpp.so` 或功能等价插件文件，本地 H.264 文件可通过 `GStreamer+mpp` 跑通硬解。
- **依赖资源**：上述两套链路所需的全部依赖资源（对应版本的 Rockchip MPP 运行库、FFmpeg 依赖库、GStreamer 依赖库、必要的配置文件和安装后处理步骤）。
- **安装说明**：资源包名称、版本、适配的系统/BSP、安装命令、是否需要重启、安装后验证命令。
- **版本对应关系说明**：板卡型号、BSP 版本、内核版本、Ubuntu 版本、FFmpeg 版本、GStreamer 版本、Rockchip MPP 版本、GStreamer Rockchip 插件版本，并明确承诺本次交付与上述版本组合兼容。

### 5.2 推荐交付形式

任选其一：

- 完整板端系统镜像
- 完整 rootfs / BSP 多媒体增量包
- 完整 deb 包集合

不接受以下交付方式：

- 只回复"环境支持 mpp"但不提供安装资源
- 只回复"库已经有了"但不能提供 `rkmpp` 或 `mppvideodec`
- 只提供单个 `.so` 文件让现场手工拷贝
- 不说明适配的系统/BSP 版本
- 不提供安装说明和验收方法

### 5.3 建议厂商一并提供

- 一份最小测试视频、最小验证命令和预期输出样例，方便现场快速确认资源生效
- 若某项能力不支持，请明确书面说明原因，避免现场继续在"插件缺失"和"ffmpeg 能力缺失"之间反复排查

## 6. 厂商交付后的验收标准

### 6.1 FFmpeg 能力

```bash
ffmpeg -hide_banner -decoders | grep rkmpp
ffmpeg -hide_banner -encoders | grep -E "rkmpp|v4l2m2m|omx"
```

通过标准：

- 解码器列表能看到 `rkmpp` 相关项
- 编码器列表至少能看到一个项目可用的 H.264 编码器，优先 `h264_rkmpp`

### 6.2 GStreamer 插件

```bash
gst-inspect-1.0 mppvideodec
gst-inspect-1.0 | grep -Ei "mpp|rockchip"
find /usr/lib /usr/lib/aarch64-linux-gnu /usr/local/lib -path "*gstreamer-1.0*" -type f | grep -Ei "rockchip|mpp"
```

通过标准：

- `gst-inspect-1.0 mppvideodec` 能输出插件详情
- 能看到 Rockchip MPP 插件信息
- 能找到 `libgstrockchipmpp.so` 或同等作用插件文件

### 6.3 最小本地文件硬解

```bash
gst-launch-1.0 -v \
  filesrc location="/home/hinlink/视频/黑夜.mp4" ! \
  qtdemux ! h264parse ! mppvideodec ! \
  videoconvert ! fpsdisplaysink video-sink=fakesink sync=false
```

通过标准：

- 管线能启动
- 不再报 `no element "mppvideodec"`
- 能正常跑完或持续输出 FPS

### 6.4 项目内实际启用

启用 `video.hw_decode=true` 后，项目日志必须出现以下任一内容：

- `[reader] Using FFmpeg hardware decoder ...`
- `[reader] Using GStreamer+mpp file pipeline for ...`
- `[reader] Using GStreamer+mpp RTSP TCP pipeline for ...`

若未出现，视为板端环境仍未真正满足项目所需硬解链路。

## 7. 可选方案

### 方案 1：更换系统并对齐实验室环境（推荐）

- 将甲方设备更换为与实验室一致或兼容的系统镜像
- 至少对齐到同时支持 `ffmpeg rkmpp` 与 `gstreamer1.0-rockchip1` 的系统和软件源

### 方案 2：保留 Focal，由厂家补齐 FFmpeg + GStreamer 硬件加速链路

若甲方设备必须继续使用 `Ubuntu 20.04 (Focal)`，则需要由厂家或 BSP 供应方针对 `Ubuntu 20.04 + GStreamer 1.16.3 + 当前 BSP + librockchip-mpp1` 补齐 `ffmpeg` 的 `rkmpp` 能力，并编译交付可用的 `mppvideodec` 插件。验收口径与方案 1 相同。

## 8. 可直接发给板商/厂商的简版说明

### 给板商/BSP 的简版

> 2026-03-26 现场对比结果显示：甲方 RK3588 设备虽然已安装 `librockchip-mpp1`、`librockchip-mpp-dev`、`librockchip-vpu0`，且 OpenCV 已支持 GStreamer，但 `gst-inspect-1.0 mppvideodec` 返回 `No such element or plugin 'mppvideodec'`，GStreamer 插件目录下也未找到 Rockchip MPP 插件文件，最小硬解命令 `gst-launch-1.0 ... ! mppvideodec ! ...` 直接失败；同时 `ffmpeg -hide_banner -decoders | grep rkmpp` 与 `ffmpeg -hide_banner -encoders | grep rkmpp` 也未搜到可用项。请提供与当前 `Ubuntu 20.04.6 + Linux 5.10.226 + GStreamer 1.16.3 + RK3588 BSP` 匹配的 `FFmpeg rkmpp` 能力和 Rockchip GStreamer MPP 插件包或完整多媒体环境，至少补齐 `h264_rkmpp` / `mppvideodec` 及其依赖，并确保现场命令能够通过验收。

### 给厂商的简版

> 请按当前甲方 RK3588 板端环境提供与 `Ubuntu 20.04.6 + Linux 5.10.226 + GStreamer 1.16.3 + aarch64` 匹配的 `FFmpeg rkmpp` 资源和 Rockchip GStreamer MPP 插件资源。安装后必须能让项目优先命中 `FFmpeg` 硬解/硬编，并在缺少 `FFmpeg` 命中时可退到 `mppvideodec`。请确保 `ffmpeg -hide_banner -decoders | grep rkmpp`、`ffmpeg -hide_banner -encoders | grep -E "rkmpp|v4l2m2m|omx"`、`gst-inspect-1.0 mppvideodec` 和 `gst-launch-1.0 ... ! mppvideodec ! ...` 都能通过，并同时提供完整依赖、安装说明和版本对应关系。仅回复"环境支持"或只给零散 `.so` 文件，不视为完成交付。
