# web 目录说明

`web/` 是项目的 FastAPI 管理层，负责配置页面、推理进程管理、健康检查、日志查看和手动截图接口。

当前 Web 控制台的运行口径：

- 场景切换时不再自动抓 RTSP 背景帧，避免切换配置时按钮长时间变暗
- “实时调试画面”改为点击获取，不自动轮询
- Web 端不再浏览按车辆 ID 的单车录像，但后台单车录像保存功能保留

## 目录内容

- `server.py`
  - Web 入口
  - 负责路由注册和应用启动
- `inference.py`
  - 推理 guardian
  - 负责推理启动、停止、健康检查和异常重启
- `helpers.py`
  - 配置读取、日志读取、调试帧读取等辅助逻辑
- `models.py`
  - Web 请求模型
- `config_tiers.py`
  - 配置字段分级注册表
  - 负责用户参数 / 开发者参数白名单和分层提取
- `state.py`
  - Web 运行状态和默认路径
- `templates/`
  - 页面模板
- `static/`
  - 前端静态资源

## 使用方式

推荐通过根目录脚本启动：

```bash
chmod +x start_web_server.sh
./start_web_server.sh start
```

说明：

- 首次执行会自动安装或修复 `venv-gst`
- 环境已就绪时会直接拉起 Web

如需直接调试 Web 入口，可在环境已准备好后执行：

```bash
source venv-gst/bin/activate
python -m web.server --config configs/config.json --host 0.0.0.0 --port 8000
```

## 当前配置分层口径

- 用户参数
  - 给甲方在 Web 基础面板直接修改
  - 主要包含：设备 ID、API、视频源、视频源模式、硬解、推理线程数、NPU Core Mask、车道名、左右车轮视频源、按车 ID 录像开关和目录
- 开发者参数
  - 给算法/调试/运维人员修改
  - 主要包含：FP 输出模式、逻辑阈值、车轮旁路参数、调试帧、心跳路径、输出目录、车牌绘制开关等

## 当前相关接口

- `GET /config`
  - 读取完整配置
- `GET /config/schema`
  - 读取字段分级注册表
- `GET /config/user`
  - 读取用户参数子集
- `POST /config/user`
  - 只允许保存用户参数白名单
- `POST /config/developer`
  - 只允许保存开发者参数白名单
- `GET /frame_meta` / `GET /frame`
  - 背景帧元数据与背景帧
- `GET /debug_frame_meta` / `GET /debug_frame`
  - 调试帧元数据与调试帧
- `POST /snapshot/keep`
  - 手动保留原图/叠加图截图
- `GET /inference/status`
  - 推理 guardian 状态

说明：

- 基础面板现在只走 `/config/user`
- 开发者面板现在只走 `/config/developer`
- 若用户面板提交了开发者字段，后端会直接拒绝保存
- 旧的 `/videos/per_id` Web 浏览接口已删除，不再暴露按车 ID 录像下载入口
