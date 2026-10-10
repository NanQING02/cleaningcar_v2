# RGA3-only项目私有预加载库

状态：默认关闭，6006离线底层验证；6002真实四路灰度仍待安排。不承诺消除整机异常重启。

本库只拦截厂家gst-rockchip中的`c_RkRgaBlit`，将已验证的fd/DMA-BUF NV12到BGR任务限制为两颗RGA3。仍由系统`librga.so.2`执行原转换，没有新的resize、虚拟地址调用或CPU转换。输出buffer、stride、fence、返回码与系统接口保持一致，仅调用期间改变core并立即恢复。

## 生效范围和版本保护

- 系统必须为Linux aarch64 RK3588，multi-RGA必须为v1.3.10i。
- 系统librga及gst-rockchip SHA必须匹配代码中已验证的哈希；系统升级后拒绝加载，不能静默退回RGA2。
- 原生构建时静态断言`sizeof(rga_info_t)=696`、`offsetof(core)=188`和RGA3双核掩码为3。
- 源码、系统库、插件、构建库均在启动前校验；已存在其他LD_PRELOAD时拒绝叠加。
- 只允许来自指定插件文件的直接调用。其他调用方不修改。
- 指定插件的未验证格式、虚拟地址、旋转、缩放、混合或不合规尺寸/stride任务返回EINVAL，不自动调用RGA2或CPU。
- 初始化日志明确记录PID、ABI、mask、scope；每4096个调用记录累计guarded/rejected/foreign/failed，正常退出记录总数。

## 原生构建及预检查

在已授权的板端测试副本执行：

```bash
venv-gst/bin/python tools/rga3_guard.py build
venv-gst/bin/python tools/rga3_guard.py check
venv-gst/bin/python tools/rga3_guard.py status
```

构建产物及灰度开关放在`.local/rga3_guard/`，不纳入Git，不写`/usr/lib`。check只验证加载与自己的API，没有提交RGA任务。依赖现有gcc和厂家librga开发头文件，不安装第二份librga。

## 独立探针

```bash
venv-gst/bin/python tools/rga3_guard.py run -- venv-gst/bin/python /path/to/probe.py
```

run只注入指定进程。constructor清除LD_PRELOAD，因此再exec的FFmpeg、上传和守护子进程不会加载本库。

## 6002灰度和回滚（须在测试结果确认后执行）

先备份代码、配置、日志与PID/launch_id，并采集同时间窗四路FPS/丢帧、RGA失败job数、内存及业务基线。关闭自动重复压力测试，生产上不人为占用2GB内存。

```bash
venv-gst/bin/python tools/rga3_guard.py enable --configs configs/config.json configs/config_绕行.json
```

这一步仅保存灰度名单，不重启任何进程。下一次正常重启指定推理时，`run_zone_detect.py`在导入算法/视频模块前exec一次加载so，保持PID、进程组、launch_id及原启动参数。Web进程不变，旧Web也能启动新入口，无需重启Web。必须同时核对`ready`、`guarded>0`、`rejected=0`、失败job增量、真实四路首帧/FPS/队列和业务事件。unsupported-job、failed增加、RGA3 timeout/reset、心跳异常或显著性能下降时结束灰度。

```bash
venv-gst/bin/python tools/rga3_guard.py disable
```

关闭只影响下次启动。随后按现有方式只重启推理，动态库从新PID中消失。整个过程不替换系统库、驱动或镜像。

## 证据边界

6006合成H264/H265四路测试能够复现RGA2高地址失败并验证规避方法，不能替代6002相机、NPU推理、上传、长期内存状态和业务效果。此前1815是三个错误关键词的日志总行数，实际为605个失败job。A/B采样必须单独统计`job buffer map failed`，不能把同一job的三行错误算三次。

Rockchip官方提醒调度core配置不当有死锁风险，所以先验证、后灰度，不全局启用：https://github.com/airockchip/librga/blob/main/docs/Rockchip_Developer_Guide_RGA_CN.md
