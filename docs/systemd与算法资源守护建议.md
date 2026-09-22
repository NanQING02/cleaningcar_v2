# systemd 与算法资源守护建议

> 本文仅作为运维建议，不随算法代码自动安装或修改任何生产设备的 systemd 单元。

## 1. 问题结论

3号设备此前出现的“静止时推理突然重启”，直接触发链路是：主推理进程内存上涨后被 Linux OOM Killer 杀死，`cleaningcar_v2.service` 因配置了 `Restart=always` 又把整个服务拉起。心跳异常是进程失效后的表现，不是该次重启的最初原因。

算法侧原先还存在两个长期运行风险：逐帧时延字典没有上限，锚点trace即使关闭也会持续保存逐帧去重键；实时任务队列默认可保存32张1080p BGR帧，也会放大瞬时内存。这些问题应优先在算法内约束，不能只依赖 systemd 重启掩盖。

## 2. 算法侧处理原则

- RTSP/camera实时任务队列上限设为8；离线文件保持原配置，避免影响回放吞吐。
- 结果队列超过实时任务队列一半时主动排空，尽早释放帧引用。
- 帧时延缓存最多保留4096条，按最旧条目淘汰。
- 未开启事件trace时不保存锚点逐帧去重状态；开启时也随轨迹超时回收。
- 主视频内部重连最多20次，重连期间心跳明确报告 `waiting_reader`、`reader_reconnect` 或 `reader_reopen`，Guardian将其显示为“读流降级”，不误判为算法卡死。
- 内部重连耗尽、进程异常退出或真实心跳失效后，Guardian按5、10、20秒退避拉起；10分钟内达到3次后自动熔断，不再静默循环重启。Web控制台显示失败原因并弹出告警，人工处理后点击“启动/重启”才解除熔断。

这些限制负责减少资源增长并保留故障现场。systemd仍可负责Web服务本身的进程级兜底，但不应代替算法Guardian的诊断和限流。

## 3. `cleaningcar_v2.service` 建议

建议由负责系统服务的人员在维护窗口调整并验证：

1. 将 `Restart=always` 改为 `Restart=on-failure`，避免正常人工停止或正常退出也被无条件拉起。
2. 将 `RestartSec` 调整为至少30秒，给日志和资源回收留出窗口。
3. 在 `[Unit]` 中设置 `StartLimitIntervalSec=600`、`StartLimitBurst=3`，防止Web服务自身发生重启风暴。
4. 保留 `OOMPolicy=stop`，发生OOM时停止该单元并暴露失败；不要用无上限自动重启掩盖内存问题。
5. 长期建议从 `Type=forking` 加包装脚本改为 `Type=simple`，让 systemd 直接跟踪Web主进程。切换前必须先确认启动参数、工作目录、环境变量和停止流程。
6. 暂不直接设置激进的 `MemoryMax`。先连续采集稳定版本的峰值RSS与cgroup内存，再设置 `MemoryHigh` 做预警；硬限制过低会把资源压力变成频繁OOM。

示例仅供运维人员改写，不能直接复制到生产环境：

```ini
[Unit]
Description=CleaningCar Web and inference guardian
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=3

[Service]
Type=simple
WorkingDirectory=/home/serve/cleaningcar_v2
ExecStart=/home/serve/cleaningcar_v2/venv-gst/bin/python -m web.server
Restart=on-failure
RestartSec=30
OOMPolicy=stop
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

若现场必须继续使用包装脚本，则先保留现有 `Type=forking` 和启动命令，只修改重启策略与频率限制，避免一次同时改变进程模型和守护策略。

## 4. 日志与告警建议

- Web日志和每路推理日志必须分文件；推理文件名应包含配置名、设备ID、启动时间与launch ID。
- 对 `[perf]`、`[diag]`、`[monitor]` 等例行日志限频，但异常退出、OOM、读流耗尽、重启排程和熔断记录不得限频。
- 为文件日志配置 `logrotate`，建议按天或达到100MB轮转，保留7至14份并压缩；轮转方式需按进程是否支持重新打开日志文件选择，避免直接截断仍在写入的文件。
- 监控至少包含：进程PID、最近心跳、最近有效帧时间、内部重连次数、自动重启窗口计数、熔断状态、RSS、cgroup内存、任务队列和丢帧数。

## 5. 后续现场验收

算法版本上线后建议分两阶段：

1. 先只更新算法，保持systemd不变，连续运行至少24小时，观察RSS是否趋于稳定、任务队列是否长期满载、是否出现熔断告警。
2. 再由运维人员单独调整systemd，并分别验证正常停止、算法异常退出、读流断开20次、Web进程崩溃和OOM场景。每种场景都应能从日志区分根因，不能只看到“服务已重启”。

回滚时应分别回滚算法版本和systemd单元，避免两类改动混在一起导致责任边界不清。
