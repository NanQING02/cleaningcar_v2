# RGA 运行默认策略

## 当前默认值

- `video.rga_enable=false`
- `run_zone_detect.py` 启动时会在导入推理管线前把它映射成环境变量 `CLEANINGCAR_RGA_DISABLE=1`
- 这意味着默认运行路径会使用 `cv2.resize`

## 为什么这样改

- 双路多 worker 场景下，RGA resize 已经复现过不稳定调用
- 小尺寸车牌 ROI 走 CPU resize 的性能损失很小，但稳定性收益明显
- 默认关闭比默认开启更符合当前已验证的稳定工作点

## 仍然保留的能力

- 如需验证或重新启用 RGA，可在配置中显式设置 `video.rga_enable=true`
- 即使启用后，过小的 source/destination 尺寸仍会直接回退到 `cv2.resize`
- RGA 调用现在会串行化，并在回退或失败时打印更完整的诊断信息

## 建议使用方式

- 生产或准生产双路任务：保持 `video.rga_enable=false`
- 重新验证 RGA 时：只在压测环境开启，并结合双路、多 worker、录像开启三种条件一起回归
