# future_modules/acceleration 目录说明

这个目录存放当前未接入主链路的加速或实验模块。

## 当前文件

（空。原 `rga_resize_plugin.py` 已于 2026-08-26 删除——其实现
`wrapbuffer_virtualaddr_t + imresize_t` 为实机验证的 RGA 死机触发组合
（malloc 虚拟地址 + RGA2 + >4G → librga 错误路径 SIGSEGV → 内核 core dump
BUG → 整机挂死），按 RGA 管控口径永久移除，不允许以任何形式恢复。）

## RGA 管控口径（2026-08-26 定）

本项目**只允许** GStreamer 解码端 BGR 直出（`mppvideodec format=BGR`，
fd/DMA-BUF 路径）隐式使用 RGA；其余一切显式 RGA 用法（含本目录任何未来
加速模块）未经重新评估与实机错误路径验证，一律不允许接入。
