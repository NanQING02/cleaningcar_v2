# 甲方上报 JSON 模板（逐字段版）

> 本文档给出每种上报事件的**完整 JSON 示例**与**逐字段读取位置**，供甲方平台对接调整。
> 所有示例字段均与当前代码实际构造的 payload 逐字段一致。
> 接口：`POST {system.api.url}`（默认 `/api/vehicle/wash-event`），`Content-Type: application/json`，可选 `Authorization: Bearer <token>`。
> 同一辆车的 type1-6 使用同一个 `id`，严格按阶段顺序串行到达。

## 通用说明

- 示例中的时间均为 `YYYY-MM-DD HH:MM:SS` 格式的板端实时本机时间字符串
- `captureImage`、`photoUrl` 均为板端绝对路径字符串，**不按 base64 解码**
- 字段名区分大小写；除标注"条件字段"外，其余字段在对应 type 中**总是存在**

---

## type=1 进入检测区

```json
{
  "id": "RK3588-DEV-1759178400123-a1b2c3d4e5",
  "type": 1,
  "captureTime": "2026-09-30 10:15:23",
  "captureImage": "/data/ftp/event_captures/config/20260930/10/20260930_101523_RK3588-DEV_12_t1_f3120.jpg",
  "lane": "冲洗",
  "plateNumber": "苏A12345",
  "plateConfidence": 0.912,
  "plateColor": "蓝色",
  "plateColorConfidence": 0.95,
  "vehicleType": "渣土车",
  "vehicleTypeConfidence": 0.87,
  "plateIsGuess": false,
  "plateRecognitionAbnormal": false
}
```

## type=2 进入冲洗区

字段集合与 type=1 完全相同，仅 `type` 为 2、时间和截图不同：

```json
{
  "id": "RK3588-DEV-1759178400123-a1b2c3d4e5",
  "type": 2,
  "captureTime": "2026-09-30 10:15:31",
  "captureImage": "/data/ftp/event_captures/config/20260930/10/20260930_101531_RK3588-DEV_12_t2_f3320.jpg",
  "lane": "冲洗",
  "plateNumber": "苏A12345",
  "plateConfidence": 0.934,
  "plateColor": "蓝色",
  "plateColorConfidence": 0.97,
  "vehicleType": "渣土车",
  "vehicleTypeConfidence": 0.91,
  "plateIsGuess": false,
  "plateRecognitionAbnormal": false
}
```

## type=3 冲洗确认（首次检测到冲洗目标）

字段集合同上，`type` 为 3，并携带冲洗开始时间：

```json
{
  "id": "RK3588-DEV-1759178400123-a1b2c3d4e5",
  "type": 3,
  "captureTime": "2026-09-30 10:15:33",
  "captureImage": "/data/ftp/event_captures/config/20260930/10/20260930_101533_RK3588-DEV_12_t3_f3370.jpg",
  "lane": "冲洗",
  "plateNumber": "苏A12345",
  "plateConfidence": 0.934,
  "plateColor": "蓝色",
  "plateColorConfidence": 0.97,
  "vehicleType": "渣土车",
  "vehicleTypeConfidence": 0.91,
  "plateIsGuess": false,
  "plateRecognitionAbnormal": false,
  "washStartTime": "2026-09-30 10:15:33"
}
```

## type=4 离开冲洗区

字段集合同上，`type` 为 4：

```json
{
  "id": "RK3588-DEV-1759178400123-a1b2c3d4e5",
  "type": 4,
  "captureTime": "2026-09-30 10:16:20",
  "captureImage": "/data/ftp/event_captures/config/20260930/10/20260930_101620_RK3588-DEV_12_t4_f4780.jpg",
  "lane": "冲洗",
  "plateNumber": "苏A12345",
  "plateConfidence": 0.951,
  "plateColor": "蓝色",
  "plateColorConfidence": 0.98,
  "vehicleType": "渣土车",
  "vehicleTypeConfidence": 0.93,
  "plateIsGuess": false,
  "plateRecognitionAbnormal": false,
  "washStartTime": "2026-09-30 10:15:33"
}
```

## type=5 冲洗闭环（完整示例，含车轮结果）

```json
{
  "id": "RK3588-DEV-1759178400123-a1b2c3d4e5",
  "type": 5,
  "captureTime": "2026-09-30 10:16:25",
  "captureImage": "/data/ftp/event_captures/config/20260930/10/20260930_101625_RK3588-DEV_12_t5_f4890.jpg",
  "lane": "冲洗",
  "plateNumber": "苏A12345",
  "plateConfidence": 0.951,
  "plateColor": "蓝色",
  "plateColorConfidence": 0.98,
  "vehicleType": "渣土车",
  "vehicleTypeConfidence": 0.93,
  "plateIsGuess": false,
  "plateRecognitionAbnormal": false,
  "washStartTime": "2026-09-30 10:15:33",
  "washEndTime": "2026-09-30 10:16:20",
  "videoEndTime": "2026-09-30 10:16:25",
  "totalWashDuration": 47.12,
  "manualWashDuration": 18.32,
  "cleaningTableWashDuration": 38.96,
  "cleanliness": 0,
  "videoDuration": 62.0,
  "direction": 5,
  "directionLabel": "正向前出",
  "wheelResults": [
    {
      "side": "left",
      "captureTime": "2026-09-30 10:16:21",
      "photoUrl": "/data/ftp/box/20260930/10/143025_12_left_1.jpg",
      "className": "25-50"
    },
    {
      "side": "right",
      "captureTime": "2026-09-30 10:16:22",
      "photoUrl": "/data/ftp/box/20260930/10/143025_12_right_1.jpg",
      "className": "75-100"
    }
  ]
}
```

### type=5 超时强制闭环时的条件字段（仅该场景携带）

15 分钟未正常闭环时 type5 会额外携带（正常过车**不含**这两个字段）：

```json
{
  "forceVideoStop": true,
  "forcedCompletionReason": "type2_dwell_timeout_15m"
}
```

### type=5 异常场景的条件字段（仅 `abnormalReasons` 非空时携带）

```json
{
  "isAbnormal": true,
  "abnormalReason": "TRACK_LOST_IN_ZONE_A_TIMEOUT|PLATE_NOT_LOCKED"
}
```

## type=6 单车录像结束通知

```json
{
  "id": "RK3588-DEV-1759178400123-a1b2c3d4e5",
  "type": 6,
  "lane": "冲洗",
  "perIdVideoEnabled": true
}
```

## 车轮照片批量接口 `POST /api/vehicle/wheel-photo`

车辆冲洗过程中稳定照片桶实时批量上报（与主事件独立）：

```json
{
  "photoUrl": "/data/ftp/box/20260930/10/143025_12_left_1.jpg",
  "type": "4",
  "cleanValue": 2
}
```

---

# 逐字段读取位置表

## 所有 type=1/2/3/4 事件共有字段

| 读取位置 | 类型 | 说明 |
| --- | --- | --- |
| `$.id` | string | 车辆生命周期 ID，同一辆车 type1-6 相同，长度 ≤36 字符 |
| `$.type` | number | 事件类型 1/2/3/4 |
| `$.captureTime` | string | 事件发生时刻，板端实时本机时间 `YYYY-MM-DD HH:MM:SS` |
| `$.captureImage` | string | 事件截图**绝对路径**（按路径字符串接收，不 base64 解码） |
| `$.lane` | string | 车道名称：`冲洗` 或 `绕行` |
| `$.plateNumber` | string | 锁定的车牌号，未能锁定时为空字符串 `""` |
| `$.plateConfidence` | number | 车牌平均置信度 0~1；无车牌时为 `0.0` |
| `$.plateColor` | string | 车牌颜色：`蓝色`/`黄色`/`绿色`/`黄绿`/`""`（黄绿为融合颜色值） |
| `$.plateColorConfidence` | number | 颜色置信度 0~1 |
| `$.vehicleType` | string | 车型：`小汽车`/`蓝色卡车`/`黄色卡车`/`渣土车`/`五小工程车` |
| `$.vehicleTypeConfidence` | number | 车型置信度 0~1 |
| `$.plateIsGuess` | boolean | `true` 表示车牌为推测值（未完全锁定） |
| `$.plateRecognitionAbnormal` | boolean | `true` 表示车牌识别异常（配合 `abnormalReason`） |
| `$.washStartTime` | string | **条件字段**：type3 起携带冲洗确认时间；type1/type2 在异常路径下也可能携带 |
| `$.isAbnormal` | boolean | **条件字段**：仅有异常原因时出现 |
| `$.abnormalReason` | string | **条件字段**：异常原因，多个用 `\|` 拼接 |

## type=5 专有字段

| 读取位置 | 类型 | 说明 |
| --- | --- | --- |
| `$.washEndTime` | string | 冲洗结束时间 |
| `$.videoEndTime` | string | 视频结束时间（闭环发射时刻，按发射时刻近似） |
| `$.totalWashDuration` | number | 总冲洗时长（秒，2位小数）＝检测到任一冲洗目标（人工或清洗台）的并集时长 |
| `$.manualWashDuration` | number | **新增**：人工冲洗时长（秒，2位小数）；与 `cleaningTableWashDuration` 之和**可能大于** `totalWashDuration`（同帧两类并存各计一次） |
| `$.cleaningTableWashDuration` | number | **新增**：清洗台冲洗时长（秒，2位小数） |
| `$.cleanliness` | number | 清洁度（当前固定 `0`，预留字段） |
| `$.videoDuration` | number | 单车录像时长（秒，type1 到 type5 的时间差） |
| `$.direction` | number | 流向码：`5`=正向前出，`6`=正向反出，`7`=反向前出，`8`=反向反出，`0`=未判定 |
| `$.directionLabel` | string | 流向中文标签 |
| `$.wheelResults` | array | 车轮结果列表，仅在该车已锁定车轮结果时出现；未锁定则整个数组缺失 |
| `$.wheelResults[0].side` | string | `left`（左轮）或 `right`（右轮） |
| `$.wheelResults[0].captureTime` | string | 该车轮照片的拍摄时刻 |
| `$.wheelResults[0].photoUrl` | string | 车轮照片**绝对路径**（当前为车轮特写图，无标注原图） |
| `$.wheelResults[0].className` | string | 车轮清洁度区间：`0-25`/`25-50`/`50-75`/`75-100` |
| `$.forceVideoStop` | boolean | **条件字段**：仅超时强制闭环时出现 |
| `$.forcedCompletionReason` | string | **条件字段**：仅超时强制闭环时出现，如 `type2_dwell_timeout_15m` |

## type=6 专有字段

| 读取位置 | 类型 | 说明 |
| --- | --- | --- |
| `$.perIdVideoEnabled` | boolean | 本次是否开启单车录像；`true` 时录像已落盘，路径规则见平台适配文档 |

## 车轮照片批量接口字段

| 读取位置 | 类型 | 说明 |
| --- | --- | --- |
| `$.photoUrl` | string | 车轮照片**绝对路径** |
| `$.type` | string | 侧别：**字符串** `"4"`=左轮，`"5"`=右轮（注意是字符串不是数字） |
| `$.cleanValue` | number | 车轮清洁度：`1`=0-25，`2`=25-50，`3`=50-75，`4`=75-100 |

## 平台对接注意事项

1. **JSON 反序列化必须容忍未知字段**：后续版本可能新增条件字段（如 `forceVideoStop`），不得因未知字段解析失败。
2. **`type` 字段在 wheel-photo 接口里是字符串**（`"4"`/`"5"`），在主事件里是数字（1-6）。
3. **条件字段的缺失语义**：`manualWashDuration`/`cleaningTableWashDuration` 未检测到时是 `0` 而不是缺失；`wheelResults`、`isAbnormal` 等是"不出现即无"。
4. 同一 `id` 的 type1 在平台侧可能在 type2 之后才收到（type1 缓冲到 type2 确认后一起发出，避免未进冲洗区的车误报）。
