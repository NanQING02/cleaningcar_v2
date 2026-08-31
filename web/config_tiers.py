from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Tuple


TIER_USER = "user"
TIER_DEVELOPER = "developer"


class ConfigTierError(ValueError):
    """Raised when a tiered config update contains disallowed fields."""


CONFIG_FIELD_REGISTRY: List[Dict[str, str]] = [
    {"path": "system.device_id", "tier": TIER_USER, "group": "system", "label": "设备 ID"},
    {"path": "system.api.url", "tier": TIER_USER, "group": "system", "label": "API 地址"},
    {"path": "system.api.token", "tier": TIER_USER, "group": "system", "label": "API Token"},
    {"path": "system.api.capture_mode", "tier": TIER_USER, "group": "system", "label": "截图上报方式"},
    {"path": "system.api.wheel_photo_url", "tier": TIER_USER, "group": "system", "label": "车轮照片上报地址"},
    {"path": "system.wheel_photo_base_dir", "tier": TIER_USER, "group": "system", "label": "车轮照片根目录"},
    {"path": "video.source", "tier": TIER_USER, "group": "video", "label": "视频源"},
    {"path": "video.source_mode", "tier": TIER_USER, "group": "video", "label": "视频源模式"},
    {"path": "video.hw_decode", "tier": TIER_USER, "group": "video", "label": "硬解"},
    {"path": "video.workers", "tier": TIER_USER, "group": "video", "label": "推理线程数"},
    {"path": "video.core_mask", "tier": TIER_USER, "group": "video", "label": "NPU Core Mask"},
    {"path": "wheel.enabled", "tier": TIER_USER, "group": "wheel", "label": "启用车轮旁路"},
    {"path": "wheel.pause_bypass_during_wash_enabled", "tier": TIER_USER, "group": "wheel", "label": "冲洗道优先暂停绕行道"},
    {"path": "wheel.pause_bypass_config_key", "tier": TIER_USER, "group": "wheel", "label": "绕行配置"},
    {"path": "wheel.pause_bypass_resume_delay_seconds", "tier": TIER_USER, "group": "wheel", "label": "绕行恢复延迟秒数"},
    {"path": "wheel.left_source", "tier": TIER_USER, "group": "wheel", "label": "左车轮视频源"},
    {"path": "wheel.right_source", "tier": TIER_USER, "group": "wheel", "label": "右车轮视频源"},
    {"path": "logic.lane_name", "tier": TIER_USER, "group": "logic", "label": "车道名称"},
    {"path": "logic.enable_per_id_video", "tier": TIER_USER, "group": "logic", "label": "按车 ID 录像"},
    {"path": "logic.per_id_video_dir", "tier": TIER_USER, "group": "logic", "label": "按车 ID 录像目录"},
    {"path": "logic.enable_event_disk", "tier": TIER_USER, "group": "logic", "label": "本地事件 JSON 留存"},
    {"path": "event_capture_dir", "tier": TIER_USER, "group": "output", "label": "事件图片目录"},
    {"path": "logic.per_id_video_source", "tier": TIER_DEVELOPER, "group": "logic", "label": "单车录像帧来源"},
    {"path": "logic.per_id_type6_require_plate_candidate", "tier": TIER_DEVELOPER, "group": "logic", "label": "Type6 要求车牌候选"},
    {"path": "zones.zone_a_detection", "tier": TIER_USER, "group": "zones", "label": "Zone A 检测区"},
    {"path": "zones.zone_b_wash", "tier": TIER_USER, "group": "zones", "label": "Zone B 清洗区"},
    {"path": "zones.flow_vector.start", "tier": TIER_USER, "group": "zones", "label": "流向起点"},
    {"path": "zones.flow_vector.end", "tier": TIER_USER, "group": "zones", "label": "流向终点"},
    {"path": "system.monitor_interval", "tier": TIER_DEVELOPER, "group": "system", "label": "监控间隔"},
    {"path": "system.metrics_path", "tier": TIER_DEVELOPER, "group": "system", "label": "性能指标路径"},
    {"path": "system.command_dir", "tier": TIER_DEVELOPER, "group": "system", "label": "命令目录"},
    {"path": "system.startup_flag_path", "tier": TIER_DEVELOPER, "group": "system", "label": "启动标志路径"},
    {"path": "system.heartbeat_path", "tier": TIER_DEVELOPER, "group": "system", "label": "心跳路径"},
    {"path": "system.heartbeat_interval_seconds", "tier": TIER_DEVELOPER, "group": "system", "label": "心跳间隔"},
    {"path": "system.heartbeat_timeout_seconds", "tier": TIER_DEVELOPER, "group": "system", "label": "心跳超时"},
    {"path": "system.progress_timeout_seconds", "tier": TIER_DEVELOPER, "group": "system", "label": "进度超时"},
    {"path": "system.heartbeat_startup_grace_seconds", "tier": TIER_DEVELOPER, "group": "system", "label": "启动宽限"},
    {"path": "system.startup_capture_dir", "tier": TIER_DEVELOPER, "group": "system", "label": "启动截图目录"},
    {"path": "system.manual_capture_dir", "tier": TIER_DEVELOPER, "group": "system", "label": "手动截图目录"},
    {"path": "system.cpu_mask", "tier": TIER_DEVELOPER, "group": "system", "label": "CPU Mask"},
    {"path": "system.performance_lock_enabled", "tier": TIER_DEVELOPER, "group": "system", "label": "启用定频"},
    {"path": "system.performance_lock_restore_on_stop", "tier": TIER_DEVELOPER, "group": "system", "label": "停止时恢复频率"},
    {"path": "system.performance_lock_script", "tier": TIER_DEVELOPER, "group": "system", "label": "定频脚本路径"},
    {"path": "video.fp_output_mode", "tier": TIER_DEVELOPER, "group": "video", "label": "FP 输出模式"},
    {"path": "video.csv", "tier": TIER_DEVELOPER, "group": "video", "label": "检测 CSV 路径"},
    {"path": "video.debug_frame_path", "tier": TIER_DEVELOPER, "group": "video", "label": "调试帧路径"},
    {"path": "video.debug_frame_interval", "tier": TIER_DEVELOPER, "group": "video", "label": "调试帧间隔"},
    {"path": "video.debug_frame_max_width", "tier": TIER_DEVELOPER, "group": "video", "label": "调试帧最大宽度"},
    {"path": "video.debug_frame_quality", "tier": TIER_DEVELOPER, "group": "video", "label": "调试帧质量"},
    {"path": "video.reader_frame_timeout_seconds", "tier": TIER_DEVELOPER, "group": "video", "label": "读流卡死超时"},
    {"path": "video.segment_minutes", "tier": TIER_DEVELOPER, "group": "video", "label": "切片分钟数"},
    {"path": "video.worker_core_strategy", "tier": TIER_DEVELOPER, "group": "video", "label": "Worker Core 策略"},
    {"path": "wheel.model", "tier": TIER_DEVELOPER, "group": "wheel", "label": "车轮模型路径"},
    {"path": "wheel.classes", "tier": TIER_DEVELOPER, "group": "wheel", "label": "车轮分类标签"},
    {"path": "wheel.event_driven", "tier": TIER_DEVELOPER, "group": "wheel", "label": "事件驱动车轮推理"},
    {"path": "wheel.reader_event_driven", "tier": TIER_DEVELOPER, "group": "wheel", "label": "事件驱动车轮拉流"},
    {"path": "wheel.reader_idle_fps", "tier": TIER_DEVELOPER, "group": "wheel", "label": "车轮空闲拉流 FPS"},
    {"path": "wheel.target_fps", "tier": TIER_DEVELOPER, "group": "wheel", "label": "车轮目标 FPS"},
    {"path": "wheel.active_target_fps", "tier": TIER_DEVELOPER, "group": "wheel", "label": "车轮活动 FPS"},
    {"path": "wheel.core_mask", "tier": TIER_DEVELOPER, "group": "wheel", "label": "车轮 NPU Core Mask"},
    {"path": "wheel.imgsz", "tier": TIER_DEVELOPER, "group": "wheel", "label": "车轮输入尺寸"},
    {"path": "wheel.bind_window_seconds", "tier": TIER_DEVELOPER, "group": "wheel", "label": "车轮绑定窗口秒数"},
    {"path": "wheel.bind_pre_start_seconds", "tier": TIER_DEVELOPER, "group": "wheel", "label": "车轮绑定前置秒数"},
    {"path": "wheel.bind_after_end_seconds", "tier": TIER_DEVELOPER, "group": "wheel", "label": "车轮绑定后置秒数"},
    {"path": "wheel.bind_require_active", "tier": TIER_DEVELOPER, "group": "wheel", "label": "仅活动轨迹绑定车轮"},
    {"path": "wheel.bind_wait_seconds", "tier": TIER_DEVELOPER, "group": "wheel", "label": "Type5 等待车轮秒数"},
    {"path": "wheel.bind_wait_poll_seconds", "tier": TIER_DEVELOPER, "group": "wheel", "label": "Type5 等待轮询秒数"},
    {"path": "wheel.photo_bucket_seconds", "tier": TIER_DEVELOPER, "group": "wheel", "label": "车轮照片分桶秒数"},
    {"path": "wheel.photo_min_score", "tier": TIER_DEVELOPER, "group": "wheel", "label": "车轮照片最低分数"},
    {"path": "logic.detection_anchor", "tier": TIER_DEVELOPER, "group": "logic", "label": "检测锚点"},
    {"path": "logic.zone_a_mask_enable", "tier": TIER_DEVELOPER, "group": "logic", "label": "Zone A Mask"},
    {"path": "logic.zone_b_entry_hysteresis", "tier": TIER_DEVELOPER, "group": "logic", "label": "Zone B 进入迟滞"},
    {"path": "logic.zone_b_exit_hysteresis", "tier": TIER_DEVELOPER, "group": "logic", "label": "Zone B 离开迟滞"},
    {"path": "logic.stationary_min_frames", "tier": TIER_DEVELOPER, "group": "logic", "label": "静止最小帧数"},
    {"path": "logic.stationary_speed_thresh", "tier": TIER_DEVELOPER, "group": "logic", "label": "静止速度阈值"},
    {"path": "logic.type34_min_interval_frames", "tier": TIER_DEVELOPER, "group": "logic", "label": "Type3/4 最小间隔"},
    {"path": "logic.track_timeout_frames", "tier": TIER_DEVELOPER, "group": "logic", "label": "Track 超时"},
    {"path": "logic.track_max_age", "tier": TIER_DEVELOPER, "group": "logic", "label": "Track 最大年龄"},
    {"path": "logic.track_lost_grace_seconds", "tier": TIER_DEVELOPER, "group": "logic", "label": "Track 丢失宽限秒数"},
    {"path": "logic.event_trace_enabled", "tier": TIER_DEVELOPER, "group": "logic", "label": "事件输入追踪"},
    {"path": "logic.event_trace_dir", "tier": TIER_DEVELOPER, "group": "logic", "label": "事件追踪目录"},
    {"path": "logic.event_trace_queue_size", "tier": TIER_DEVELOPER, "group": "logic", "label": "事件追踪队列"},
    {"path": "logic.vehicle_iou_threshold", "tier": TIER_DEVELOPER, "group": "logic", "label": "车辆 IoU 阈值"},
    {"path": "logic.vehicle_center_gate_ratio", "tier": TIER_DEVELOPER, "group": "logic", "label": "车辆中心门控"},
    {"path": "logic.plate_requires_vehicle", "tier": TIER_DEVELOPER, "group": "logic", "label": "车牌识别车辆依赖"},
    {"path": "logic.pending_plate_cache_ttl_frames", "tier": TIER_DEVELOPER, "group": "logic", "label": "待绑定车牌 TTL"},
    {"path": "logic.pending_plate_cache_max_entries", "tier": TIER_DEVELOPER, "group": "logic", "label": "待绑定车牌容量"},
    {"path": "logic.disable_plate_only_events", "tier": TIER_DEVELOPER, "group": "logic", "label": "禁用纯车牌事件"},
    {"path": "logic.single_lifecycle_events", "tier": TIER_DEVELOPER, "group": "logic", "label": "单生命周期事件"},
    {"path": "logic.min_zone_a_dwell_frames_for_type5", "tier": TIER_DEVELOPER, "group": "logic", "label": "Type5 旧版最小停留"},
    {"path": "logic.min_track_frames_for_type1", "tier": TIER_DEVELOPER, "group": "logic", "label": "Type1 最小帧数"},
    {"path": "logic.no_draw", "tier": TIER_DEVELOPER, "group": "logic", "label": "禁用绘制"},
    {"path": "logic.draw_plate_boxes", "tier": TIER_DEVELOPER, "group": "logic", "label": "绘制车牌框"},
    {"path": "logic.require_vehicle_type_for_events", "tier": TIER_DEVELOPER, "group": "logic", "label": "车型锁定后触发 Type3/4"},
    {"path": "logic.vehicle_shrink_ratio", "tier": TIER_DEVELOPER, "group": "logic", "label": "车辆收缩比"},
    {"path": "logic.vehicle_lock_min_votes", "tier": TIER_DEVELOPER, "group": "logic", "label": "车型锁定票数"},
    {"path": "logic.vehicle_lock_on_confirm", "tier": TIER_DEVELOPER, "group": "logic", "label": "确认即锁车型"},
    {"path": "logic.plate_lock_frames", "tier": TIER_DEVELOPER, "group": "logic", "label": "车牌锁定帧数"},
    {"path": "logic.plate_output_shape_log_once", "tier": TIER_DEVELOPER, "group": "logic", "label": "车牌模型 Shape 日志"},
    {"path": "logic.plate_draw_stable_only", "tier": TIER_DEVELOPER, "group": "logic", "label": "仅绘制稳定车牌"},
    {"path": "logic.plate_core_mask", "tier": TIER_DEVELOPER, "group": "logic", "label": "车牌 NPU Core Mask"},
    {"path": "logic.plate_infer_stride", "tier": TIER_DEVELOPER, "group": "logic", "label": "车牌推理间隔"},
    {"path": "logic.default_plate_color", "tier": TIER_DEVELOPER, "group": "logic", "label": "默认车牌颜色"},
    {"path": "logic.default_plate_color_conf", "tier": TIER_DEVELOPER, "group": "logic", "label": "默认颜色置信度"},
    {"path": "logic.default_cleanliness", "tier": TIER_DEVELOPER, "group": "logic", "label": "默认洁净度"},
    {"path": "logic.car_plate_cache_ttl", "tier": TIER_DEVELOPER, "group": "logic", "label": "车牌缓存 TTL"},
    {"path": "logic.allowed_event_types", "tier": TIER_DEVELOPER, "group": "logic", "label": "允许事件类型"},
    {"path": "logic.anchor_offset_ratio", "tier": TIER_DEVELOPER, "group": "logic", "label": "锚点上移比例"},
    {"path": "logic.zone_b_anchor_min_frames", "tier": TIER_DEVELOPER, "group": "logic", "label": "Zone B 锚点延迟"},
    {"path": "logic.debug_overlay", "tier": TIER_DEVELOPER, "group": "logic", "label": "调试覆盖"},
    {"path": "logic.debug_track_state", "tier": TIER_DEVELOPER, "group": "logic", "label": "调试 Track 状态"},
    {"path": "logic.debug_anchor_points", "tier": TIER_DEVELOPER, "group": "logic", "label": "调试锚点"},
    {"path": "logic.debug_water_boxes", "tier": TIER_DEVELOPER, "group": "logic", "label": "调试水雾框"},
    {"path": "logic.wash_duration_offset_seconds", "tier": TIER_DEVELOPER, "group": "logic", "label": "洗车时长偏移"},
    {"path": "logic.min_zone_b_dwell_frames_for_type4", "tier": TIER_DEVELOPER, "group": "logic", "label": "Type4 最小停留"},
    {"path": "logic.shadow_plate_pool.max_candidates", "tier": TIER_DEVELOPER, "group": "logic", "label": "影子车牌池容量"},
    {"path": "logic.shadow_plate_pool.max_age_frames", "tier": TIER_DEVELOPER, "group": "logic", "label": "影子车牌池年龄"},
    {"path": "logic.shadow_plate_pool.text_window_frames", "tier": TIER_DEVELOPER, "group": "logic", "label": "车牌文本投票窗口"},
    {"path": "logic.shadow_plate_pool.text_margin_ratio", "tier": TIER_DEVELOPER, "group": "logic", "label": "车牌文本领先比例"},
    {"path": "logic.shadow_plate_pool.text_switch_min_consecutive", "tier": TIER_DEVELOPER, "group": "logic", "label": "车牌切换连续帧"},
    {"path": "logic.shadow_plate_pool.color_min_confidence", "tier": TIER_DEVELOPER, "group": "logic", "label": "颜色最小置信度"},
    {"path": "logic.shadow_plate_pool.color_lock_frames", "tier": TIER_DEVELOPER, "group": "logic", "label": "颜色锁定帧数"},
    {"path": "logic.event_track_quality.enabled", "tier": TIER_DEVELOPER, "group": "logic", "label": "事件轨迹质量门槛"},
    {"path": "logic.event_track_quality.min_hits_type1", "tier": TIER_DEVELOPER, "group": "logic", "label": "Type1 基础命中"},
    {"path": "logic.event_track_quality.fast_vehicle_min_hits_type1", "tier": TIER_DEVELOPER, "group": "logic", "label": "Type1 快车命中"},
    {"path": "logic.event_track_quality.min_avg_vehicle_conf", "tier": TIER_DEVELOPER, "group": "logic", "label": "车辆平均置信度"},
    {"path": "logic.event_track_quality.fast_vehicle_min_avg_conf", "tier": TIER_DEVELOPER, "group": "logic", "label": "快车平均置信度"},
    {"path": "logic.event_track_quality.plate_candidate_min_hits", "tier": TIER_DEVELOPER, "group": "logic", "label": "车牌候选命中"},
    {"path": "logic.event_track_quality.plate_candidate_can_confirm_type1", "tier": TIER_DEVELOPER, "group": "logic", "label": "车牌候选确认 Type1"},
    {"path": "logic.event_track_quality.min_zone_a_dwell_type5", "tier": TIER_DEVELOPER, "group": "logic", "label": "Type5 ZoneA 最小停留"},
    {"path": "logic.event_track_quality.suppress_obvious_false_type5", "tier": TIER_DEVELOPER, "group": "logic", "label": "抑制可疑 Type5"},
    {"path": "logic.event_track_quality.suspicious_cooldown_seconds", "tier": TIER_DEVELOPER, "group": "logic", "label": "可疑事件冷却秒数"},
    {"path": "event_output_dir", "tier": TIER_DEVELOPER, "group": "output", "label": "事件目录"},
    {"path": "event_capture_quality", "tier": TIER_DEVELOPER, "group": "output", "label": "事件截图质量"},
]


_ROOT_KEYS = {"event_output_dir", "event_capture_dir", "event_capture_quality"}
_TIER_PATHS = {
    TIER_USER: {item["path"] for item in CONFIG_FIELD_REGISTRY if item["tier"] == TIER_USER},
    TIER_DEVELOPER: {item["path"] for item in CONFIG_FIELD_REGISTRY if item["tier"] == TIER_DEVELOPER},
}
_ALL_PATHS = set().union(*_TIER_PATHS.values())


def _set_nested(target: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = target
    for key in parts[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[parts[-1]] = deepcopy(value)


def _get_nested(source: Dict[str, Any], path: str) -> Tuple[bool, Any]:
    cursor: Any = source
    for key in path.split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            return False, None
        cursor = cursor[key]
    return True, deepcopy(cursor)


def _iter_leaf_paths(payload: Any, prefix: Tuple[str, ...] = ()) -> Iterable[str]:
    if isinstance(payload, dict):
        if not payload and prefix:
            yield ".".join(prefix)
            return
        for key, value in payload.items():
            yield from _iter_leaf_paths(value, prefix + (str(key),))
        return
    if prefix:
        yield ".".join(prefix)


def _deep_merge(target: Dict[str, Any], updates: Dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def allowed_paths(tier: str | None = None) -> List[str]:
    if tier == TIER_USER:
        return sorted(_TIER_PATHS[TIER_USER])
    if tier == TIER_DEVELOPER:
        return sorted(_TIER_PATHS[TIER_DEVELOPER])
    return sorted(_ALL_PATHS)


def extract_tier_subset(config: Dict[str, Any], tier: str) -> Dict[str, Any]:
    subset: Dict[str, Any] = {}
    for path in allowed_paths(tier):
        found, value = _get_nested(config, path)
        if found:
            _set_nested(subset, path, value)
    return subset


def validate_tier_payload(payload: Dict[str, Any], tier: str | None = None) -> None:
    if not isinstance(payload, dict):
        raise ConfigTierError("配置 payload 必须是 JSON 对象")
    allowed = set(allowed_paths(tier))
    invalid = sorted(path for path in _iter_leaf_paths(payload) if path and path not in allowed)
    if invalid:
        raise ConfigTierError(f"包含不允许修改的字段: {', '.join(invalid)}")


def merge_tier_payload(config: Dict[str, Any], payload: Dict[str, Any], tier: str | None = None) -> None:
    validate_tier_payload(payload, tier)
    _deep_merge(config, payload)


def schema_payload() -> Dict[str, Any]:
    grouped: Dict[str, Dict[str, List[Dict[str, str]]]] = {
        TIER_USER: {},
        TIER_DEVELOPER: {},
    }
    for item in CONFIG_FIELD_REGISTRY:
        tier_groups = grouped[item["tier"]]
        tier_groups.setdefault(item["group"], []).append(
            {
                "path": item["path"],
                "label": item["label"],
            }
        )
    return {
        "tiers": [TIER_USER, TIER_DEVELOPER],
        "root_keys": sorted(_ROOT_KEYS),
        "groups": grouped,
        "fields": deepcopy(CONFIG_FIELD_REGISTRY),
    }
