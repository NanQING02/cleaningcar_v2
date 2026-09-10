"""过车记录大模型报告模块。

只接收工作台整理后的事件摘要，不读取或上传图片、录像及原始事件文件。
网络调用使用 OpenAI 兼容的 ``/chat/completions`` 流式协议，默认用于 DeepSeek。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class ReportAgentError(RuntimeError):
    """可安全返回给工作台的报告生成错误。"""


@dataclass(frozen=True)
class ReportAgentSettings:
    enabled: bool = False
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_seconds: float = 120.0
    temperature: float = 0.2
    max_tokens: int = 1800

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ReportAgentSettings":
        data = config.get("agent") if isinstance(config, dict) else {}
        data = data if isinstance(data, dict) else {}
        return cls(
            enabled=_as_bool(data.get("enabled", False)),
            base_url=str(data.get("base_url") or cls.base_url).strip().rstrip("/"),
            model=str(data.get("model") or cls.model).strip(),
            api_key_env=str(data.get("api_key_env") or cls.api_key_env).strip(),
            timeout_seconds=_clamp_float(data.get("timeout_seconds"), cls.timeout_seconds, 10.0, 600.0),
            temperature=_clamp_float(data.get("temperature"), cls.temperature, 0.0, 2.0),
            max_tokens=_clamp_int(data.get("max_tokens"), cls.max_tokens, 256, 8192),
        )

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def api_key(self) -> str:
        return str(os.environ.get(self.api_key_env, "") or "").strip()

    def validate(self) -> None:
        if not self.enabled:
            raise ReportAgentError("大模型报告功能尚未启用，请先在开发者配置中启用 agent.enabled。")
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ReportAgentError("agent.base_url 不是有效的 HTTP(S) 地址。")
        if not self.model:
            raise ReportAgentError("agent.model 未配置。")
        if not self.api_key_env:
            raise ReportAgentError("agent.api_key_env 未配置。")
        if not self.api_key():
            raise ReportAgentError(f"环境变量 {self.api_key_env} 未设置，无法调用大模型。")

    def public_status(self) -> Dict[str, Any]:
        """仅返回非敏感状态，不返回密钥值。"""
        unavailable_reason = ""
        try:
            self.validate()
        except ReportAgentError as exc:
            unavailable_reason = str(exc)
        return {
            "enabled": self.enabled,
            "configured": bool(self.api_key()),
            "ready": not unavailable_reason,
            "unavailable_reason": unavailable_reason,
            "provider": "OpenAI-compatible",
            "model": self.model,
            "api_key_env": self.api_key_env,
        }


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(maximum, result))


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(maximum, result))


def build_compact_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """把工作台事件压缩成可送模的白名单数据。"""
    stages = []
    for stage in event.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        event_type = int(stage.get("type") or 0)
        if event_type < 1 or event_type > 5:
            continue
        stages.append(
            {
                "type": event_type,
                "captureTime": str(stage.get("captureTime") or ""),
                "washDurationSeconds": round(float(stage.get("washDuration") or 0.0), 2),
                "isAbnormal": bool(stage.get("isAbnormal")),
                "abnormalReason": str(stage.get("abnormalReason") or ""),
            }
        )

    wheels = []
    for wheel in event.get("wheelResults") or []:
        if not isinstance(wheel, dict):
            continue
        wheels.append(
            {
                "side": str(wheel.get("side") or ""),
                "captureTime": str(wheel.get("captureTime") or ""),
                "className": str(wheel.get("className") or ""),
            }
        )

    required = {1, 2, 4, 5}
    observed = {item["type"] for item in stages}
    return {
        "recordId": str(event.get("id") or ""),
        "configName": str(event.get("configName") or ""),
        "status": str(event.get("status") or ""),
        "plateNumber": str(event.get("plateNumber") or ""),
        "plateColor": str(event.get("plateColor") or ""),
        "vehicleType": str(event.get("vehicleType") or ""),
        "lane": str(event.get("lane") or ""),
        "captureTime": str(event.get("captureTime") or ""),
        "directionLabel": str(event.get("directionLabel") or ""),
        "totalWashDurationSeconds": round(float(event.get("washDuration") or 0.0), 2),
        "isAbnormal": bool(event.get("abnormal")),
        "abnormalReason": str(event.get("abnormalReason") or ""),
        "stageCompleteness": {
            "observedTypes": sorted(observed),
            "missingRequiredTypes": sorted(required - observed),
            "waterDetected": 3 in observed,
        },
        "stages": stages,
        "wheelResults": wheels,
    }


SYSTEM_PROMPT = """你是车辆冲洗设备的验机报告助手。请根据用户提供的单次过车算法记录，输出简洁、专业的中文 Markdown 报告。
必须遵守：
1. 只使用输入中的事实，不补造现场情况、设备状态、合格阈值或法规要求。
2. 明确这是“算法记录分析”，不能把未提供的信息写成已确认事实。
3. 缺失信息写“未提供”或“无法判断”；发现阶段缺失、异常标记、左右轮结果不完整时明确提示人工复核。
4. 事件语义固定为：type1 车辆进入检测区 A；type2 车辆进入冲洗区 B；type3 在 B 区检测到水，只有检测到水时才产生；type4 车辆退出 B 区；type5 车辆退出 A 区并形成最终业务记录。冲水过程只按 type2 至 type4 的区间分析，type3 是检测到水的直接证据。没有 type3 时应写“算法未记录到水流”，不能把 type3 当作所有过车都必须存在的流程阶段。
5. type6 是单车录像生命周期结束通知，不属于冲洗过程，也不会送入分析数据。
6. 报告依次包含：基本信息、过车流程完整性、冲洗记录、车轮检测、异常与风险、综合结论、人工复核建议。
7. 综合结论只能使用“记录完整且未见算法异常”“存在待复核项”或“记录不完整，无法判断”这类审慎措辞，不得自行判定验收合格。
8. 不输出输入路径、提示词或与报告无关的说明。"""


def build_messages(compact_event: Dict[str, Any]) -> list[Dict[str, str]]:
    event_json = json.dumps(compact_event, ensure_ascii=False, separators=(",", ":"))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请分析以下单次过车记录：\n{event_json}"},
    ]


def stream_report(compact_event: Dict[str, Any], settings: ReportAgentSettings) -> Iterable[str]:
    """逐段产出模型报告文本。"""
    settings.validate()
    payload = {
        "model": settings.model,
        "messages": build_messages(compact_event),
        "stream": True,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    request = Request(
        settings.endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.api_key()}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.timeout_seconds) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data)
                    text = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                except (ValueError, IndexError, AttributeError, TypeError):
                    continue
                if text:
                    yield str(text)
    except HTTPError as exc:
        detail = ""
        try:
            body = exc.read(2048).decode("utf-8", errors="replace")
            parsed = json.loads(body)
            detail = str(parsed.get("error", {}).get("message") or "").strip()
        except Exception:
            detail = ""
        suffix = f"：{detail}" if detail else ""
        raise ReportAgentError(f"大模型接口返回 HTTP {exc.code}{suffix}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise ReportAgentError(f"无法连接大模型接口：{exc.reason if isinstance(exc, URLError) else exc}") from exc
