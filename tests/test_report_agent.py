import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cleaningcar import report_agent
from web.config_tiers import ConfigTierError, TIER_DEVELOPER, TIER_USER, merge_tier_payload
from web import workbench


def _event():
    return {
        "id": "RK3588-DEV-20260909-001",
        "configName": "config.json",
        "status": "已结束",
        "plateNumber": "鲁A12345",
        "plateColor": "蓝色",
        "vehicleType": "货车",
        "lane": "冲洗道",
        "captureTime": "2026-09-09 10:00:30",
        "washDuration": 12.345,
        "abnormal": False,
        "sourceFiles": ["events/config/private.json"],
        "captureImage": "/data/ftp/private.jpg",
        "videoPath": "/data/ftp/private.mp4",
        "stages": [
            {
                "type": 1,
                "captureTime": "2026-09-09 10:00:00",
                "captureImage": "/data/ftp/type1.jpg",
                "sourceFile": "events/config/type1.json",
            },
            {
                "type": 5,
                "captureTime": "2026-09-09 10:00:30",
                "washDuration": 12.345,
                "isAbnormal": True,
                "abnormalReason": "WHEEL_MISSING",
            },
            {
                "type": 6,
                "captureTime": "2026-09-09 10:00:35",
                "perIdVideoEnabled": True,
            },
        ],
        "wheelResults": [
            {
                "side": "left",
                "captureTime": "2026-09-09 10:00:20",
                "className": "clean",
                "photoUrl": "/data/ftp/wheel.jpg",
                "score": 0.93,
            }
        ],
    }


def test_build_compact_event_only_keeps_report_evidence():
    compact = report_agent.build_compact_event(_event())
    serialized = json.dumps(compact, ensure_ascii=False)

    assert compact["recordId"] == "RK3588-DEV-20260909-001"
    assert compact["totalWashDurationSeconds"] == 12.35
    assert compact["stageCompleteness"] == {
        "observedTypes": [1, 5],
        "missingRequiredTypes": [2, 4],
        "waterDetected": False,
    }
    assert compact["wheelResults"] == [
        {"side": "left", "captureTime": "2026-09-09 10:00:20", "className": "clean"}
    ]
    assert all(stage["type"] <= 5 for stage in compact["stages"])
    for private_value in ("private.json", "private.jpg", "private.mp4", "wheel.jpg"):
        assert private_value not in serialized


def test_settings_require_enabled_and_environment_key(monkeypatch):
    settings = report_agent.ReportAgentSettings.from_config({"agent": {"enabled": True}})
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(report_agent.ReportAgentError, match="DEEPSEEK_API_KEY"):
        settings.validate()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    settings.validate()
    status = settings.public_status()
    assert status["configured"] is True
    assert status["ready"] is True
    assert "secret" not in json.dumps(status)


def test_agent_settings_are_developer_only():
    payload = {"agent": {"enabled": True, "model": "deepseek-v4-flash"}}
    target = {}
    merge_tier_payload(target, payload, TIER_DEVELOPER)
    assert target == payload

    with pytest.raises(ConfigTierError):
        merge_tier_payload({}, payload, TIER_USER)


def test_stream_report_parses_openai_compatible_sse(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"# Report"}}]}\n'
            yield b'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}\n'
            yield b'data: {"choices":[{"delta":{"content":" done"}}]}\n'
            yield b"data: [DONE]\n"

    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setattr(report_agent, "urlopen", fake_urlopen)
    settings = report_agent.ReportAgentSettings.from_config({"agent": {"enabled": True}})

    assert "".join(report_agent.stream_report({"recordId": "1"}, settings)) == "# Report done"
    assert captured["request"].full_url == "https://api.deepseek.com/chat/completions"
    body = json.loads(captured["request"].data.decode("utf-8"))
    assert body["stream"] is True
    assert body["model"] == "deepseek-v4-flash"
    assert "secret" not in captured["request"].data.decode("utf-8")


def test_workbench_report_endpoint_streams_tokens(monkeypatch, tmp_path):
    event = _event()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setattr(workbench, "_find_event", lambda _event_id, _key: event)
    monkeypatch.setattr(
        workbench,
        "_event_config",
        lambda _row, _key: SimpleNamespace(
            data={"agent": {"enabled": True}, "event_output_dir": str(tmp_path)},
            path=tmp_path / "config.json",
        ),
    )
    monkeypatch.setattr(workbench, "stream_report", lambda _compact, _settings: iter(["第一段", "第二段"]))
    app = FastAPI()
    app.include_router(workbench.router)

    response = TestClient(app).post(f"/workbench/events/{event['id']}/agent/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: meta" in response.text
    assert '"text":"第一段"' in response.text
    assert '"text":"第二段"' in response.text
    assert "event: done" in response.text
    cached_files = list((tmp_path / "agent_reports").glob("*.json"))
    assert len(cached_files) == 1
    assert json.loads(cached_files[0].read_text(encoding="utf-8"))["report"] == "第一段第二段"

    cached_response = TestClient(app).get(f"/workbench/events/{event['id']}/agent/report")
    assert cached_response.status_code == 200
    assert cached_response.json()["report"] == "第一段第二段"


def test_workbench_report_endpoint_rejects_incomplete_record(monkeypatch):
    event = {**_event(), "status": "进行中"}
    monkeypatch.setattr(workbench, "_find_event", lambda _event_id, _key: event)
    app = FastAPI()
    app.include_router(workbench.router)

    response = TestClient(app).post(f"/workbench/events/{event['id']}/agent/report")

    assert response.status_code == 409
    assert "尚未结束" in response.json()["detail"]


def test_workbench_uses_type5_business_result_instead_of_type6(monkeypatch, tmp_path):
    event_id = "RK3588-DEV-001"
    common = {
        "id": event_id,
        "trackId": 1,
        "captureTime": "2026-09-09 10:00:00",
        "plateNumber": "鲁A12345",
        "vehicleType": "car",
        "lane": "冲洗",
    }
    (tmp_path / "t5.json").write_text(
        json.dumps(
            {
                **common,
                "type": 5,
                "direction": 8,
                "directionLabel": "反向反出",
                "totalWashDuration": 12.5,
                "wheelResults": [{"side": "left", "className": "75-100"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "t6.json").write_text(
        json.dumps({**common, "type": 6, "captureTime": "2026-09-09 10:00:02", "perIdVideoEnabled": True}),
        encoding="utf-8",
    )
    monkeypatch.setattr(workbench, "_event_dirs", lambda _key: [tmp_path])
    monkeypatch.setattr(
        workbench,
        "_load_config",
        lambda _key=None: SimpleNamespace(path=tmp_path / "config.json", data={"system": {"device_id": "RK3588-DEV"}}),
    )

    row = workbench._read_events(None)[0]

    assert row["status"] == "已结束"
    assert row["directionLabel"] == "反向反出"
    assert row["washDuration"] == 12.5
    assert row["wheelResults"] == [{"side": "left", "className": "75-100"}]
    assert row["captureTime"] == "2026-09-09 10:00:00"
    assert row["process"]["videoCompleteTime"] == "2026-09-09 10:00:02"
    assert {stage["type"]: stage["label"] for stage in row["stages"]}[6] == "单车录像结束"
    cfg = SimpleNamespace(data={"event_output_dir": str(tmp_path)}, path=tmp_path / "config.json")
    monkeypatch.setattr(workbench, "_load_config", lambda _key=None: cfg)
    monkeypatch.setattr(workbench, "_event_dirs", lambda _key: [tmp_path])

    row = workbench._read_events(None)[0]

    assert row["status"] == "已结束"
    assert row["latestType"] == 6
    assert row["latestBusinessType"] == 5
    assert row["captureTime"] == "2026-09-09 10:00:00"
    assert row["directionLabel"] == "反向反出"
    assert row["washDuration"] == 12.5
    assert row["wheelResults"][0]["side"] == "left"
    assert row["process"]["videoCompleteTime"] == "2026-09-09 10:00:02"


def test_cached_report_round_trip(monkeypatch, tmp_path):
    cfg = SimpleNamespace(data={"event_output_dir": str(tmp_path)})
    workbench._write_cached_report(cfg, "record/unsafe", "报告正文", "model-a")

    cached = workbench._read_cached_report(cfg, "record/unsafe")

    assert cached["available"] is True
    assert cached["report"] == "报告正文"
    assert cached["model"] == "model-a"
    assert len(list((tmp_path / "agent_reports").glob("*.json"))) == 1
