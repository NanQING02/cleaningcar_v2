# Wheel Sidechain Test Kit

This kit tests the CleaningCar wheel sidechain without running the main vehicle detector.

## Files

- `wheel_sidechain_probe.py`: board-side probe. It starts wheel RTSP readers, RKNN inference, wheel result cache, binding, photo bucketing, disk write, SQLite queue, and HTTP upload.
- `run_wheel_probe_board.sh`: convenience wrapper for the board.
- `mock_wheel_photo_api.py`: PC-side mock API receiver for `POST /api/vehicle/wheel-photo`.

## PC Side

Run on the PC:

```bash
python tools/mock_wheel_photo_api.py --host 0.0.0.0 --port 28014
```

Use the PC LAN IP in the board command:

```text
http://<PC_IP>:28014/api/vehicle/wheel-photo
```

## Board Side

Copy this kit into the CleaningCar project root on the RK3588 board, then run:

```bash
chmod +x tools/run_wheel_probe_board.sh
./tools/run_wheel_probe_board.sh \
  --config configs/config.json \
  --wheel-photo-url http://<PC_IP>:28014/api/vehicle/wheel-photo \
  --duration 120 \
  --target-fps 10 \
  --force-event-driven false
```

Override RTSP sources when needed:

```bash
./tools/run_wheel_probe_board.sh \
  --config configs/config.json \
  --left-source rtsp://user:pass@left-camera/stream \
  --right-source rtsp://user:pass@right-camera/stream \
  --wheel-photo-url http://<PC_IP>:28014/api/vehicle/wheel-photo \
  --duration 120 \
  --force-event-driven false
```

## Outputs

Board output directory:

```text
wheel_probe_output/YYYYMMDD_HHMMSS/
```

Important files:

- `resolved_settings.json`
- `summary.json`
- `wheel_photo_queue.db`

Wheel photo files are saved under:

```text
<system.wheel_photo_base_dir>/box/YYYYMMDD/HH/
```

PC mock output directory:

```text
mock_api_output/YYYYMMDD_HHMMSS/
```

Important files:

- `wheel_photo_requests.jsonl`
- `requests.jsonl`

## Success Criteria

- Board status shows wheel reader and processor frame counts increasing.
- Board `summary.json` contains wheel photo representatives.
- PC terminal receives `POST /api/vehicle/wheel-photo`.
- PC `wheel_photo_requests.jsonl` contains payloads with `photoUrl`, `type`, and `cleanValue`.
