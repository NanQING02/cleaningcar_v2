# future_modules/legacy_tracking

This folder stores the legacy vehicle tracking implementation (IoU greedy) for rollback.

## Files

- `tracking_legacy_simple.py`
  - Backup copied from `cleaningcar/tracking.py` on 2026-03-29.
- `__init__.py`
  - Re-exports backup tracker symbols.

## Quick rollback

Set in config:

```json
{
  "logic": {
    "vehicle_tracker_impl": "legacy"
  }
}
```

Switch back to new tracker:

```json
{
  "logic": {
    "vehicle_tracker_impl": "bytetrack"
  }
}
```
