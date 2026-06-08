# Storm capture: email notification on start

**Date:** 2026-06-07
**Status:** Approved, pending implementation

## Problem

The system already emails on capture failures, upload successes, and processing
failures, but there is no notification when a storm capture *begins*. The Pi
silently kicks off a 60-minute capture (and aborts any active sunset capture)
when Tempest fires a storm callback. The operator has no awareness until the
post-upload email lands an hour-plus later — which is too late to look outside,
check on equipment, or notice that a forecast/lightning-override decision was
made.

## Goal

Send one plain-text email at the start of every storm-workflow invocation,
including lightning-override re-fires, so the operator knows in real time that
a capture is running and why it was triggered.

## Decisions

- **Trigger point:** top of `StormWorkflow.complete_storm_workflow`, immediately
  after the existing `[STORM] Starting workflow…` log line and before
  `_capture_storm_sequence` is called.
- **Fire policy:** every invocation. The lightning-override path can re-enter
  this method after cooldown is bypassed; we want to know each time.
  `complete_storm_workflow` itself is already gated by `capture_state !=
  'STORM_ACTIVE'` upstream in `on_storm_detected`, so no in-progress dupe is
  possible.
- **Body format:** plain text, not HTML. Matches "minimal noise" intent and
  avoids reusing the failure-email HTML template (different purpose).
- **Failure handling:** SMTP exceptions are caught, logged at WARNING, and
  swallowed. The capture proceeds regardless. The send is fire-and-forget
  from the workflow's perspective.
- **Config:** no new keys. Reuses the existing `email.*` block. When
  `email.enabled` is false, `EmailNotifier.send_notification` already no-ops,
  so the call site doesn't need to guard.

## Email contents

**Subject:**
```
⛈️ Storm capture started — YYYY-MM-DD HH:MM
```
(Local time, naive, derived from `start_time`.)

**Body (plain text):**
```
Storm capture started at <HH:MM:SS local>.
Planned duration: <N> minutes (ends ~<HH:MM> local).
Confidence: <NN>%

Detection reasons:
  • <reason 1>
  • <reason 2>
  ...

Nearest recent lightning: <X.X> km
```

- `Planned duration` is derived from `_compute_storm_duration_seconds() / 60`.
- `Confidence` is `int(round(conditions.confidence * 100))`.
- `Detection reasons` are `conditions.trigger_reasons` (list[str], already
  human-readable, e.g. `"Lightning: 3 strikes within 15km in last 10 min"`).
- The `Nearest recent lightning` line is included only when
  `conditions.lightning_avg_distance` is not `None`. We use the average
  distance field that `_evaluate_storm_conditions` already populates from
  the last 10 minutes of strikes; if we want true "nearest" later, we can
  add a min-distance metric, but this is fine for v1.

## Implementation outline

In `storm_workflow.py`:

```python
def _send_storm_start_notification(
    self,
    conditions,           # StormConditions
    start_time: datetime,
    duration_seconds: int,
) -> None:
    """Fire-and-forget start notification. Never raises."""
    try:
        subject = f"⛈️ Storm capture started — {start_time.strftime('%Y-%m-%d %H:%M')}"
        end_time = start_time + timedelta(seconds=duration_seconds)
        confidence_pct = int(round(conditions.confidence * 100))

        lines = [
            f"Storm capture started at {start_time.strftime('%H:%M:%S')} local.",
            f"Planned duration: {duration_seconds // 60} minutes (ends ~{end_time.strftime('%H:%M')} local).",
            f"Confidence: {confidence_pct}%",
            "",
            "Detection reasons:",
        ]
        for reason in conditions.trigger_reasons or ["(none reported)"]:
            lines.append(f"  • {reason}")

        if conditions.lightning_avg_distance is not None:
            lines.append("")
            lines.append(f"Nearest recent lightning: {conditions.lightning_avg_distance:.1f} km")

        self.email_notifier.send_notification(subject, "\n".join(lines), is_html=False)
    except Exception as e:
        self.logger.warning(f"[STORM] Start notification send failed (non-fatal): {e}")
```

Call site in `complete_storm_workflow`:

```python
target_date = start_time.date()
self.logger.info(f"[STORM] Starting workflow for {target_date} at {start_time.time()}")
self._send_storm_start_notification(
    conditions, start_time, self._compute_storm_duration_seconds()
)
```

## Out of scope

- No end-of-capture / pre-upload notification. `send_upload_success` already
  fires after the standard post-pipeline.
- No HTML body, no inline frame previews. (Future enhancement if the start
  email proves useful but under-detailed.)
- No new config keys; the user already has working SMTP for the existing
  notifications.
- No retry on SMTP failure. The existing `EmailNotifier.send_notification`
  doesn't retry, and a missed start email is not worth blocking capture or
  adding queue/backoff machinery for.
- No tests. The project has no pytest suite and `main.py test` does not
  exercise the storm workflow. Validation is via the next real storm trigger
  or a manual one-off invocation.

## Validation

- `python main.py test --camera` and the existing config validation continue
  to pass.
- On the next storm trigger (or via a one-off `complete_storm_workflow`
  invocation against a stubbed conditions object), the email is received
  within seconds of the `[STORM] Starting workflow…` log line.
- With `email.enabled=false`, no error is raised and the capture proceeds
  normally.
- With SMTP credentials wrong on purpose, the workflow logs the warning and
  proceeds.
