# Facebook/Instagram tracking: key by event type, not date alone

## Problem

On 2026-08-17, a manually-triggered storm capture ran after the daily
sunset video had already posted to Facebook/Instagram. The storm workflow's
attempt to post was silently skipped:

```
INFO:facebook_uploader:Sunset for 2026-08-17 already posted to Facebook (post + reel) and Instagram
```

`FacebookUploader`'s tracking database (`facebook_posts.json`) keys entries
by `date_str` alone. `post_sunset()` checks `db.get(date_str, {})` for
`facebook_id` / `instagram_id` / `facebook_reel_id` and skips posting
entirely if all three are already present — with no awareness that the
existing entry belongs to a *different* video (the sunset) than the one
it's currently trying to post (the storm). Only YouTube uploaded, since
`youtube_uploader.py` has no equivalent tracking gate.

This will recur every time a sunset and a storm post happen on the same
calendar date, in either order, regardless of whether the separate
cancel-sunset-on-storm bug (tracked separately) gets fixed — a fixed
cancellation only changes which video posts first, not whether the second
is blocked.

## Goal

Make Facebook/Instagram posting idempotency per **(date, event type)**,
not per date alone, so a sunset post and a storm post on the same date
track independently.

## Design

Introduce a composite tracking key: `f"{date_str}:{event_type}"`, where
`event_type = metadata.get('event_type', 'sunset')` — the same
absent-means-sunset convention `_generate_vision_caption` already uses
elsewhere in this file (`facebook_uploader.py:278`).

**New helper:**
```python
def _tracking_key(self, date_str: str, event_type: str) -> str:
    return f"{date_str}:{event_type}"
```

**`update_tracking_db`** gains `event_type: str = 'sunset'`. Internally
builds the composite key via `_tracking_key` and stores `date` and
`event_type` as explicit fields on the entry (in addition to using them in
the key), so the raw JSON stays self-describing without decoding the key
format.

**`post_video`** gains `event_type: str = 'sunset'`, threaded straight
through to its `update_tracking_db` call. (Its existing `date_str` param
stays a plain date — used verbatim in the failure-email subject line,
which should stay human-readable — only the tracking-db call changes.)

**`post_sunset`** computes `event_type = metadata.get('event_type',
'sunset')` once, builds the composite key via `_tracking_key`, and uses
that key for the three `already_posted` lookups. Passes `event_type`
through to `post_video` and to both remaining `update_tracking_db` calls
(Reel, Instagram).

**Log message fix (same root cause as the confusing diagnosis):** the
line `f"Sunset for {date_str} already posted to Facebook (post + reel)
and Instagram"` becomes `f"{event_type.title()} post for {date_str}
already fully posted (Facebook + Reel + Instagram)"` — it fired for a
*storm* post while saying "Sunset", which is what made this bug take
longer to diagnose than it should have. Apply the same event_type-aware
wording to the three "already exists, skipping" log lines inside
`post_sunset` (Facebook post / Reel / Instagram).

## Changes needed

**`facebook_uploader.py`** only — confirmed via `grep` that no other file
in the codebase reads `facebook_posts.json` or calls
`load_tracking_db`/`update_tracking_db`/`save_tracking_db`.

- Add `_tracking_key` helper.
- `update_tracking_db`: add `event_type` param, use composite key, store
  `date`/`event_type` fields on the entry.
- `post_video`: add `event_type` param, thread through.
- `post_sunset`: compute `event_type`, use composite key for lookups, pass
  `event_type` through to `post_video` and both `update_tracking_db`
  calls, update the four log messages to name the actual event type.

No other files change.

## Migration

None needed. Existing entries keyed by bare `date_str` (e.g.
`"2026-08-17"`, currently holding the storm's IDs after last night's
manual retry overwrote the sunset's) become inert history under the old
key format — they're simply never looked up again, since all future reads
use the composite key. Dates don't repeat, so there's no double-post risk
from the old entries going stale.

## Testing

Standalone `verify_*.py` script (gitignored, this repo's convention — see
`verify_storm_lightning_arm.py` for the pattern), using a `FacebookUploader`
instance with network calls monkeypatched out (`post_video`,
`post_facebook_reel`, `post_to_instagram` replaced with fakes that record
calls and return a fixed ID) and `tracking_db_path` pointed at a temp file:

- Posting a sunset video for a date, then a storm video for the *same*
  date, both fully complete — sunset's post/reel/instagram all get called
  once, and storm's all get called once (the bug: storm calls would be
  skipped entirely).
- Calling `post_sunset` twice in a row for the *same* date and event type
  (e.g. sunset posted, then daily maintenance retries) — second call is a
  no-op (existing idempotency behavior preserved for the same-event-type
  case).
- `_tracking_key` produces the expected `"date:event_type"` string, and
  omitting `event_type` from metadata defaults to `"sunset"`.
- The persisted entry in the tracking db JSON carries explicit `date` and
  `event_type` fields matching the key.

## Out of scope

- The separate sunset-capture-doesn't-actually-cancel bug (tracked
  separately, not fixed here).
- YouTube's upload path — no equivalent tracking gate exists there, so
  nothing to fix.
- Any migration/cleanup of existing bare-date-keyed entries in
  `facebook_posts.json` — see Migration above.
