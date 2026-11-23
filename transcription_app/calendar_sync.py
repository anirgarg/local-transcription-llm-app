"""
calendar_sync.py
────────────────
Fetches today's events from macOS Calendar via AppleScript.
No extra Python dependencies — uses subprocess + osascript.

macOS will ask for Calendar access the first time this runs.
"""

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

# Meeting URL patterns — ordered by specificity
_URL_PATTERNS = [
    r'https://[^\s<>"\']*zoom\.us/[^\s<>"\']*',
    r'https://meet\.google\.com/[^\s<>"\']*',
    r'https://teams\.microsoft\.com/[^\s<>"\']*',
    r'https://[^\s<>"\']*webex\.com/[^\s<>"\']*',
    r'https://[^\s<>"\']*gotomeeting\.com/[^\s<>"\']*',
    r'https://[^\s<>"\']*whereby\.com/[^\s<>"\']*',
    r'https://[^\s<>"\']*bluejeans\.com/[^\s<>"\']*',
]

# AppleScript date formats macOS can produce
_DATE_FMTS = [
    "%A, %B %d, %Y at %I:%M:%S %p",
    "%A, %B %d, %Y at %I:%M %p",
    "%A, %B  %d, %Y at %I:%M:%S %p",   # double-space for single-digit days
    "%A, %B  %d, %Y at %I:%M %p",
]

_APPLESCRIPT = r"""
set output to ""
tell application "Calendar"
    set today_start to current date
    set time of today_start to 0
    set today_end to today_start + (24 * 60 * 60)
    repeat with cal in calendars
        try
            set evts to (every event of cal whose start date >= today_start and start date <= today_end)
            repeat with evt in evts
                set evt_title to summary of evt
                set evt_start to (start date of evt) as string
                set evt_end   to (end date of evt)   as string
                try
                    set evt_url to url of evt
                    if evt_url is missing value then set evt_url to ""
                on error
                    set evt_url to ""
                end try
                try
                    set evt_notes to description of evt
                    if evt_notes is missing value then set evt_notes to ""
                on error
                    set evt_notes to ""
                end try
                set output to output & evt_title & "|||" & evt_start & "|||" & evt_end & "|||" & evt_url & "|||" & evt_notes & "\n"
            end repeat
        end try
    end repeat
end tell
return output
"""


@dataclass
class CalendarEvent:
    title: str
    start: datetime
    end:   datetime
    url:   Optional[str] = None
    notes: str = ""

    @property
    def time_str(self) -> str:
        return self.start.strftime("%-I:%M %p")

    @property
    def is_now(self) -> bool:
        now = datetime.now()
        return self.start <= now <= self.end

    @property
    def has_link(self) -> bool:
        return bool(self.url)


def _find_url(text: str) -> Optional[str]:
    for pat in _URL_PATTERNS:
        m = re.search(pat, text or "", re.IGNORECASE)
        if m:
            url = m.group(0).rstrip(".,;)")
            return url
    return None


def _parse_date(s: str) -> Optional[datetime]:
    s = s.strip()
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def get_todays_events() -> list:
    """Return today's CalendarEvent list, sorted by start time. Empty list on failure."""
    try:
        result = subprocess.run(
            ["osascript", "-e", _APPLESCRIPT],
            capture_output=True, text=True, timeout=12,
        )
        events = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|||")
            if len(parts) < 5:
                continue
            title, start_s, end_s, url_s, notes_s = parts[:5]
            start = _parse_date(start_s)
            end   = _parse_date(end_s)
            if not start or not end:
                continue
            url = url_s.strip() or _find_url(notes_s)
            events.append(CalendarEvent(
                title=title.strip(),
                start=start,
                end=end,
                url=url,
                notes=notes_s.strip(),
            ))
        events.sort(key=lambda e: e.start)
        return events
    except Exception:
        return []
