"""
main.py  v2.0
─────────────
Local Meeting Transcription App
macOS / Apple Silicon

Single-window design:
  • Left sidebar  — meeting list + audio settings (no pop-ups)
  • Right panel   — transcript + summary (context-switches between
                    "new meeting" view and "past meeting" view)
"""

import re
import queue
import threading
import time
import datetime
from pathlib import Path
from typing import Optional

import customtkinter as ctk

from audio_capture import AudioCapturer, AudioSource, get_blackhole_index
from transcriber import Transcriber, TranscriptSegment
from summarizer import Summarizer
from diarizer import Diarizer        # pyannote loaded lazily inside Diarizer.load()
from calendar_sync import get_todays_events, CalendarEvent


# ── App-level constants ───────────────────────────────────────────────────────
APP_NAME    = "Transcribe"
APP_VERSION = "2.0"
WINDOW_W, WINDOW_H = 1120, 720
SIDEBAR_W           = 268
SAVES_DIR           = Path.home() / "Documents" / "Transcriptions"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "bg":         "#0f172a",   # window background
    "sidebar":    "#0d1424",   # sidebar background
    "panel":      "#1a2236",   # content panel background
    "card":       "#1e2a3a",   # sidebar row / settings card
    "card_h":     "#27364a",   # sidebar row hover
    "accent":     "#3b82f6",   # blue accent (buttons, selection)
    "accent_lt":  "#60a5fa",   # lighter blue (hover)
    "success":    "#22c55e",   # green (record-start button)
    "success_dk": "#16a34a",   # green hover
    "rec":        "#ef4444",   # recording red
    "rec_dk":     "#dc2626",   # recording hover
    "warning":    "#f59e0b",   # amber warning
    "text":       "#f1f5f9",   # primary text
    "text_dim":   "#475569",   # dim labels / captions
    "text_mid":   "#94a3b8",   # secondary text
    "border":     "#1e2d40",   # subtle borders
    "sel":        "#1d3461",   # selected meeting row fill
    "sel_bdr":    "#3b82f6",   # selected meeting row border
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _file_sort_key(path: Path) -> str:
    """
    Sort key that works for both old 'transcript_YYYY-MM-DD_HH-MM.md'
    and new 'meeting_YYYY-MM-DD_HH-MM-SS.md' filenames by extracting
    the date/time portion so newest always sorts first (reverse=True).
    """
    try:
        parts = path.stem.split("_", 1)   # split on first _ only
        return parts[1] if len(parts) > 1 else path.stem
    except Exception:
        return path.stem

def _date_group(path: Path) -> str:
    """Bucket a transcript file into a human-readable date group."""
    try:
        date_part = path.stem.split("_")[1]          # meeting_YYYY-MM-DD_…
        file_date = datetime.date.fromisoformat(date_part)
    except Exception:
        return "Older"
    today = datetime.date.today()
    delta = (today - file_date).days
    if delta == 0:   return "Today"
    if delta == 1:   return "Yesterday"
    if delta <= 6:   return "This Week"
    # Use full month name for anything older — groups naturally by month
    return file_date.strftime("%B %Y")   # e.g. "May 2026"


def _parse_file(path: Path) -> dict:
    """Read a transcript .md file and return display metadata."""
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"path": path, "time": "—", "synopsis": "",
                "group": "Older", "summary_text": "", "transcript_text": ""}

    time_str  = ""
    date_str  = ""
    dur_str   = ""
    title_str = ""
    synopsis_lines, summary_lines, transcript_lines = [], [], []
    in_summary = in_transcript = False

    for line in raw.splitlines():
        if "**Title:**" in line:
            title_str = line.replace("**Title:**", "").strip()
        if "**Date:**" in line:
            raw_dt = line.replace("**Date:**", "").strip()
            try:
                dt = datetime.datetime.strptime(raw_dt, "%B %d, %Y %H:%M")
                time_str = dt.strftime("%-I:%M %p")
                date_str = dt.strftime("%b %-d")
            except Exception:
                time_str = raw_dt[-5:] if len(raw_dt) >= 5 else raw_dt
        if "**Duration:**" in line:
            dur_str = line.replace("**Duration:**", "").strip()
        if line.strip() == "## Summary":
            in_summary, in_transcript = True, False;  continue
        elif line.strip() == "## Full Transcript":
            in_summary, in_transcript = False, True;  continue
        elif line.strip() == "---":
            in_summary = False;  continue
        if in_summary:
            summary_lines.append(line)
            # Skip markdown headers (### Executive Summary etc.) for the sidebar synopsis
            if line.strip().startswith("#"):
                continue
            stripped = line.strip().lstrip("*> -")
            if stripped and len(synopsis_lines) < 2:
                synopsis_lines.append(stripped)
        if in_transcript:
            transcript_lines.append(line)

    # Fallback time/date from filename (meeting_2024-01-15_14-30-00.md)
    if not time_str:
        try:
            parts = path.stem.split("_")[2].split("-")
            h, m  = int(parts[0]), int(parts[1])
            time_str = datetime.time(h, m).strftime("%-I:%M %p")
        except Exception:
            time_str = "—"
    if not date_str:
        try:
            date_part = path.stem.split("_")[1]
            date_str = datetime.date.fromisoformat(date_part).strftime("%b %-d")
        except Exception:
            date_str = ""

    parts = [p for p in [date_str, time_str] if p]
    datetime_str = "  ·  ".join(parts) if parts else "—"
    if dur_str:
        datetime_str += f"  ·  {dur_str}"

    synopsis = " ".join(synopsis_lines)
    if len(synopsis) > 110:
        synopsis = synopsis[:110] + "…"

    return {
        "path":            path,
        "title":           title_str,
        "time":            datetime_str,
        "synopsis":        synopsis or "No summary — tap to view",
        "group":           _date_group(path),
        "summary_text":    "\n".join(summary_lines).strip(),
        "transcript_text": "\n".join(transcript_lines).strip(),
    }


_FILE_CACHE: dict = {}   # module-level cache: path → (mtime, info)

def _parse_file_cached(path: Path) -> dict:
    """Return cached parse result, refreshing only if the file has changed."""
    try:
        mtime = path.stat().st_mtime
    except Exception:
        return _parse_file(path)
    entry = _FILE_CACHE.get(path)
    if entry and entry[0] == mtime:
        return entry[1]
    info = _parse_file(path)
    _FILE_CACHE[path] = (mtime, info)
    return info

def _invalidate_cache(path: Path):
    _FILE_CACHE.pop(path, None)


# ── Speaker Name Dialog ───────────────────────────────────────────────────────

class SpeakerNameDialog(ctk.CTkToplevel):
    """
    Modal dialog shown after diarization.
    Shows a sample quote per speaker so the user can identify and name them.
    """

    def __init__(self, parent, speakers: list):
        # speakers: [(display_label, sample_quote), …]
        super().__init__(parent)
        self.title("Name the Speakers")
        self.resizable(False, False)
        self.grab_set()
        self.configure(fg_color=C["bg"])

        self._entries: dict[str, ctk.CTkEntry] = {}
        self._result:  dict[str, str]          = {}

        ctk.CTkLabel(
            self,
            text="Who was in the meeting?\nEdit each name, then click Apply.",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=C["text"], justify="center",
        ).pack(padx=28, pady=(22, 14))

        for label, sample in speakers:
            card = ctk.CTkFrame(self, fg_color=C["card"], corner_radius=8)
            card.pack(padx=20, pady=5, fill="x")
            card.grid_columnconfigure(0, weight=1)

            if sample:
                ctk.CTkLabel(
                    card,
                    text=f'"{sample}"',
                    font=ctk.CTkFont(size=11, slant="italic"),
                    text_color=C["text_mid"],
                    wraplength=360, justify="left", anchor="w",
                ).grid(row=0, column=0, padx=14, pady=(10, 4), sticky="w")

            entry = ctk.CTkEntry(
                card,
                font=ctk.CTkFont(size=13, weight="bold"),
                fg_color=C["card_h"], border_color=C["accent"],
                text_color=C["text"], height=36,
            )
            entry.insert(0, label)
            entry.select_range(0, "end")
            entry.grid(row=1, column=0, padx=14, pady=(0, 12), sticky="ew")
            self._entries[label] = entry

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(padx=20, pady=(6, 22), fill="x")

        ctk.CTkButton(
            btn_row, text="Cancel", width=88,
            fg_color=C["card"], hover_color=C["card_h"], text_color=C["text_mid"],
            command=self._cancel,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            btn_row, text="✓  Apply Names", width=130,
            fg_color=C["accent"], hover_color=C["accent_lt"],
            command=self._confirm,
        ).pack(side="right")

        self.bind("<Return>", lambda e: self._confirm())
        self.bind("<Escape>", lambda e: self._cancel())

        # Centre over parent
        self.update_idletasks()
        px, py = parent.winfo_x(), parent.winfo_y()
        pw, ph = parent.winfo_width(), parent.winfo_height()
        w, h   = self.winfo_reqwidth(), self.winfo_reqheight()
        self.geometry(f"{w}x{h}+{px+(pw-w)//2}+{py+(ph-h)//2}")

    def _confirm(self):
        for label, entry in self._entries.items():
            name = entry.get().strip()
            self._result[label] = name if name else label
        self.destroy()

    def _cancel(self):
        self._result = {}
        self.destroy()

    def get_result(self) -> dict:
        """Returns {original_label: user_name} or {} if cancelled."""
        return self._result


# ── Main App ──────────────────────────────────────────────────────────────────

class TranscriptionApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.minsize(860, 560)
        self.resizable(True, True)
        self.configure(fg_color=C["bg"])

        # ── Core state ──────────────────────────────────────────────────
        self._recording:           bool               = False
        self._paused:              bool               = False
        self._session_duration:    int                = 0   # seconds, stored on stop
        self._transcript_segments: list[TranscriptSegment] = []
        self._ui_queue:            queue.Queue        = queue.Queue()
        self._capturer:            Optional[AudioCapturer] = None
        self._transcriber:         Optional[Transcriber]   = None
        self._summarizer:          Optional[Summarizer]    = None
        self._session_start:       Optional[float]    = None
        self._elapsed_timer_id                        = None
        self._level_value:         float              = 0.0
        self._blackhole_available: bool               = get_blackhole_index() is not None
        self._search_term:         str                = ""

        # ── View state ──────────────────────────────────────────────────
        self._current_view:   str           = "new"   # "new" | "past"
        self._viewing_path:   Optional[Path] = None
        self._viewing_xscript: str           = ""     # raw text for AI
        self._meeting_rows:   dict           = {}     # path → CTkFrame
        self._live_summary:   str           = ""     # summary from current live session
        self._current_save_path: Optional[Path] = None  # path of the auto-saved file
        self._file_cache:        dict           = {}    # path → (mtime, info)
        self._diarizer:          Diarizer       = Diarizer()
        self._audio_chunks:      list           = []    # raw audio for diarization
        self._speaker_map:       dict           = {}    # segment_idx → speaker label
        self._calendar_events:   list           = []    # today's CalendarEvent list
        self._pending_title:     str            = ""    # title pre-filled from calendar

        SAVES_DIR.mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self._bind_shortcuts()
        self._poll_ui_queue()
        self._init_backend()

    # ═══════════════════════════════════════════════════════════════════════════
    #  UI Construction
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()
        self._select_new_meeting()  # sets correct initial topbar layout

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=SIDEBAR_W, fg_color=C["sidebar"],
                          corner_radius=0)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_propagate(False)
        # row 4 = meetings list (weight=1), everything else fixed height
        sb.grid_rowconfigure(4, weight=1)
        sb.grid_columnconfigure(0, weight=1)

        # Logo row
        logo = ctk.CTkFrame(sb, fg_color="transparent")
        logo.grid(row=0, column=0, sticky="ew", padx=16, pady=(18, 10))
        logo.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(logo, text="🎙", font=ctk.CTkFont(size=22)
                     ).grid(row=0, column=0, padx=(0, 8))
        ctk.CTkLabel(logo, text=APP_NAME,
                     font=ctk.CTkFont(size=17, weight="bold"),
                     text_color=C["text"]).grid(row=0, column=1, sticky="w")

        # "＋ New Meeting" button
        ctk.CTkButton(
            sb, text="＋  New Meeting",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=C["accent"], hover_color=C["accent_lt"],
            text_color="white", height=36, corner_radius=8,
            command=self._select_new_meeting,
        ).grid(row=1, column=0, padx=12, pady=(0, 6), sticky="ew")

        # ── Search box  (directly above meetings)
        search_frame = ctk.CTkFrame(sb, fg_color=C["card"], corner_radius=8)
        search_frame.grid(row=2, column=0, padx=12, pady=(0, 4), sticky="ew")
        search_frame.grid_columnconfigure(0, weight=1)
        self._search_entry = ctk.CTkEntry(
            search_frame,
            placeholder_text="🔍  Search meetings…",
            font=ctk.CTkFont(size=12),
            fg_color="transparent", border_width=0,
            text_color=C["text"], height=30,
        )
        self._search_entry.grid(row=0, column=0, padx=8, pady=3, sticky="ew")
        self._search_entry.bind("<KeyRelease>", self._on_search_change)

        # ── MEETINGS header + month jump strip
        mhdr = ctk.CTkFrame(sb, fg_color="transparent")
        mhdr.grid(row=3, column=0, sticky="ew", padx=14, pady=(2, 0))
        mhdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(mhdr, text="MEETINGS",
                     font=ctk.CTkFont(size=9, weight="bold"),
                     text_color=C["text_dim"]).grid(row=0, column=0, sticky="w")

        self._month_nav = ctk.CTkFrame(sb, fg_color="transparent")
        self._month_nav.grid(row=3, column=0, sticky="e", padx=8, pady=(2, 0))
        self._group_headers: dict = {}

        # ── Scrollable meetings list
        self._meetings_scroll = ctk.CTkScrollableFrame(
            sb, fg_color="transparent",
            scrollbar_button_color=C["border"],
            scrollbar_button_hover_color=C["card_h"],
        )
        self._meetings_scroll.grid(row=4, column=0, sticky="nsew", padx=4, pady=(2, 4))
        self._meetings_scroll.grid_columnconfigure(0, weight=1)
        # Enable trackpad / mouse-wheel scrolling
        self._setup_meetings_scroll()

        # ── Settings card (below meetings list)
        sc = ctk.CTkFrame(sb, fg_color=C["card"], corner_radius=8)
        sc.grid(row=5, column=0, padx=12, pady=(0, 4), sticky="ew")
        sc.grid_columnconfigure(0, weight=1)

        # Audio source
        src_values = ["Mic", "System", "Both"] if self._blackhole_available else ["Mic"]
        self._source_var = ctk.StringVar(value="Mic")
        self._source_seg = ctk.CTkSegmentedButton(
            sc, values=src_values, variable=self._source_var,
            font=ctk.CTkFont(size=12),
            fg_color=C["card"],
            selected_color=C["accent"], selected_hover_color=C["accent_lt"],
            unselected_color=C["card"], unselected_hover_color=C["card_h"],
            text_color=C["text_mid"],
            height=28, command=self._on_source_change,
        )
        self._source_seg.grid(row=0, column=0, padx=8, pady=(8, 4), sticky="ew")

        if not self._blackhole_available:
            ctk.CTkLabel(sc, text="⚠ Install BlackHole for system audio",
                         font=ctk.CTkFont(size=9), text_color=C["warning"],
                         anchor="w").grid(row=1, column=0, padx=10, pady=(0, 2), sticky="w")

        # Language
        self._lang_var = ctk.StringVar(value="Auto-detect")
        ctk.CTkOptionMenu(
            sc, values=["Auto-detect","English","Spanish","French",
                        "German","Japanese","Chinese","Hindi"],
            variable=self._lang_var,
            fg_color=C["card_h"], button_color=C["accent"],
            button_hover_color=C["accent_lt"], text_color=C["text"],
            font=ctk.CTkFont(size=12), height=28,
        ).grid(row=2, column=0, padx=8, pady=(0, 6), sticky="ew")

        # HF token
        existing_tok = Diarizer.get_hf_token() or ""
        self._hf_token_var = ctk.StringVar(value=existing_tok)
        hf_row = ctk.CTkFrame(sc, fg_color="transparent")
        hf_row.grid(row=3, column=0, padx=8, pady=(0, 8), sticky="ew")
        hf_row.grid_columnconfigure(0, weight=1)

        hf_entry = ctk.CTkEntry(
            hf_row, textvariable=self._hf_token_var,
            placeholder_text="HuggingFace token (diarization)…",
            show="•" if existing_tok else "",
            font=ctk.CTkFont(size=11),
            fg_color=C["card_h"], height=26,
        )
        hf_entry.grid(row=0, column=0, sticky="ew")
        hf_entry.bind("<Return>",   lambda e: self._save_hf_token())
        hf_entry.bind("<FocusOut>", lambda e: self._save_hf_token())

        if existing_tok:
            ctk.CTkLabel(hf_row, text="✓ Diarization ready",
                         font=ctk.CTkFont(size=9), text_color=C["success"],
                         ).grid(row=0, column=1, padx=(6, 0))

        # ── Footer (mic level + backend status)
        footer = ctk.CTkFrame(sb, fg_color=C["card"], corner_radius=8)
        footer.grid(row=6, column=0, padx=12, pady=(0, 12), sticky="ew")
        footer.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(footer, text="🔊",
                     font=ctk.CTkFont(size=10)).grid(
            row=0, column=0, padx=(10, 4), pady=(8, 4))

        self._level_bar = ctk.CTkProgressBar(
            footer, height=6,
            fg_color=C["border"], progress_color=C["success"])
        self._level_bar.set(0)
        self._level_bar.grid(row=0, column=1, padx=(0, 10), pady=(8, 4), sticky="ew")

        self._backend_lbl = ctk.CTkLabel(
            footer, text="Loading…",
            font=ctk.CTkFont(size=9), text_color=C["text_dim"], anchor="w")
        self._backend_lbl.grid(row=1, column=0, columnspan=2,
                                padx=10, pady=(0, 8), sticky="w")

        self._refresh_meetings_list()

    # ── Main panel ────────────────────────────────────────────────────────────

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(2, weight=3)   # transcript
        main.grid_rowconfigure(4, weight=2)   # summary
        main.grid_columnconfigure(0, weight=1)
        self._main = main

        # ── Top bar
        topbar = ctk.CTkFrame(main, fg_color=C["panel"], height=62,
                               corner_radius=0)
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.grid_columnconfigure(2, weight=1)
        topbar.grid_propagate(False)
        self._topbar = topbar

        # Recording button (new-meeting mode)
        self._rec_btn = ctk.CTkButton(
            topbar, text="⏺  Start Recording",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=C["success"], hover_color=C["success_dk"],
            text_color="white", width=172, height=38, corner_radius=19,
            command=self._toggle_recording,
        )
        self._rec_btn.grid(row=0, column=0, padx=16, pady=12)

        # Timer (new-meeting mode — shown at col 2 only while recording/in new view)
        self._timer_lbl = ctk.CTkLabel(
            topbar, text="00:00",
            font=ctk.CTkFont(size=22, weight="bold", family="Menlo"),
            text_color=C["text_dim"])
        # not gridded initially — placed by _select_new_meeting

        # Context title (past-meeting mode) — hidden until needed
        self._ctx_title = ctk.CTkLabel(
            topbar, text="",
            font=ctk.CTkFont(size=12),
            text_color=C["text_mid"], anchor="w")
        # not gridded initially

        # Editable meeting title (past-meeting mode)
        self._title_var = ctk.StringVar()
        self._title_entry = ctk.CTkEntry(
            topbar,
            textvariable=self._title_var,
            placeholder_text="Meeting title…",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="transparent", border_width=0,
            text_color=C["text"], height=36,
        )
        self._title_entry.bind("<Return>",   lambda e: self._save_title())
        self._title_entry.bind("<FocusOut>", lambda e: self._save_title())
        # not gridded initially

        # Title entry for new-meeting mode (pre-fill from calendar)
        self._new_title_var = ctk.StringVar()
        self._new_title_entry = ctk.CTkEntry(
            topbar,
            textvariable=self._new_title_var,
            placeholder_text="Meeting title (optional)…",
            font=ctk.CTkFont(size=13),
            fg_color=C["card"], border_color=C["border"],
            text_color=C["text"], height=32, width=220,
        )
        self._new_title_entry.grid(row=0, column=1, padx=(0, 8), pady=15, sticky="w")

        # Action buttons frame (always visible at column 3)
        af = ctk.CTkFrame(topbar, fg_color="transparent")
        af.grid(row=0, column=3, padx=16, pady=12)
        self._action_frame = af

        self._pause_btn = ctk.CTkButton(
            af, text="⏸  Pause",
            font=ctk.CTkFont(size=12), width=90, height=36,
            fg_color=C["card"], hover_color=C["warning"],
            command=self._toggle_pause,
        )
        self._save_btn = ctk.CTkButton(
            af, text="💾  Save",
            font=ctk.CTkFont(size=12), width=86, height=36,
            fg_color=C["card"], hover_color=C["accent"],
            command=self._save_transcript,
        )
        self._summarize_btn = ctk.CTkButton(
            af, text="✨  Summarize",
            font=ctk.CTkFont(size=12), width=112, height=36,
            fg_color=C["card"], hover_color=C["accent"],
            command=self._run_summary,
        )
        self._export_btn = ctk.CTkButton(
            af, text="📤  Export",
            font=ctk.CTkFont(size=12), width=92, height=36,
            fg_color=C["card"], hover_color=C["accent"],
            command=self._export_meeting,
        )
        self._clear_btn = ctk.CTkButton(
            af, text="🗑  Clear",
            font=ctk.CTkFont(size=12), width=78, height=36,
            fg_color=C["card"], hover_color=C["rec"],
            command=self._clear_transcript,
        )
        self._delete_btn = ctk.CTkButton(
            af, text="🗑  Delete",
            font=ctk.CTkFont(size=12), width=90, height=36,
            fg_color=C["card"], hover_color=C["rec"],
            command=self._delete_meeting,
        )
        # Pack default layout (new-meeting mode)
        self._save_btn.pack(side="left", padx=(0, 6))
        self._summarize_btn.pack(side="left", padx=(0, 6))
        self._export_btn.pack(side="left", padx=(0, 6))
        self._clear_btn.pack(side="left")

        # ── Transcript header row (label + Copy/Share)
        thdr = ctk.CTkFrame(main, fg_color="transparent")
        thdr.grid(row=1, column=0, sticky="ew", padx=16, pady=(14, 4))
        thdr.grid_columnconfigure(1, weight=1)

        self._transcript_lbl = ctk.CTkLabel(
            thdr, text="LIVE TRANSCRIPT",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=C["text_dim"], anchor="w")
        self._transcript_lbl.grid(row=0, column=0, sticky="w")

        self._txn_copy_btn = ctk.CTkButton(
            thdr, text="📋  Copy",
            font=ctk.CTkFont(size=10), width=72, height=24,
            fg_color=C["card"], hover_color=C["accent"],
            command=self._copy_transcript,
        )
        self._txn_copy_btn.grid(row=0, column=2, padx=(0, 4))

        ctk.CTkButton(
            thdr, text="↗  Share",
            font=ctk.CTkFont(size=10), width=72, height=24,
            fg_color=C["card"], hover_color=C["accent"],
            command=self._share_transcript,
        ).grid(row=0, column=3)

        # ── Transcript textbox
        self._transcript_box = ctk.CTkTextbox(
            main,
            font=ctk.CTkFont(size=14, family="Helvetica Neue"),
            fg_color=C["panel"], text_color=C["text"],
            border_color=C["border"], border_width=1,
            corner_radius=8, wrap="word",
        )
        self._transcript_box.grid(
            row=2, column=0, sticky="nsew", padx=16, pady=(0, 6))
        self._transcript_box.configure(state="disabled")
        self._set_transcript_placeholder()

        # ── Summary header row
        shdr = ctk.CTkFrame(main, fg_color="transparent")
        shdr.grid(row=3, column=0, sticky="ew", padx=16, pady=(6, 4))
        shdr.grid_columnconfigure(1, weight=1)
        self._sum_hdr = shdr

        self._summary_lbl = ctk.CTkLabel(
            shdr, text="SUMMARY & ACTION ITEMS",
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=C["text_dim"])
        self._summary_lbl.grid(row=0, column=0, sticky="w")

        self._model_lbl = ctk.CTkLabel(
            shdr, text="",
            font=ctk.CTkFont(size=9), text_color=C["text_dim"])
        self._model_lbl.grid(row=0, column=1, padx=8, sticky="w")

        # Copy/Share buttons for summary
        self._sum_copy_btn = ctk.CTkButton(
            shdr, text="📋  Copy",
            font=ctk.CTkFont(size=10), width=72, height=24,
            fg_color=C["card"], hover_color=C["accent"],
            command=self._copy_summary,
        )
        self._sum_copy_btn.grid(row=0, column=2, padx=(0, 4))

        ctk.CTkButton(
            shdr, text="↗  Share",
            font=ctk.CTkFont(size=10), width=72, height=24,
            fg_color=C["card"], hover_color=C["accent"],
            command=self._share_summary,
        ).grid(row=0, column=3, padx=(0, 4))

        # "Save Changes" button — only visible in past-meeting mode (column 4)
        self._past_save_btn = ctk.CTkButton(
            shdr, text="💾  Save Changes",
            font=ctk.CTkFont(size=12), width=120, height=30,
            fg_color=C["card"], hover_color=C["success"],
            command=self._save_past_summary,
        )
        # not gridded initially

        # ── Summary textbox
        self._summary_box = ctk.CTkTextbox(
            main,
            font=ctk.CTkFont(size=13, family="Helvetica Neue"),
            fg_color=C["panel"], text_color=C["text"],
            border_color=C["border"], border_width=1,
            corner_radius=8, wrap="word",
        )
        self._summary_box.grid(
            row=4, column=0, sticky="nsew", padx=16, pady=(0, 4))
        self._summary_box.configure(state="disabled")
        self._set_summary_placeholder()

        # ── Refine row
        refine_frame = ctk.CTkFrame(main, fg_color=C["panel"], corner_radius=8)
        refine_frame.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 12))
        refine_frame.grid_columnconfigure(0, weight=1)

        self._refine_entry = ctk.CTkEntry(
            refine_frame,
            placeholder_text="Refine summary: e.g. 'make it shorter', 'focus on action items'…",
            font=ctk.CTkFont(size=12),
            fg_color=C["card"], border_color=C["border"], height=34,
        )
        self._refine_entry.grid(row=0, column=0, padx=(12, 6), pady=8, sticky="ew")
        self._refine_entry.bind("<Return>", lambda e: self._do_refine())

        self._refine_btn = ctk.CTkButton(
            refine_frame, text="✏️  Refine",
            font=ctk.CTkFont(size=12), width=100, height=34,
            fg_color=C["card"], hover_color=C["accent"],
            command=self._do_refine,
        )
        self._refine_btn.grid(row=0, column=1, padx=(0, 12), pady=8)

        # ── Status bar
        sbar = ctk.CTkFrame(main, fg_color=C["panel"], height=24, corner_radius=0)
        sbar.grid(row=6, column=0, sticky="ew")
        sbar.grid_columnconfigure(0, weight=1)
        sbar.grid_propagate(False)
        self._status_var = ctk.StringVar(value="Ready")
        ctk.CTkLabel(sbar, textvariable=self._status_var,
                     font=ctk.CTkFont(size=10), text_color=C["text_dim"],
                     anchor="w").grid(row=0, column=0, padx=12, pady=2, sticky="w")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Meetings List
    # ═══════════════════════════════════════════════════════════════════════════

    def _make_meeting_row(self, info: dict, grid_row: int):
        path = info["path"]
        row = ctk.CTkFrame(
            self._meetings_scroll,
            fg_color=C["card"], corner_radius=6, cursor="hand2",
        )
        row.grid(row=grid_row, column=0, sticky="ew", padx=6, pady=2)
        row.grid_columnconfigure(0, weight=1)

        primary = info.get("title") or info["time"]
        secondary = info["time"] if info.get("title") else info["synopsis"]

        time_lbl = ctk.CTkLabel(
            row, text=primary,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C["text"], anchor="w", wraplength=220, justify="left")
        time_lbl.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 1))

        syn_lbl = ctk.CTkLabel(
            row, text=secondary,
            font=ctk.CTkFont(size=10), text_color=C["text_mid"],
            anchor="w", wraplength=216, justify="left")
        syn_lbl.grid(row=1, column=0, sticky="w", padx=10, pady=(0, 8))

        def on_enter(e):
            if self._viewing_path != path:
                row.configure(fg_color=C["card_h"])

        def on_leave(e):
            if self._viewing_path != path:
                row.configure(fg_color=C["card"])

        def on_click(e):
            self._open_meeting(path)

        for widget in (row, time_lbl, syn_lbl):
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
            widget.bind("<Button-1>", on_click)

        self._meeting_rows[path] = row
        return row

    def _set_selected_row(self, path: Optional[Path]):
        for p, row in self._meeting_rows.items():
            try:
                row.configure(fg_color=C["sel"] if p == path else C["card"])
            except Exception:
                pass

    # ═══════════════════════════════════════════════════════════════════════════
    #  View Switching
    # ═══════════════════════════════════════════════════════════════════════════

    def _select_new_meeting(self):
        """Switch to the live-recording view."""
        self._current_view   = "new"
        self._viewing_path   = None
        self._viewing_xscript = ""

        # Topbar: restore rec_btn + new-title entry + timer, hide past-mode widgets
        self._ctx_title.grid_remove()
        self._title_entry.grid_remove()
        self._rec_btn.grid(row=0, column=0, padx=16, pady=12)
        self._new_title_entry.grid(row=0, column=1, padx=(0, 8), pady=15, sticky="w")
        self._timer_lbl.grid(row=0, column=2, padx=8, sticky="w")

        # Action buttons: restore new-meeting set
        for btn in (self._pause_btn, self._save_btn, self._summarize_btn,
                    self._export_btn, self._clear_btn, self._delete_btn):
            btn.pack_forget()
        self._save_btn.pack(side="left", padx=(0, 6))
        self._summarize_btn.pack(side="left", padx=(0, 6))
        self._export_btn.pack(side="left", padx=(0, 6))
        self._clear_btn.pack(side="left")
        self._save_btn.configure(command=self._save_transcript)
        self._summarize_btn.configure(text="✨  Summarize",
                                       state="normal", command=self._run_summary)

        # Hide past-save button
        self._past_save_btn.grid_remove()

        # Restore labels
        self._transcript_lbl.configure(text="LIVE TRANSCRIPT")
        self._summary_lbl.configure(text="SUMMARY & ACTION ITEMS")

        # Restore live transcript
        self._transcript_box.configure(state="normal")
        self._transcript_box.delete("1.0", "end")
        if self._transcript_segments:
            for seg in self._transcript_segments:
                ts  = datetime.datetime.fromtimestamp(seg.timestamp).strftime("%H:%M:%S")
                ico = "🎤" if seg.source == "microphone" else "🖥"
                self._transcript_box.insert("end", f"[{ts}] {ico}  {seg.text}\n\n")
            self._transcript_box.see("end")
        else:
            self._set_transcript_placeholder_inline()
        self._transcript_box.configure(state="disabled")

        # Restore live session summary (or placeholder)
        self._summary_box.configure(state="normal")
        self._summary_box.delete("1.0", "end")
        if self._live_summary:
            self._summary_box.insert("1.0", self._live_summary)
        else:
            self._summary_box.insert("1.0",
                "Click ✨ Summarize after your meeting to get an AI-written summary "
                "and action items — all processed locally via Ollama.\n\n"
                "Once generated, use the Refine box below to adjust verbosity or focus.")
        self._summary_box.configure(state="disabled")

        # Deselect all rows
        self._set_selected_row(None)
        self._refine_btn.configure(state="normal")
        self._set_status("New meeting — press Start Recording")

    def _open_meeting(self, path: Path):
        """Switch to the past-meeting view for a saved transcript."""
        try:
            self._open_meeting_inner(path)
        except Exception as e:
            import traceback
            self._set_status(f"⚠ Error opening meeting: {e}")
            traceback.print_exc()

    def _open_meeting_inner(self, path: Path):
        # Save live session summary before switching away
        try:
            s = self._summary_box.get("1.0", "end").strip()
            if not s.startswith("Click ✨"):
                self._live_summary = s
        except Exception:
            pass

        self._current_view    = "past"
        self._viewing_path    = path
        self._set_selected_row(path)

        info = _parse_file_cached(path)
        self._viewing_xscript = info["transcript_text"]

        # Topbar: hide new-meeting widgets, show title entry + date label
        self._rec_btn.grid_remove()
        self._timer_lbl.grid_remove()
        self._new_title_entry.grid_remove()
        self._title_var.set(info.get("title", ""))
        self._title_entry.grid(row=0, column=0, padx=(16, 4), pady=12, sticky="w")
        self._ctx_title.configure(text=f"📅  {info['time']}")
        self._ctx_title.grid(row=0, column=1, padx=(0, 8), pady=12, sticky="w")

        # Action buttons: past-meeting set (Summarize + Export + Delete)
        for btn in (self._pause_btn, self._save_btn, self._summarize_btn,
                    self._export_btn, self._clear_btn, self._delete_btn):
            btn.pack_forget()
        self._summarize_btn.pack(side="left", padx=(0, 6))
        self._export_btn.pack(side="left", padx=(0, 6))
        self._delete_btn.pack(side="left")
        self._summarize_btn.configure(text="✨  Summarize", state="normal",
                                       command=self._run_summary_past)

        # Show "Save Changes" in summary header
        self._past_save_btn.grid(row=0, column=4)

        # Labels
        self._transcript_lbl.configure(text="TRANSCRIPT")
        self._summary_lbl.configure(text="SUMMARY")

        # Load transcript (read-only)
        self._transcript_box.configure(state="normal")
        self._transcript_box.delete("1.0", "end")
        self._transcript_box.insert("1.0",
            info["transcript_text"] or "(No transcript content saved.)")
        self._transcript_box.configure(state="disabled")

        # Load summary (editable so user can adjust before saving)
        self._summary_box.configure(state="normal")
        self._summary_box.delete("1.0", "end")
        if info["summary_text"]:
            self._summary_box.insert("1.0", info["summary_text"])
            # leave enabled for inline editing
        else:
            self._summary_box.insert("1.0",
                "No summary saved yet.\n\n"
                "Click ✨ Summarize to generate one with local AI.")
            self._summary_box.configure(state="disabled")

        self._refine_btn.configure(state="normal")
        self._set_status(f"Viewing: {path.name}")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Backend Init
    # ═══════════════════════════════════════════════════════════════════════════

    def _init_backend(self):
        def _load():
            try:
                self._set_status("Loading Whisper model…")
                self._transcriber = Transcriber(
                    on_segment=self._on_transcript_segment,
                    on_status=lambda s: self._ui_queue.put(("status", s)),
                )
                self._transcriber.load_model()
                self._summarizer = Summarizer(
                    on_token=lambda t: self._ui_queue.put(("summary_token", t)),
                    on_done=lambda _: self._ui_queue.put(("summary_done", None)),
                    on_error=lambda e: self._ui_queue.put(("summary_error", e)),
                )
                backend = self._transcriber._backend
                avail   = self._summarizer.is_available()
                self._ui_queue.put(("backend_label",
                    f"Whisper {backend} · Ollama: {'✓' if avail else '✗'}"))
                self._ui_queue.put(("status",
                    "Ready — press Start Recording"))
                self._ui_queue.put(("model_lbl",
                    f"via Ollama ({self._summarizer.model})" if avail
                    else "Ollama not detected"))
            except Exception as e:
                self._ui_queue.put(("status", f"Error loading model: {e}"))

        threading.Thread(target=_load, daemon=True).start()
        # Load today's calendar events in the background
        threading.Thread(target=self._load_calendar, daemon=True).start()

    def _load_calendar(self):
        try:
            events = get_todays_events()
            self._ui_queue.put(("calendar_ready", events))
        except Exception:
            pass

    def _build_calendar_section(self, events: list):
        """Inject calendar events into the top of the meetings scroll area."""
        if not events:
            return
        # Remove any existing calendar section widgets tagged with "cal_"
        for w in self._meetings_scroll.winfo_children():
            if getattr(w, "_is_cal_widget", False):
                w.destroy()

        # Re-insert at the top by rebuilding the list
        self._refresh_meetings_list()

    def _setup_meetings_scroll(self):
        """
        Enable trackpad / mouse-wheel scrolling for the meetings list.
        Binds directly to the CTkScrollableFrame's internal canvas and
        propagates to every child widget so scrolling works regardless of
        which card the cursor is over.  Called after every list rebuild.
        """
        try:
            canvas = self._meetings_scroll._parent_canvas
        except AttributeError:
            return

        def _on_scroll(event):
            try:
                canvas.yview_scroll(int(-event.delta), "units")
            except Exception:
                pass

        # Bind to the canvas itself (catches hover over empty space)
        canvas.bind("<MouseWheel>", _on_scroll)

        # Recursively bind to every child so card hover also scrolls
        def _bind_tree(widget):
            try:
                widget.bind("<MouseWheel>", _on_scroll)
                for child in widget.winfo_children():
                    _bind_tree(child)
            except Exception:
                pass

        _bind_tree(self._meetings_scroll)

    def _refresh_meetings_list(self):
        """Rebuild the sidebar meetings list from disk, with calendar events at top."""
        for w in self._meetings_scroll.winfo_children():
            w.destroy()
        self._meeting_rows.clear()
        self._group_headers.clear()

        grid_row = 0

        # ── Today's upcoming calendar events ─────────────────────────────────
        upcoming = [e for e in self._calendar_events
                    if e.start.date() == datetime.datetime.now().date()]
        if upcoming:
            ctk.CTkLabel(
                self._meetings_scroll, text="UPCOMING TODAY",
                font=ctk.CTkFont(size=9, weight="bold"),
                text_color=C["accent"], anchor="w",
            ).grid(row=grid_row, column=0, sticky="w",
                   padx=10, pady=(8, 2))
            grid_row += 1
            for evt in upcoming:
                self._make_calendar_row(evt, grid_row)
                grid_row += 1

        # ── Saved meetings ────────────────────────────────────────────────────
        all_files = sorted(SAVES_DIR.glob("*.md"), key=_file_sort_key, reverse=True)
        term = getattr(self, "_search_term", "")
        if term:
            files = [f for f in all_files
                     if term in f.name.lower() or
                     term in _parse_file_cached(f)["synopsis"].lower() or
                     term in _parse_file_cached(f)["transcript_text"].lower()]
        else:
            files = all_files

        if not files and not upcoming:
            msg = "No meetings match your search." if term else \
                  "No saved meetings yet.\n\nStop a recording to auto-save."
            ctk.CTkLabel(
                self._meetings_scroll, text=msg,
                font=ctk.CTkFont(size=11), text_color=C["text_dim"],
                justify="center",
            ).grid(row=grid_row, column=0, padx=8, pady=28)
            return

        current_group = None
        for f in files:
            info  = _parse_file_cached(f)
            group = info["group"]
            if group != current_group:
                current_group = group
                hdr = ctk.CTkLabel(
                    self._meetings_scroll, text=group.upper(),
                    font=ctk.CTkFont(size=9, weight="bold"),
                    text_color=C["text_dim"], anchor="w",
                )
                hdr.grid(row=grid_row, column=0, sticky="w",
                         padx=10, pady=(12 if grid_row > 0 else 4, 2))
                self._group_headers[group] = hdr
                grid_row += 1
            self._make_meeting_row(info, grid_row)
            grid_row += 1

        if self._viewing_path:
            self._set_selected_row(self._viewing_path)

        # Rebuild month jump chips, then re-bind scroll to new child widgets
        self._rebuild_month_nav()
        self.after(50, self._setup_meetings_scroll)

    def _rebuild_month_nav(self):
        """Repopulate the month chip strip above the meetings list."""
        for w in self._month_nav.winfo_children():
            w.destroy()

        # Only show month-style groups (not Today/Yesterday/This Week)
        special = {"Today", "Yesterday", "This Week"}
        months  = [g for g in self._group_headers if g not in special]
        if not months:
            return

        for i, month in enumerate(months):
            # Abbreviated chip label: "Jun '26" style
            try:
                dt  = datetime.datetime.strptime(month, "%B %Y")
                lbl = dt.strftime("%b '%y")
            except Exception:
                lbl = month[:6]

            btn = ctk.CTkButton(
                self._month_nav,
                text=lbl,
                font=ctk.CTkFont(size=9),
                width=44, height=20, corner_radius=10,
                fg_color=C["card"], hover_color=C["accent"],
                text_color=C["text_mid"],
                command=lambda g=month: self._jump_to_group(g),
            )
            btn.pack(side="left", padx=(0 if i == 0 else 3, 0))

    def _jump_to_group(self, group: str):
        """Scroll the meetings list to the given group header."""
        hdr = self._group_headers.get(group)
        if not hdr:
            return
        try:
            # CTkScrollableFrame exposes its internal canvas as _parent_canvas
            canvas = self._meetings_scroll._parent_canvas
            self._meetings_scroll.update_idletasks()
            y     = hdr.winfo_y()
            total = self._meetings_scroll.winfo_reqheight()
            if total > 0:
                canvas.yview_moveto(max(0.0, (y - 8) / total))
        except Exception:
            pass

    def _make_calendar_row(self, evt: CalendarEvent, grid_row: int):
        """Sidebar card for an upcoming calendar event."""
        is_now = evt.is_now
        row = ctk.CTkFrame(
            self._meetings_scroll,
            fg_color=C["sel"] if is_now else C["card"],
            corner_radius=6, cursor="hand2",
        )
        row.grid(row=grid_row, column=0, sticky="ew", padx=6, pady=2)
        row.grid_columnconfigure(0, weight=1)

        # Dot indicator — green if happening now
        dot   = "🔴 " if is_now else ""
        title = ctk.CTkLabel(
            row, text=f"{dot}{evt.title}",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=C["text"], anchor="w", wraplength=200, justify="left")
        title.grid(row=0, column=0, sticky="w", padx=10, pady=(8, 1))

        time_lbl = ctk.CTkLabel(
            row, text=evt.time_str,
            font=ctk.CTkFont(size=10), text_color=C["text_mid"], anchor="w")
        time_lbl.grid(row=1, column=0, sticky="w", padx=10)

        btn_row = ctk.CTkFrame(row, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="w", padx=8, pady=(4, 8))

        if evt.has_link:
            import subprocess as _sp
            ctk.CTkButton(
                btn_row, text="▶  Join",
                font=ctk.CTkFont(size=10), width=62, height=24,
                fg_color=C["success"], hover_color=C["success_dk"],
                command=lambda u=evt.url: _sp.Popen(["open", u]),
            ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="⏺  Record",
            font=ctk.CTkFont(size=10), width=72, height=24,
            fg_color=C["accent"], hover_color=C["accent_lt"],
            command=lambda e=evt: self._start_from_calendar(e),
        ).pack(side="left")

    def _start_from_calendar(self, evt: CalendarEvent):
        """Pre-fill title from calendar event and switch to new-meeting view."""
        self._select_new_meeting()
        self._new_title_var.set(evt.title)
        self._pending_title = evt.title
        if evt.has_link:
            import subprocess as _sp
            _sp.Popen(["open", evt.url])
        self._set_status(f"Ready to record: {evt.title}")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Recording
    # ═══════════════════════════════════════════════════════════════════════════

    def _toggle_recording(self):
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        src_map   = {"Mic": "microphone", "System": "system", "Both": "both"}
        source_str = src_map.get(self._source_var.get(), "microphone")
        source     = AudioSource(source_str)

        if source in (AudioSource.SYSTEM, AudioSource.BOTH) and not self._blackhole_available:
            self._set_status("⚠ BlackHole not found — install it to capture system audio")
            return

        lang_map = {
            "Auto-detect": None, "English": "en", "Spanish": "es",
            "French": "fr", "German": "de", "Japanese": "ja",
            "Chinese": "zh", "Hindi": "hi",
        }
        self._transcriber.language = lang_map.get(self._lang_var.get())
        self._transcriber.start()

        def _on_chunk(chunk):
            self._transcriber.enqueue(chunk)
            self._audio_chunks.append({"audio": chunk.audio, "timestamp": chunk.timestamp})

        try:
            self._capturer = AudioCapturer(
                source=source, on_chunk=_on_chunk,
                on_level=lambda v: self._ui_queue.put(("level", v)),
            )
            self._capturer.start()
        except Exception as e:
            self._set_status(f"Audio error: {e}")
            return

        self._recording     = True
        self._paused        = False
        self._session_start = time.time()
        self._audio_chunks  = []
        self._speaker_map   = {}
        self._pending_title = self._new_title_var.get().strip()
        # Warm up the LLM while recording so it's ready when the meeting ends
        if self._summarizer:
            self._summarizer.preload()
        # Preload diarization model in background if configured
        if self._diarizer.is_available():
            self._diarizer.preload(
                on_status=lambda s: self._ui_queue.put(("status", s))
            )
        self._rec_btn.configure(text="⏹  Stop Recording",
                                 fg_color=C["rec"], hover_color=C["rec_dk"])
        # Show pause button
        self._pause_btn.pack_forget()
        self._save_btn.pack_forget()
        self._summarize_btn.pack_forget()
        self._export_btn.pack_forget()
        self._clear_btn.pack_forget()
        self._pause_btn.pack(side="left", padx=(0, 6))
        self._save_btn.pack(side="left", padx=(0, 6))
        self._summarize_btn.pack(side="left", padx=(0, 6))
        self._export_btn.pack(side="left", padx=(0, 6))
        self._clear_btn.pack(side="left")
        self._set_status(f"Recording · {source_str}")
        self._tick_timer()

    def _stop_recording(self):
        self._recording = False
        self._paused    = False
        if self._capturer:
            self._capturer.stop()
            self._capturer = None
        if self._elapsed_timer_id:
            try:
                self.after_cancel(self._elapsed_timer_id)
            except Exception:
                pass
        # Capture duration
        if self._session_start:
            self._session_duration = int(time.time() - self._session_start)
        self._rec_btn.configure(text="⏺  Start Recording",
                                 fg_color=C["success"], hover_color=C["success_dk"])
        # Hide pause button
        self._pause_btn.pack_forget()
        self._pause_btn.configure(text="⏸  Pause", fg_color=C["card"],
                                   hover_color=C["warning"])
        n = len(self._transcript_segments)
        self._set_status(f"Stopped · {n} segments captured")
        self._level_bar.set(0)
        self._timer_lbl.configure(text_color=C["text_dim"])
        if self._transcript_segments:
            self.after(500, lambda: self._save_transcript(auto=True))
            # Auto-summarize after save settles
            if self._summarizer and self._summarizer.is_available():
                self.after(2000, self._auto_summarize_on_stop)
            # Kick off diarization in background
            if self._diarizer.is_available() and self._audio_chunks:
                session_start = self._session_start
                chunks        = list(self._audio_chunks)
                self.after(300, lambda: self._start_diarization(chunks, session_start))

    def _toggle_pause(self):
        if not self._recording:
            return
        if self._paused:
            # Resume
            self._paused = False
            if self._capturer:
                self._capturer.resume()
            self._pause_btn.configure(text="⏸  Pause", fg_color=C["card"],
                                       hover_color=C["warning"])
            self._set_status("Recording resumed")
            self._tick_timer()
        else:
            # Pause
            self._paused = True
            if self._capturer:
                self._capturer.pause()
            if self._elapsed_timer_id:
                try:
                    self.after_cancel(self._elapsed_timer_id)
                except Exception:
                    pass
            self._pause_btn.configure(text="▶  Resume", fg_color=C["warning"],
                                       hover_color=C["success"])
            self._timer_lbl.configure(text_color=C["warning"])
            self._set_status("Recording paused — press Resume to continue")

    def _start_diarization(self, chunks: list, session_start: float):
        """Build WAV and start diarization pipeline."""
        self._set_status("Building audio for speaker diarization…")
        wav_path = Diarizer.build_wav(chunks, session_start)
        if not wav_path:
            return
        self._diarizer.diarize(
            wav_path,
            on_done=lambda segs: self._ui_queue.put(("diarization_done",
                                                     (segs, session_start))),
            on_error=lambda e: self._ui_queue.put(("status",
                                                   f"Diarization error: {e}")),
            on_status=lambda s: self._ui_queue.put(("status", s)),
        )

    def _apply_diarization(self, diar_segments: list, session_start: float):
        """Map speaker segments onto transcript, re-render with names."""
        self._speaker_map = Diarizer.assign_speakers(
            self._transcript_segments, diar_segments, session_start
        )
        # Assign labels to segment objects
        speakers = sorted(set(self._speaker_map.values()))
        label_map = {sp: f"Speaker {i+1}" for i, sp in enumerate(speakers)}
        for i, seg in enumerate(self._transcript_segments):
            raw = self._speaker_map.get(i, "")
            seg.speaker = label_map.get(raw, "")

        # Re-render transcript box
        if self._current_view == "new":
            self._transcript_box.configure(state="normal")
            self._transcript_box.delete("1.0", "end")
            for seg in self._transcript_segments:
                ts  = datetime.datetime.fromtimestamp(seg.timestamp).strftime("%H:%M:%S")
                ico = "🎤" if seg.source == "microphone" else "🖥"
                speaker_tag = f"[{seg.speaker}] " if seg.speaker else ""
                self._transcript_box.insert("end",
                    f"[{ts}] {ico} {speaker_tag} {seg.text}\n\n")
            self._transcript_box.see("end")
            self._transcript_box.configure(state="disabled")

        # Re-save with speaker labels
        if self._current_save_path and self._current_save_path.exists():
            try:
                text = self._current_save_path.read_text(encoding="utf-8")
                new_lines = []
                for seg in self._transcript_segments:
                    ts_str = datetime.datetime.fromtimestamp(seg.timestamp).strftime("%H:%M:%S")
                    ico    = "🎤" if seg.source == "microphone" else "🖥"
                    sp     = f" **[{seg.speaker}]**" if seg.speaker else ""
                    new_lines.append(f"**[{ts_str}]** {ico}{sp} {seg.text}\n\n")
                # Replace Full Transcript section
                marker = "## Full Transcript\n\n"
                if marker in text:
                    text = text[:text.index(marker) + len(marker)] + "".join(new_lines)
                    self._current_save_path.write_text(text, encoding="utf-8")
                    _invalidate_cache(self._current_save_path)
                    self.after(300, self._refresh_meetings_list)
            except Exception:
                pass
        n_speakers = len(set(self._speaker_map.values()))
        self._set_status(f"Diarization complete — {n_speakers} speaker(s) identified")

        # Build dialog info: (display_label, sample_quote)
        speakers_info = []
        for raw_label, display_name in sorted(label_map.items(),
                                               key=lambda x: x[1]):
            sample = ""
            for i, seg in enumerate(self._transcript_segments):
                if self._speaker_map.get(i) == raw_label and seg.text.strip():
                    sample = seg.text.strip()[:90]
                    if len(seg.text.strip()) > 90:
                        sample += "…"
                    break
            speakers_info.append((display_name, sample))

        if n_speakers > 0:
            self.after(200, lambda: self._show_speaker_name_dialog(speakers_info))

    def _show_speaker_name_dialog(self, speakers_info: list):
        """Open the speaker naming modal and apply results."""
        dialog = SpeakerNameDialog(self, speakers_info)
        self.wait_window(dialog)
        name_map = dialog.get_result()   # {old_label: new_name} or {} if cancelled
        if name_map:
            self._apply_speaker_names(name_map)

    def _apply_speaker_names(self, name_map: dict):
        """Replace generic labels with real names in segments, transcript, and file."""
        for seg in self._transcript_segments:
            if seg.speaker in name_map:
                seg.speaker = name_map[seg.speaker]

        # Re-render transcript box
        if self._current_view == "new":
            self._transcript_box.configure(state="normal")
            self._transcript_box.delete("1.0", "end")
            for seg in self._transcript_segments:
                ts  = datetime.datetime.fromtimestamp(seg.timestamp).strftime("%H:%M:%S")
                ico = "🎤" if seg.source == "microphone" else "🖥"
                sp  = f"[{seg.speaker}] " if seg.speaker else ""
                self._transcript_box.insert("end", f"[{ts}] {ico} {sp}{seg.text}\n\n")
            self._transcript_box.see("end")
            self._transcript_box.configure(state="disabled")

        # Overwrite saved file
        if self._current_save_path and self._current_save_path.exists():
            try:
                text = self._current_save_path.read_text(encoding="utf-8")
                new_lines = []
                for seg in self._transcript_segments:
                    ts_str = datetime.datetime.fromtimestamp(seg.timestamp).strftime("%H:%M:%S")
                    ico    = "🎤" if seg.source == "microphone" else "🖥"
                    sp     = f" **[{seg.speaker}]**" if seg.speaker else ""
                    new_lines.append(f"**[{ts_str}]** {ico}{sp} {seg.text}\n\n")
                marker = "## Full Transcript\n\n"
                if marker in text:
                    text = text[:text.index(marker) + len(marker)] + "".join(new_lines)
                    self._current_save_path.write_text(text, encoding="utf-8")
                    _invalidate_cache(self._current_save_path)
                    self.after(300, self._refresh_meetings_list)
            except Exception:
                pass
        self._set_status("Speaker names applied ✓")

    def _auto_summarize_on_stop(self):
        """Auto-trigger summarize after recording stops (if no summary yet)."""
        if self._live_summary or not self._transcript_segments:
            return
        if self._current_view == "new":
            self._run_summary()

    def _tick_timer(self):
        if not self._recording or self._paused or self._session_start is None:
            return
        elapsed = int(time.time() - self._session_start)
        h, rem  = divmod(elapsed, 3600)
        m, s    = divmod(rem, 60)
        text = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        self._timer_lbl.configure(text=text, text_color=C["rec"])
        self._elapsed_timer_id = self.after(1000, self._tick_timer)

    # ═══════════════════════════════════════════════════════════════════════════
    #  Transcript
    # ═══════════════════════════════════════════════════════════════════════════

    def _on_transcript_segment(self, seg: TranscriptSegment):
        self._ui_queue.put(("segment", seg))

    def _append_segment(self, seg: TranscriptSegment):
        if self._current_view != "new":
            # buffer silently — user can switch back later
            self._transcript_segments.append(seg)
            return
        self._transcript_segments.append(seg)
        ts  = datetime.datetime.fromtimestamp(seg.timestamp).strftime("%H:%M:%S")
        ico = "🎤" if seg.source == "microphone" else "🖥"
        line = f"[{ts}] {ico}  {seg.text}\n\n"

        self._transcript_box.configure(state="normal")
        current = self._transcript_box.get("1.0", "end").strip()
        if current.startswith("Your transcript"):
            self._transcript_box.delete("1.0", "end")
        self._transcript_box.insert("end", line)
        self._transcript_box.see("end")
        self._transcript_box.configure(state="disabled")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Summary & Refinement
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_summary(self):
        """Generate summary for the live (new-meeting) session."""
        if not self._transcript_segments:
            self._set_status("No transcript yet — record a meeting first.")
            return
        if not self._summarizer:
            self._set_status("Summarizer not ready.")
            return
        self._start_summary_stream("\n".join(s.text for s in self._transcript_segments))

    def _run_summary_past(self):
        """Generate summary for the currently viewed past meeting."""
        if not self._viewing_xscript:
            self._set_status("No transcript content to summarize.")
            return
        if not self._summarizer:
            self._set_status("Summarizer not ready.")
            return
        self._summary_box.configure(state="normal")
        self._start_summary_stream(self._viewing_xscript)

    def _start_summary_stream(self, transcript: str):
        self._summary_box.configure(state="normal")
        self._summary_box.delete("1.0", "end")
        self._summary_box.configure(state="disabled")
        self._summarize_btn.configure(text="⏳  Generating…", state="disabled")
        self._refine_btn.configure(state="disabled")
        self._set_status("Generating summary with local LLM…")
        self._summarizer.summarize(transcript)

    def _do_refine(self):
        """Shared refine action — works for both new and past meeting views."""
        instruction = self._refine_entry.get().strip()
        if not instruction:
            return

        try:
            current_summary = self._summary_box.get("1.0", "end").strip()
        except Exception:
            current_summary = ""

        # Skip if it's a placeholder
        if not current_summary or current_summary.startswith("Click ✨") \
                or current_summary.startswith("No summary"):
            self._set_status("Generate a summary first, then use Refine.")
            return
        if not self._summarizer:
            return

        if self._current_view == "new":
            transcript = "\n".join(s.text for s in self._transcript_segments)
        else:
            transcript = self._viewing_xscript

        if not transcript:
            self._set_status("No transcript to base the refinement on.")
            return

        self._summary_box.configure(state="normal")
        self._summary_box.delete("1.0", "end")
        self._summary_box.configure(state="disabled")
        self._summarize_btn.configure(state="disabled")
        self._refine_btn.configure(text="⏳  Refining…", state="disabled")
        self._set_status(f'Refining: "{instruction}"…')
        self._summarizer.refine(transcript, current_summary, instruction)
        self._refine_entry.delete(0, "end")

    def _append_summary_token(self, token: str):
        self._summary_box.configure(state="normal")
        self._summary_box.insert("end", token)
        self._summary_box.see("end")
        self._summary_box.configure(state="disabled")

    def _on_summary_done(self):
        if self._current_view == "past":
            self._summarize_btn.configure(text="✨  Summarize", state="normal",
                                           command=self._run_summary_past)
            # Leave summary_box editable so user can tweak before saving
            self._summary_box.configure(state="normal")
            # Auto-write summary into the viewed file
            if self._viewing_path:
                try:
                    new_summary = self._summary_box.get("1.0", "end").strip()
                    self._write_summary_to_file(self._viewing_path, new_summary)
                except Exception:
                    pass
        else:
            self._summarize_btn.configure(text="✨  Summarize", state="normal",
                                           command=self._run_summary)
            # Cache live summary for view switching
            try:
                self._live_summary = self._summary_box.get("1.0", "end").strip()
            except Exception:
                pass
            # Auto-write summary into the auto-saved file
            if self._current_save_path and self._current_save_path.exists():
                try:
                    self._write_summary_to_file(self._current_save_path, self._live_summary)
                    self._set_status("Summary complete — saved to file automatically")
                except Exception:
                    pass
        self._refine_btn.configure(text="✏️  Refine", state="normal")
        self._set_status("Summary complete — use Refine box to adjust")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Copy / Share
    # ═══════════════════════════════════════════════════════════════════════════

    def _flash_btn(self, btn: ctk.CTkButton, original: str, flash: str = "✓ Copied!", ms: int = 1800):
        """Briefly change a button's label then restore it."""
        btn.configure(text=flash)
        self.after(ms, lambda: btn.configure(text=original))

    def _get_transcript_text(self) -> str:
        if self._current_view == "new":
            if not self._transcript_segments:
                return ""
            lines = []
            for seg in self._transcript_segments:
                ts  = datetime.datetime.fromtimestamp(seg.timestamp).strftime("%H:%M:%S")
                ico = "🎤" if seg.source == "microphone" else "🖥"
                lines.append(f"[{ts}] {ico}  {seg.text}")
            return "\n\n".join(lines)
        else:
            return self._viewing_xscript

    def _get_summary_text(self) -> str:
        try:
            text = self._summary_box.get("1.0", "end").strip()
            if text.startswith("Click ✨") or text.startswith("No summary"):
                return ""
            return text
        except Exception:
            return ""

    def _copy_transcript(self):
        text = self._get_transcript_text()
        if not text:
            self._set_status("No transcript to copy.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self._flash_btn(self._txn_copy_btn, "📋  Copy")
        self._set_status("Transcript copied to clipboard")

    def _share_transcript(self):
        import tempfile, subprocess
        text = self._get_transcript_text()
        if not text:
            self._set_status("No transcript to share.")
            return
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix=f"transcript_{now}_"
        ) as f:
            f.write(text)
            tmp = f.name
        subprocess.Popen(["open", tmp])
        self._set_status("Transcript opened for sharing")

    def _copy_summary(self):
        text = self._get_summary_text()
        if not text:
            self._set_status("No summary to copy — generate one first.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self._flash_btn(self._sum_copy_btn, "📋  Copy")
        self._set_status("Summary copied to clipboard")

    def _share_summary(self):
        import tempfile, subprocess
        text = self._get_summary_text()
        if not text:
            self._set_status("No summary to share — generate one first.")
            return
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, prefix=f"summary_{now}_"
        ) as f:
            f.write(text)
            tmp = f.name
        subprocess.Popen(["open", tmp])
        self._set_status("Summary opened for sharing")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Save / Clear
    # ═══════════════════════════════════════════════════════════════════════════

    def _save_transcript(self, auto: bool = False) -> Optional[Path]:
        if not self._transcript_segments:
            if not auto:
                self._set_status("Nothing to save yet.")
            return None

        ts    = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = SAVES_DIR / f"meeting_{ts}.md"

        summary_text = ""
        try:
            summary_text = self._summary_box.get("1.0", "end").strip()
            if summary_text.startswith("Click") or summary_text.startswith("No summary"):
                summary_text = ""
        except Exception:
            pass

        dur_str = ""
        if self._session_duration > 0:
            dm, ds = divmod(self._session_duration, 60)
            dh, dm = divmod(dm, 60)
            dur_str = f"{dh}h {dm}m" if dh else f"{dm}m {ds}s"

        title = getattr(self, "_pending_title", "").strip()
        lines = [
            "# Meeting Transcript\n",
            f"**Date:** {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}\n\n",
        ]
        if title:
            lines.insert(1, f"**Title:** {title}\n\n")
        if dur_str:
            lines.append(f"**Duration:** {dur_str}\n\n")
        if summary_text:
            lines += ["## Summary\n\n", summary_text + "\n\n", "---\n\n"]
        lines.append("## Full Transcript\n\n")
        for seg in self._transcript_segments:
            ts_str = datetime.datetime.fromtimestamp(seg.timestamp).strftime("%H:%M:%S")
            ico    = "🎤" if seg.source == "microphone" else "🖥"
            lines.append(f"**[{ts_str}]** {ico} {seg.text}\n\n")

        fname.write_text("".join(lines), encoding="utf-8")
        self._current_save_path = fname
        _invalidate_cache(fname)

        if auto:
            self._set_status(f"Auto-saved → {fname.name}")
            self.after(300, self._refresh_meetings_list)
        else:
            self._set_status(f"Saved → {fname.name}")
            self._refresh_meetings_list()
        return fname

    def _write_summary_to_file(self, path: Path, summary: str):
        """Insert or replace the ## Summary section in a transcript file."""
        text = path.read_text(encoding="utf-8")
        pattern     = r'(## Summary\n\n).*?(\n\n---\n|\n\n## Full Transcript)'
        replacement = r'\g<1>' + summary.replace("\\", "\\\\") + r'\2'
        new_text    = re.sub(pattern, replacement, text, flags=re.DOTALL)
        if new_text == text:
            # No existing summary section — insert one before Full Transcript
            new_text = text.replace(
                "## Full Transcript",
                f"## Summary\n\n{summary}\n\n---\n\n## Full Transcript",
            )
        path.write_text(new_text, encoding="utf-8")
        _invalidate_cache(path)
        self.after(400, self._refresh_meetings_list)

    def _save_past_summary(self):
        """Overwrite the ## Summary section in the currently viewed file."""
        if not self._viewing_path:
            return
        new_summary = self._summary_box.get("1.0", "end").strip()
        try:
            self._write_summary_to_file(self._viewing_path, new_summary)
            self._set_status("✓ Summary saved to file")
        except Exception as e:
            self._set_status(f"Save error: {e}")

    def _clear_transcript(self):
        self._live_summary = ""
        self._session_duration = 0
        self._transcript_segments.clear()
        self._transcript_box.configure(state="normal")
        self._transcript_box.delete("1.0", "end")
        self._transcript_box.configure(state="disabled")
        self._set_transcript_placeholder()
        self._summary_box.configure(state="normal")
        self._summary_box.delete("1.0", "end")
        self._summary_box.configure(state="disabled")
        self._set_summary_placeholder()
        self._timer_lbl.configure(text="00:00", text_color=C["text_dim"])
        self._set_status("Cleared")

    def _delete_meeting(self):
        """Delete the currently viewed past meeting after confirmation."""
        if not self._viewing_path or not self._viewing_path.exists():
            return
        fname = self._viewing_path.name
        dialog = ctk.CTkInputDialog(
            text=f'Type DELETE to confirm removing\n"{fname}"',
            title="Delete Meeting",
        )
        result = dialog.get_input()
        if result and result.strip().upper() == "DELETE":
            try:
                self._viewing_path.unlink()
                self._select_new_meeting()
                self._refresh_meetings_list()
                self._set_status(f"Deleted {fname}")
            except Exception as e:
                self._set_status(f"Delete failed: {e}")

    def _export_meeting(self):
        """Export the current meeting (transcript + summary) as formatted HTML."""
        import subprocess, tempfile

        if self._current_view == "new":
            transcript = self._get_transcript_text()
            summary    = self._get_summary_text()
            title      = datetime.datetime.now().strftime("Meeting — %B %d, %Y %H:%M")
        else:
            if not self._viewing_path:
                self._set_status("No meeting to export.")
                return
            info       = _parse_file_cached(self._viewing_path)
            transcript = info["transcript_text"]
            summary    = info["summary_text"]
            title      = f"Meeting — {info['time']}"

        if not transcript and not summary:
            self._set_status("Nothing to export yet.")
            return

        # Convert markdown-ish summary to basic HTML
        def md_to_html(md: str) -> str:
            lines, out = md.splitlines(), []
            for ln in lines:
                if ln.startswith("### "):
                    out.append(f"<h3>{ln[4:]}</h3>")
                elif ln.startswith("## "):
                    out.append(f"<h2>{ln[3:]}</h2>")
                elif ln.startswith("# "):
                    out.append(f"<h1>{ln[2:]}</h1>")
                elif ln.startswith("- ") or ln.startswith("* "):
                    out.append(f"<li>{ln[2:]}</li>")
                elif ln.strip() == "---":
                    out.append("<hr>")
                elif ln.strip() == "":
                    out.append("<br>")
                else:
                    # bold **text**
                    import re as _re
                    ln = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', ln)
                    out.append(f"<p>{ln}</p>")
            return "\n".join(out)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 820px; margin: 40px auto;
          padding: 0 24px; color: #1e293b; line-height: 1.6; }}
  h1   {{ font-size: 1.6em; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }}
  h2   {{ font-size: 1.2em; color: #3b82f6; margin-top: 2em; }}
  h3   {{ font-size: 1em; color: #475569; margin-top: 1.4em; }}
  li   {{ margin: 4px 0; }}
  hr   {{ border: none; border-top: 1px solid #e2e8f0; margin: 1.5em 0; }}
  pre  {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px;
          padding: 16px; font-size: 0.85em; white-space: pre-wrap; }}
  .meta {{ color: #64748b; font-size: 0.9em; margin-bottom: 2em; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="meta">Exported {datetime.datetime.now().strftime("%B %d, %Y at %H:%M")} · Local Transcription App</p>
"""
        if summary:
            html += f"<h2>Summary</h2>\n{md_to_html(summary)}\n<hr>\n"
        if transcript:
            html += "<h2>Full Transcript</h2>\n<pre>"
            html += transcript.replace("&", "&amp;").replace("<", "&lt;")
            html += "</pre>\n"
        html += "</body></html>"

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, prefix=f"meeting_export_{now}_"
        ) as f:
            f.write(html)
            tmp = f.name
        subprocess.Popen(["open", tmp])
        self._set_status("Export opened in browser — use File → Print → Save as PDF")

    def _save_hf_token(self):
        tok = self._hf_token_var.get().strip()
        if tok:
            Diarizer.save_hf_token(tok)
            self._set_status("✓ HuggingFace token saved — diarization enabled")

    def _save_title(self):
        """Persist the edited meeting title back into the currently viewed file."""
        if not self._viewing_path or not self._viewing_path.exists():
            return
        new_title = self._title_var.get().strip()
        try:
            text = self._viewing_path.read_text(encoding="utf-8")
            if "**Title:**" in text:
                text = re.sub(r'\*\*Title:\*\*.*\n', f"**Title:** {new_title}\n", text)
            else:
                text = text.replace("**Date:**", f"**Title:** {new_title}\n**Date:**", 1)
            self._viewing_path.write_text(text, encoding="utf-8")
            _invalidate_cache(self._viewing_path)
            self.after(300, self._refresh_meetings_list)
        except Exception as e:
            self._set_status(f"Title save error: {e}")

    def _on_search_change(self, event=None):
        """Filter the meetings list as the user types."""
        self._search_term = self._search_entry.get().strip().lower()
        self._refresh_meetings_list()

    def _focus_search(self, event=None):
        self._search_entry.focus_set()
        return "break"

    # ═══════════════════════════════════════════════════════════════════════════
    #  Placeholders
    # ═══════════════════════════════════════════════════════════════════════════

    def _set_transcript_placeholder(self):
        self._transcript_box.configure(state="normal")
        self._transcript_box.delete("1.0", "end")
        self._set_transcript_placeholder_inline()
        self._transcript_box.configure(state="disabled")

    def _set_transcript_placeholder_inline(self):
        self._transcript_box.insert("1.0",
            "Your transcript will appear here as you speak.\n\n"
            "Select an audio source on the left, then press ⏺ Start Recording.")

    def _set_summary_placeholder(self):
        self._summary_box.configure(state="normal")
        self._summary_box.delete("1.0", "end")
        self._summary_box.insert("1.0",
            "Click ✨ Summarize after your meeting to get an AI-written summary "
            "and action items — all processed locally via Ollama.\n\n"
            "Once generated, use the Refine box below to adjust verbosity or focus.")
        self._summary_box.configure(state="disabled")

    # ═══════════════════════════════════════════════════════════════════════════
    #  UI Helpers
    # ═══════════════════════════════════════════════════════════════════════════

    def _set_status(self, msg: str):
        self._ui_queue.put(("status", msg))

    def _on_source_change(self, value: str):
        color = {"Mic": C["success"], "System": C["accent"], "Both": C["warning"]}
        self._level_bar.configure(
            progress_color=color.get(value, C["success"]))

    # ═══════════════════════════════════════════════════════════════════════════
    #  UI Queue Polling
    # ═══════════════════════════════════════════════════════════════════════════

    def _poll_ui_queue(self):
        try:
            while True:
                event, data = self._ui_queue.get_nowait()
                if event == "segment":
                    self._append_segment(data)
                elif event == "status":
                    self._status_var.set(data)
                elif event == "level":
                    self._level_value = self._level_value * 0.6 + float(data) * 0.4
                    self._level_bar.set(min(1.0, self._level_value * 20))
                elif event == "backend_label":
                    self._backend_lbl.configure(text=data)
                elif event == "model_lbl":
                    self._model_lbl.configure(text=data)
                elif event == "summary_token":
                    self._append_summary_token(data)
                elif event == "summary_done":
                    self._on_summary_done()
                elif event == "summary_error":
                    self._summary_box.configure(state="normal")
                    self._summary_box.insert("end", f"\n\n⚠ {data}")
                    self._summary_box.configure(state="disabled")
                    self._on_summary_done()
                elif event == "diarization_done":
                    segs, session_start = data
                    self._apply_diarization(segs, session_start)
                elif event == "calendar_ready":
                    self._calendar_events = data
                    self._refresh_meetings_list()
        except Exception:
            pass
        finally:
            self.after(50, self._poll_ui_queue)

    # ═══════════════════════════════════════════════════════════════════════════
    #  Keyboard Shortcuts
    # ═══════════════════════════════════════════════════════════════════════════

    def _bind_shortcuts(self):
        self.bind("<Command-r>", lambda e: self._toggle_recording() or "break")
        self.bind("<Command-s>", lambda e: self._save_transcript()  or "break")
        self.bind("<Command-Return>", lambda e: (
            self._run_summary() if self._current_view == "new"
            else self._run_summary_past()) or "break")
        self.bind("<Command-k>", lambda e: self._clear_transcript()  or "break")
        self.bind("<Command-f>", self._focus_search)
        self.bind("<Command-p>", lambda e: self._toggle_pause()      or "break")
        self.bind("<Command-e>", lambda e: self._export_meeting()    or "break")

    # ═══════════════════════════════════════════════════════════════════════════
    #  Shutdown
    # ═══════════════════════════════════════════════════════════════════════════

    def on_closing(self):
        if self._recording:
            self._stop_recording()
        if self._transcriber:
            self._transcriber.stop()
        self.destroy()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    app = TranscriptionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
