"""
summarizer.py
─────────────
Post-meeting AI summarization via Ollama (100% local).
Extracts a concise summary and action items from the transcript.

Requires Ollama to be running: https://ollama.com
Default model: qwen2.5:32b (high quality, fits comfortably in 48 GB RAM)
"""

import threading
from typing import Callable, Optional


OLLAMA_AVAILABLE = False
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    pass


DEFAULT_MODEL = "qwen2.5:32b"

SUMMARY_PROMPT = """\
You are an expert meeting facilitator producing a professional meeting summary. \
Analyse the transcript below and produce a structured summary following best practices.

---
TRANSCRIPT:
{transcript}
---

Produce the following sections in clean Markdown. Omit any section where there is genuinely \
no relevant content (e.g. no decisions were made). Be concise and specific — avoid filler phrases.

### Executive Summary
2–3 sentences describing the purpose of the meeting and the most important outcome.

### Attendees & Roles
List any names or roles mentioned. If none are identifiable, write "Not specified."

### Key Decisions
Bullet list of decisions that were made and agreed upon. Each bullet should be a clear, \
actionable statement (not vague). If no decisions were made, omit this section.

### Action Items
Table or bullet list. For each action item include:
- **What**: the specific task
- **Who**: the owner (write "Unassigned" if not mentioned)
- **By when**: deadline if mentioned, otherwise "No date set"

### Discussion Topics
Brief bullets covering the main topics discussed, in the order they arose.

### Risks & Blockers
Any concerns, blockers, or open questions raised that are not yet resolved.

### Next Steps
What happens after this meeting — follow-up meetings, reviews, deliverables, etc.

---
Respond only with the Markdown. Do not add preamble or commentary.
"""

REFINE_PROMPT = """\
You are an expert meeting facilitator. The user wants to refine an existing meeting summary.

ORIGINAL TRANSCRIPT:
{transcript}

CURRENT SUMMARY:
{current_summary}

USER REQUEST: {instruction}

Produce an updated summary in the same structured Markdown format (Executive Summary, \
Attendees & Roles, Key Decisions, Action Items, Discussion Topics, Risks & Blockers, Next Steps). \
Be directly responsive to the request — if asked for more detail expand that section; \
if asked to shorten, condense. Omit sections that have no content.
"""


class Summarizer:
    """
    Sends a transcript to a local Ollama model and streams the summary
    back via a callback.

    Usage:
        s = Summarizer(on_token=my_callback, on_done=done_callback)
        s.summarize(transcript_text)
        s.refine(transcript_text, current_summary, instruction)
    """

    def __init__(
        self,
        on_token: Callable[[str], None],
        on_done: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        model: str = DEFAULT_MODEL,
    ):
        self.on_token = on_token
        self.on_done = on_done or (lambda _: None)
        self.on_error = on_error or (lambda _: None)
        self.model = model
        self._thread: Optional[threading.Thread] = None

    def is_available(self) -> bool:
        if not OLLAMA_AVAILABLE:
            return False
        try:
            ollama.list()
            return True
        except Exception:
            return False

    def preload(self):
        """
        Warm up the model in the background so it's already loaded in RAM
        by the time the meeting ends and summarization is requested.
        Sends a single-token request — fast, silent, no side effects.
        """
        if not self.is_available():
            return
        threading.Thread(target=self._warmup, daemon=True).start()

    def _warmup(self):
        try:
            ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "ready"}],
                stream=False,
                options={"num_predict": 1},
            )
        except Exception:
            pass

    def list_models(self) -> list[str]:
        if not OLLAMA_AVAILABLE:
            return []
        try:
            resp = ollama.list()
            return [m.model for m in resp.models]
        except Exception:
            return []

    def summarize(self, transcript: str):
        """Start summarization in a background thread (non-blocking)."""
        if not transcript.strip():
            self.on_error("Transcript is empty — nothing to summarize.")
            return

        if not self.is_available():
            self.on_error(
                "Ollama is not running or not installed.\n\n"
                "Install from https://ollama.com, then run:\n"
                "  ollama pull qwen2.5:32b"
            )
            return

        self._thread = threading.Thread(
            target=self._run_summary,
            args=(transcript,),
            daemon=True,
        )
        self._thread.start()

    def refine(self, transcript: str, current_summary: str, instruction: str):
        """Refine an existing summary based on user instruction (non-blocking)."""
        if not self.is_available():
            self.on_error("Ollama is not running. Make sure Ollama.app is open.")
            return

        threading.Thread(
            target=self._run_refine,
            args=(transcript, current_summary, instruction),
            daemon=True,
        ).start()

    def _run_summary(self, transcript: str):
        prompt = SUMMARY_PROMPT.format(transcript=transcript)
        self._stream(prompt)

    def _run_refine(self, transcript: str, current_summary: str, instruction: str):
        prompt = REFINE_PROMPT.format(
            transcript=transcript,
            current_summary=current_summary,
            instruction=instruction,
        )
        self._stream(prompt)

    def _stream(self, prompt: str):
        full_response = []
        try:
            stream = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                token = chunk["message"]["content"]
                full_response.append(token)
                self.on_token(token)

            self.on_done("".join(full_response))

        except Exception as e:
            # Try fallback extractive summary
            self.on_error(f"Ollama error: {e}\n\nFalling back to extractive summary…")
            fallback = self._extractive_summary(prompt)
            self.on_token(fallback)
            self.on_done(fallback)

    @staticmethod
    def _extractive_summary(transcript: str) -> str:
        """
        Simple extractive fallback: returns the first ~200 words as a summary
        when Ollama is unavailable.
        """
        lines = [l.strip() for l in transcript.splitlines() if l.strip()]
        words = []
        for line in lines:
            words.extend(line.split())
            if len(words) >= 200:
                break
        excerpt = " ".join(words[:200])
        return (
            "**Summary (extractive fallback — Ollama not available)**\n\n"
            f"{excerpt}…\n\n"
            "_Install Ollama for AI-generated summaries and action items._"
        )
