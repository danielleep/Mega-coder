"""Safe local screen-analysis foundations for Mega Coder option 3."""

from dataclasses import dataclass, field
from importlib import import_module
from math import isfinite
from numbers import Real
import re


SCREEN_CAPTURE_INTERVAL_SECONDS = 3.0
STABLE_FRAME_COUNT = 2
TIP_REQUEST_COOLDOWN_SECONDS = 5.0
MAX_SCREEN_TEXT_CHARS = 12_000
MIN_OCR_CONFIDENCE = 0.60
SCREEN_CAPTURE_REGION = None
SCREEN_MONITOR_INDEX = None

_REGION_KEYS = ("top", "left", "width", "height")
_CODE_KEYWORD = re.compile(
    r"\b(?:def|class|import|from|return|if|elif|else|for|while|try|except|"
    r"with|function|const|let|var)\b"
)
_CODE_STATEMENT = re.compile(
    r"^\s*(?:def|class|import|from|return|if|elif|else|for|while|try|except|"
    r"with|function|const|let|var)\b(?:\s+.+|[(:{])"
)
_ASSIGNMENT = re.compile(r"(?<![=!<>])(?:[+\-*/%]?=)(?!=)")
_FUNCTION_CALL = re.compile(r"\b[A-Za-z_]\w*\s*\(")
_CODE_PUNCTUATION = re.compile(r"[()\[\]{}:;]|==|!=|<=|>=|=>|[=+\-*/%<>]")
_STATUS_BAR = re.compile(
    r"^(?:ln\s+\d+,\s*col\s+\d+|spaces:\s*\d+|utf-8\b|"
    r"(?:crlf|lf)\b|go live\b)"
)
_IGNORED_SCREEN_LINES = frozenset(
    {
        "file edit selection view go run terminal help",
        "explorer",
        "search",
        "source control",
        "run and debug",
        "extensions",
        "outline",
        "timeline",
        "problems",
        "output",
        "debug console",
        "terminal",
        "ports",
        "i’m mega coder. what would you like me to do today?",
        "i'm mega coder. what would you like me to do today?",
        "1. develop a python program.",
        "2. fix/change something in a github repository.",
        "3. look at my screen and give me realtime coding tips.",
        "not implemented yet",
        "goodbye.",
        "perfect. show me your screen and i will be giving you tips on how to "
        "improve the code i see",
    }
)
_TIP_LABEL_PREFIXES = (
    "coding tip:",
    "coding tips:",
    "mega coder tip:",
    "mega coder tips:",
    "tip:",
    "tips:",
)
_IGNORED_SCREEN_PREFIXES = (
    "describe me which python program",
    "give me the full url of a public github repository",
    "tell me what you want me to fix/change/explain",
    "please choose 1, 2, or 3",
    "repository ingested successfully",
    "generated program",
    "repair attempt ",
    "lint repair attempt ",
    "requesting optimization",
    "requesting a more efficient version",
    "code running time optimized!",
    "amazing. no lint errors/warnings",
    "there are still lint errors/warnings",
) + _TIP_LABEL_PREFIXES


class ScreenCaptureError(RuntimeError):
    """Report capture configuration or backend failures without screen data."""


class OCRProcessingError(RuntimeError):
    """Report local OCR failures without revealing recognized screen text."""


@dataclass(frozen=True)
class OCRLine:
    """Store one recognized line and its optional confidence and position."""

    text: str
    confidence: float | None
    position: tuple[tuple[float, float], ...] | None


class ScreenTextTooLargeError(ValueError):
    """Report screen text that cannot be processed without truncation."""


def validate_capture_region(region):
    """Return a validated MSS region containing integer pixel coordinates."""
    if not isinstance(region, dict):
        raise ScreenCaptureError("The screen capture region must be a dictionary.")

    for key in _REGION_KEYS:
        if (
            key not in region
            or not isinstance(region[key], int)
            or isinstance(region[key], bool)
        ):
            raise ScreenCaptureError(
                "The screen capture region requires integer coordinates and sizes."
            )
    if region["width"] <= 0 or region["height"] <= 0:
        raise ScreenCaptureError(
            "The screen capture region width and height must be positive."
        )

    return {key: region[key] for key in _REGION_KEYS}


def _resolve_monitor(capture, monitor_index):
    """Return one available positive physical-monitor target."""
    if (
        not isinstance(monitor_index, int)
        or isinstance(monitor_index, bool)
        or monitor_index <= 0
    ):
        raise ScreenCaptureError(
            "The screen monitor index must identify a positive physical monitor."
        )
    try:
        monitors = capture.monitors
        if not isinstance(monitors, (list, tuple)):
            raise TypeError
        target = monitors[monitor_index]
    # Capture backends may raise platform-specific exception types.
    except Exception as error:  # pylint: disable=broad-exception-caught
        raise ScreenCaptureError(
            "The selected physical monitor is not available."
        ) from error
    return validate_capture_region(target)


def resolve_capture_target(capture, capture_region=None, monitor_index=None):
    """Resolve region, physical monitor, then the MSS primary-monitor property.

    MSS detects the true primary monitor on supported Windows and Linux systems.
    On macOS, ``capture.primary_monitor`` falls back to the first physical
    monitor, so an editor-only region is preferred for manual use there.
    """
    effective_region = (
        capture_region if capture_region is not None else SCREEN_CAPTURE_REGION
    )
    effective_monitor = (
        monitor_index if monitor_index is not None else SCREEN_MONITOR_INDEX
    )

    if effective_region is not None:
        return validate_capture_region(effective_region)
    if effective_monitor is not None:
        return _resolve_monitor(capture, effective_monitor)

    try:
        primary_monitor = capture.primary_monitor
    # Capture backends may raise platform-specific exception types.
    except Exception as error:  # pylint: disable=broad-exception-caught
        raise ScreenCaptureError(
            "The primary screen monitor is not available."
        ) from error
    return validate_capture_region(primary_monitor)


def _load_numpy():
    """Load NumPy only when the disconnected screen-capture path is used."""
    try:
        return import_module("numpy")
    except ImportError as error:
        raise ScreenCaptureError(
            "NumPy is required for in-memory screen capture."
        ) from error


def _convert_bgra_screenshot(screenshot):
    """Convert one MSS BGRA screenshot to contiguous BGR uint8 data."""
    numpy = _load_numpy()
    try:
        bgra = numpy.asarray(screenshot, dtype=numpy.uint8)
        if bgra.ndim != 3 or bgra.shape[0] <= 0 or bgra.shape[1] <= 0:
            raise ValueError
        if bgra.shape[2] != 4:
            raise ValueError
        bgr = bgra[:, :, :3].copy()
        if (
            bgr.shape != (bgra.shape[0], bgra.shape[1], 3)
            or bgr.dtype != numpy.uint8
            or not bgr.flags.c_contiguous
        ):
            raise ValueError
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ScreenCaptureError(
            "The screen capture backend returned malformed image data."
        ) from error
    return bgr


def capture_screen_frame(capture, capture_region=None, monitor_index=None):
    """Capture one resolved target in memory without saving or retaining it."""
    target = resolve_capture_target(capture, capture_region, monitor_index)
    try:
        screenshot = capture.grab(target)
    # Capture backends may raise platform-specific exception types.
    except Exception as error:  # pylint: disable=broad-exception-caught
        raise ScreenCaptureError("The screen capture failed.") from error
    return _convert_bgra_screenshot(screenshot)


def create_ocr_engine():
    """Lazily create one unified RapidOCR engine using local OpenVINO."""
    try:
        rapidocr = import_module("rapidocr")
        openvino = rapidocr.EngineType.OPENVINO
        return rapidocr.RapidOCR(
            params={
                "Det.engine_type": openvino,
                "Cls.engine_type": openvino,
                "Rec.engine_type": openvino,
            }
        )
    # RapidOCR initialization may raise dependency- or backend-specific errors.
    except Exception as error:  # pylint: disable=broad-exception-caught
        raise OCRProcessingError("Local OCR could not be initialized.") from error


def _parse_rapidocr_output(result):  # pylint: disable=too-many-branches
    """Adapt the installed RapidOCR 3.9 output fields to ordered OCR lines."""
    try:
        texts = result.txts
        scores = result.scores
        boxes = result.boxes

        if texts is None:
            if (scores is not None and len(scores) != 0) or (
                boxes is not None and len(boxes) != 0
            ):
                raise ValueError
            return ()
        if not isinstance(texts, (list, tuple)):
            raise TypeError
        if not texts:
            if (scores is not None and len(scores) != 0) or (
                boxes is not None and len(boxes) != 0
            ):
                raise ValueError
            return ()

        line_count = len(texts)
        if scores is not None and len(scores) != line_count:
            raise ValueError
        if boxes is not None and len(boxes) != line_count:
            raise ValueError

        lines = []
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError

            confidence = None
            if scores is not None:
                score = scores[index]
                if not isinstance(score, Real) or isinstance(score, bool):
                    raise TypeError
                confidence = float(score)

            position = None
            if boxes is not None:
                box = boxes[index]
                if hasattr(box, "tolist"):
                    box = box.tolist()
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    raise ValueError
                points = []
                for point in box:
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        raise ValueError
                    if any(
                        not isinstance(coordinate, Real)
                        or isinstance(coordinate, bool)
                        for coordinate in point
                    ):
                        raise TypeError
                    points.append(tuple(float(value) for value in point))
                position = tuple(points)

            lines.append(OCRLine(text, confidence, position))
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise OCRProcessingError(
            "RapidOCR returned malformed recognition data."
        ) from error
    return tuple(lines)


def extract_ocr_lines(frame, engine):
    """Recognize one verified BGR frame with a reusable OCR engine."""
    try:
        result = engine(frame)
    # Recognition engines may raise backend-specific exception types.
    except Exception as error:  # pylint: disable=broad-exception-caught
        raise OCRProcessingError("Local OCR could not process the screen frame.") from error
    return _parse_rapidocr_output(result)


def _ensure_screen_text_size(text):
    """Reject oversized text rather than silently truncating it."""
    if not isinstance(text, str):
        raise TypeError("Screen text must be a string.")
    if len(text) > MAX_SCREEN_TEXT_CHARS:
        raise ScreenTextTooLargeError(
            "The detected screen code is too large to analyze safely."
        )


def normalize_code_candidate(text):
    """Normalize only line endings, trailing whitespace, and outer blanks."""
    _ensure_screen_text_size(text)
    normalized_lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized_lines = [line.rstrip() for line in normalized_lines]
    while normalized_lines and not normalized_lines[0]:
        normalized_lines.pop(0)
    while normalized_lines and not normalized_lines[-1]:
        normalized_lines.pop()
    return "\n".join(normalized_lines)


def _code_signals(text):
    """Return independent deterministic signals found in candidate text."""
    lines = text.splitlines()
    return (
        bool(_CODE_KEYWORD.search(text)),
        any(_CODE_STATEMENT.search(line) for line in lines),
        bool(_ASSIGNMENT.search(text)),
        bool(_FUNCTION_CALL.search(text)),
        any(line.startswith((" ", "\t")) for line in lines if line),
        bool(_CODE_PUNCTUATION.search(text)),
    )


def _line_code_score(text):
    """Count local code-like signals for deterministic block selection."""
    return sum(_code_signals(text))


def looks_like_code(text):
    """Require multiple signals, including one strong code-like structure."""
    _ensure_screen_text_size(text)
    if not text.strip():
        return False
    signals = _code_signals(text)
    has_strong_signal = any(signals[:4])
    structured_line_count = sum(
        _line_code_score(line) >= 2 for line in text.splitlines() if line.strip()
    )
    return has_strong_signal and (
        sum(signals) >= 2 or structured_line_count >= 2
    )


def _is_ignored_screen_line(text):
    """Identify obvious editor chrome and Mega Coder console messages."""
    normalized = " ".join(text.strip().casefold().split())
    return (
        normalized in _IGNORED_SCREEN_LINES
        or normalized.startswith(_IGNORED_SCREEN_PREFIXES)
        or bool(_STATUS_BAR.match(normalized))
    )


def _is_tip_label(text):
    """Identify the start of previously printed coding-tip output."""
    normalized = " ".join(text.strip().casefold().split())
    return normalized.startswith(_TIP_LABEL_PREFIXES)


def extract_code_candidate(lines):
    """Select the strongest contiguous code-like block from untrusted OCR lines."""
    blocks = []
    current_block = []
    for line in lines:
        if not isinstance(line, OCRLine):
            raise TypeError("OCR lines must use the OCRLine structure.")
        if (
            line.confidence is not None
            and line.confidence < MIN_OCR_CONFIDENCE
        ):
            continue
        if _is_tip_label(line.text):
            blocks.append(current_block)
            current_block = []
            break
        if _is_ignored_screen_line(line.text):
            blocks.append(current_block)
            current_block = []
            continue

        line_score = _line_code_score(line.text)
        if line_score:
            current_block.append((line.text, line_score))
        elif current_block:
            blocks.append(current_block)
            current_block = []
    blocks.append(current_block)

    best_candidate = ""
    best_rank = (0, 0)
    for block in blocks:
        candidate = "\n".join(text for text, _score in block)
        if not looks_like_code(candidate):
            continue
        rank = (sum(score for _text, score in block), len(block))
        if rank > best_rank:
            best_candidate = candidate
            best_rank = rank
    return best_candidate


def _validated_timestamp(timestamp):
    """Return a finite injected monotonic timestamp."""
    if (
        not isinstance(timestamp, Real)
        or isinstance(timestamp, bool)
        or not isfinite(timestamp)
    ):
        raise ValueError("The tracker timestamp must be a finite number.")
    return float(timestamp)


@dataclass
class StableCodeTracker:  # pylint: disable=too-many-instance-attributes
    """Track exact two-frame stability, attempts, successes, and cooldown."""

    stable_frame_count: int = STABLE_FRAME_COUNT
    cooldown_seconds: float = TIP_REQUEST_COOLDOWN_SECONDS
    current_normalized_candidate: str | None = field(default=None, init=False)
    current_candidate_frame_count: int = field(default=0, init=False)
    last_attempted_normalized_text: str | None = field(default=None, init=False)
    last_successfully_analyzed_normalized_text: str | None = field(
        default=None, init=False
    )
    last_api_attempt_time: float | None = field(default=None, init=False)
    _pending_normalized_candidate: str | None = field(default=None, init=False)
    _ready_normalized_candidate: str | None = field(default=None, init=False)

    def __post_init__(self):
        """Validate injected deterministic stability and cooldown settings."""
        if (
            not isinstance(self.stable_frame_count, int)
            or isinstance(self.stable_frame_count, bool)
            or self.stable_frame_count <= 0
        ):
            raise ValueError("The stable frame count must be a positive integer.")
        if (
            not isinstance(self.cooldown_seconds, Real)
            or isinstance(self.cooldown_seconds, bool)
            or not isfinite(self.cooldown_seconds)
            or self.cooldown_seconds < 0
        ):
            raise ValueError("The tip-request cooldown must be non-negative.")
        self.cooldown_seconds = float(self.cooldown_seconds)

    def _reset_current_candidate(self):
        """Reset only observation state while retaining session history."""
        self.current_normalized_candidate = None
        self.current_candidate_frame_count = 0
        self._pending_normalized_candidate = None
        self._ready_normalized_candidate = None

    def _cooldown_complete(self, timestamp):
        """Return whether a different candidate may be attempted now."""
        return self.last_api_attempt_time is None or (
            timestamp - self.last_api_attempt_time >= self.cooldown_seconds
        )

    def observe_candidate(self, candidate, timestamp):
        """Observe one candidate and return it once when stable and eligible."""
        now = _validated_timestamp(timestamp)
        normalized = normalize_code_candidate(candidate)
        if not normalized or not looks_like_code(normalized):
            self._reset_current_candidate()
            return None

        if normalized != self.current_normalized_candidate:
            self.current_normalized_candidate = normalized
            self.current_candidate_frame_count = 1
            self._pending_normalized_candidate = None
            self._ready_normalized_candidate = None
            return None

        self.current_candidate_frame_count += 1
        if self.current_candidate_frame_count == self.stable_frame_count:
            self._pending_normalized_candidate = normalized

        pending = self._pending_normalized_candidate
        if pending is None or self._ready_normalized_candidate == pending:
            return None
        if pending in {
            self.last_attempted_normalized_text,
            self.last_successfully_analyzed_normalized_text,
        }:
            self._pending_normalized_candidate = None
            return None
        if not self._cooldown_complete(now):
            return None

        self._pending_normalized_candidate = None
        self._ready_normalized_candidate = pending
        return pending

    def mark_attempted(self, candidate, timestamp):
        """Record one eligible candidate immediately before a future request."""
        now = _validated_timestamp(timestamp)
        normalized = normalize_code_candidate(candidate)
        if normalized != self._ready_normalized_candidate:
            return False
        if normalized in {
            self.last_attempted_normalized_text,
            self.last_successfully_analyzed_normalized_text,
        } or not self._cooldown_complete(now):
            self._ready_normalized_candidate = None
            return False

        self.last_attempted_normalized_text = normalized
        self.last_api_attempt_time = now
        self._ready_normalized_candidate = None
        return True

    def mark_successful(self, candidate):
        """Record a successful future response for the last attempted text."""
        normalized = normalize_code_candidate(candidate)
        if normalized != self.last_attempted_normalized_text:
            raise ValueError("Only the last attempted candidate can succeed.")
        self.last_successfully_analyzed_normalized_text = normalized
