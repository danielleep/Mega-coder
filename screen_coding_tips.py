"""Safe in-memory screen capture foundations for Mega Coder option 3."""

from importlib import import_module


SCREEN_CAPTURE_INTERVAL_SECONDS = 3.0
STABLE_FRAME_COUNT = 2
SCREEN_CAPTURE_REGION = None
SCREEN_MONITOR_INDEX = None

_REGION_KEYS = ("top", "left", "width", "height")


class ScreenCaptureError(RuntimeError):
    """Report capture configuration or backend failures without screen data."""


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
