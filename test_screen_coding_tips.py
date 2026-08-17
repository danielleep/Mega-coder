"""Mocked tests for the safe Option 3 screen-capture foundation."""

from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import screen_coding_tips as screen_tips


VIRTUAL_MONITOR = {"top": -20, "left": -40, "width": 400, "height": 200}
FIRST_MONITOR = {"top": 0, "left": 0, "width": 200, "height": 100}
SECOND_MONITOR = {"top": 10, "left": 200, "width": 200, "height": 100}
EDITOR_REGION = {"top": -5, "left": -10, "width": 2, "height": 1}
BGRA_PIXELS = [[[10, 20, 30, 99], [200, 150, 100, 77]]]
FIRST_TEXT_BOX = ((0, 0), (20, 0), (20, 10), (0, 10))
SECOND_TEXT_BOX = ((0, 12), (30, 12), (30, 22), (0, 22))
_DEFAULT_SCREENSHOT = object()


class FakeArray:
    """Implement the small array surface used by the production conversion."""

    def __init__(self, data, dtype, contiguous=True):
        self.data = data
        self.dtype = dtype
        self.flags = SimpleNamespace(c_contiguous=contiguous)
        if not isinstance(data, list) or not data:
            self.ndim = 1
            self.shape = (0,)
        elif not isinstance(data[0], list) or not data[0]:
            self.ndim = 2
            self.shape = (len(data), 0)
        else:
            self.ndim = 3
            self.shape = (len(data), len(data[0]), len(data[0][0]))

    def __getitem__(self, item):
        """Support the production all-rows, all-columns, channel slice."""
        row_slice, column_slice, channel_slice = item
        if row_slice != slice(None) or column_slice != slice(None):
            raise IndexError
        data = [
            [pixel[channel_slice] for pixel in row]
            for row in self.data
        ]
        return FakeArray(data, self.dtype, contiguous=False)

    def copy(self):
        """Return a distinct contiguous copy like a NumPy array copy."""
        copied = [[list(pixel) for pixel in row] for row in self.data]
        return FakeArray(copied, self.dtype, contiguous=True)

    def tolist(self):
        """Return the nested pixel data for readable assertions."""
        return self.data


class FakeNumpy:  # pylint: disable=too-few-public-methods
    """Provide deterministic array conversion without requiring NumPy."""

    uint8 = "uint8"

    @staticmethod
    def asarray(value, dtype):
        """Create a fake array or reproduce a malformed-conversion failure."""
        if value == "conversion failure":
            raise ValueError("private pixel detail")
        return FakeArray(value, dtype)


def fake_capture(screenshot=_DEFAULT_SCREENSHOT):
    """Return an MSS-shaped mock without opening the real capture backend."""
    if screenshot is _DEFAULT_SCREENSHOT:
        screenshot = BGRA_PIXELS
    return SimpleNamespace(
        monitors=[VIRTUAL_MONITOR, FIRST_MONITOR, SECOND_MONITOR],
        primary_monitor=FIRST_MONITOR,
        grab=Mock(return_value=screenshot),
    )


def synthetic_frame():
    """Create harmless contiguous BGR-shaped data without capturing a screen."""
    numpy = import_module("numpy")
    return numpy.zeros((2, 3, 3), dtype=numpy.uint8)


def ocr_result(texts, scores, boxes):
    """Create a mocked RapidOCR 3.9 result object."""
    return SimpleNamespace(txts=texts, scores=scores, boxes=boxes)


class CaptureConfigurationTests(unittest.TestCase):
    """Verify validation and deterministic target precedence."""

    def test_constants_match_the_fixed_plan_values(self):
        """Capture timing and stabilization retain their exact defaults."""
        self.assertEqual(screen_tips.SCREEN_CAPTURE_INTERVAL_SECONDS, 3.0)
        self.assertEqual(screen_tips.STABLE_FRAME_COUNT, 2)
        self.assertIsNone(screen_tips.SCREEN_CAPTURE_REGION)
        self.assertIsNone(screen_tips.SCREEN_MONITOR_INDEX)

    def test_negative_region_coordinates_are_valid(self):
        """Virtual desktop coordinates may extend left or above the origin."""
        self.assertEqual(
            screen_tips.validate_capture_region(EDITOR_REGION), EDITOR_REGION
        )

    def test_invalid_regions_are_rejected(self):
        """Missing, Boolean, non-integer, and non-positive region values fail."""
        invalid_regions = (
            None,
            {"top": 0, "left": 0, "width": 1},
            {"top": False, "left": 0, "width": 1, "height": 1},
            {"top": 0.0, "left": 0, "width": 1, "height": 1},
            {"top": 0, "left": "0", "width": 1, "height": 1},
            {"top": 0, "left": 0, "width": 0, "height": 1},
            {"top": 0, "left": 0, "width": 1, "height": -1},
        )
        for region in invalid_regions:
            with self.subTest(region=region):
                with self.assertRaises(screen_tips.ScreenCaptureError):
                    screen_tips.validate_capture_region(region)

    def test_region_precedence_uses_injected_then_configured_region(self):
        """An effective region wins before any effective monitor setting."""
        capture = fake_capture()
        configured_region = {"top": 1, "left": 2, "width": 3, "height": 4}
        with patch.object(screen_tips, "SCREEN_CAPTURE_REGION", configured_region):
            with patch.object(screen_tips, "SCREEN_MONITOR_INDEX", 1):
                self.assertEqual(
                    screen_tips.resolve_capture_target(
                        capture, capture_region=EDITOR_REGION, monitor_index=2
                    ),
                    EDITOR_REGION,
                )
                self.assertEqual(
                    screen_tips.resolve_capture_target(capture, monitor_index=2),
                    configured_region,
                )

    def test_injected_monitor_overrides_configured_monitor(self):
        """With no region, an injected physical monitor takes precedence."""
        capture = fake_capture()
        with patch.object(screen_tips, "SCREEN_MONITOR_INDEX", 1):
            target = screen_tips.resolve_capture_target(capture, monitor_index=2)
        self.assertEqual(target, SECOND_MONITOR)

    def test_no_setting_uses_primary_monitor_property(self):
        """The fallback uses MSS primary_monitor, including its macOS behavior."""
        capture = fake_capture()
        capture.primary_monitor = SECOND_MONITOR
        self.assertEqual(
            screen_tips.resolve_capture_target(capture), SECOND_MONITOR
        )

    def test_invalid_or_unavailable_monitor_is_rejected(self):
        """Index zero, Booleans, non-integers, and absent monitors are invalid."""
        for monitor_index in (0, -1, True, 1.0, "1", 3):
            with self.subTest(monitor_index=monitor_index):
                with self.assertRaises(screen_tips.ScreenCaptureError):
                    screen_tips.resolve_capture_target(
                        fake_capture(), monitor_index=monitor_index
                    )

    def test_no_environment_or_interactive_configuration_exists(self):
        """Capture settings are centralized and never read from user input."""
        self.assertNotIn("os", vars(screen_tips))
        self.assertNotIn("input", vars(screen_tips))


class InMemoryCaptureTests(unittest.TestCase):
    """Verify mocked BGRA capture conversion and safe failures."""

    def test_region_capture_returns_three_channel_bgr_copy(self):
        """Known BGRA pixels lose alpha without swapping red and blue."""
        capture = fake_capture()
        with patch.object(screen_tips, "_load_numpy", return_value=FakeNumpy):
            frame = screen_tips.capture_screen_frame(
                capture, capture_region=EDITOR_REGION, monitor_index=2
            )

        capture.grab.assert_called_once_with(EDITOR_REGION)
        self.assertEqual(frame.shape, (1, 2, 3))
        self.assertEqual(frame.dtype, FakeNumpy.uint8)
        self.assertTrue(frame.flags.c_contiguous)
        self.assertEqual(frame.tolist(), [[[10, 20, 30], [200, 150, 100]]])
        self.assertIsNot(frame.data, BGRA_PIXELS)

    @unittest.skipUnless(find_spec("numpy"), "NumPy is not installed yet")
    def test_installed_numpy_produces_real_contiguous_uint8_output(self):
        """The installed dependency produces the required array representation."""
        numpy = import_module("numpy")
        capture = fake_capture()
        frame = screen_tips.capture_screen_frame(
            capture, capture_region=EDITOR_REGION
        )

        self.assertIsInstance(frame, numpy.ndarray)
        self.assertEqual(frame.shape, (1, 2, 3))
        self.assertEqual(frame.dtype, numpy.uint8)
        self.assertTrue(frame.flags.c_contiguous)
        self.assertEqual(frame.tolist(), [[[10, 20, 30], [200, 150, 100]]])

    def test_empty_and_malformed_screenshot_data_are_rejected(self):
        """Empty, wrong-channel, and failed conversions become safe errors."""
        malformed_values = (
            [],
            [[[1, 2, 3]]],
            [[[1, 2, 3, 4, 5]]],
            "conversion failure",
        )
        for screenshot in malformed_values:
            with self.subTest(screenshot=screenshot):
                capture = fake_capture(screenshot)
                with patch.object(
                    screen_tips, "_load_numpy", return_value=FakeNumpy
                ):
                    with self.assertRaises(screen_tips.ScreenCaptureError) as raised:
                        screen_tips.capture_screen_frame(capture)
                self.assertNotIn("private pixel detail", str(raised.exception))

    def test_backend_exception_is_converted_to_safe_failure(self):
        """Capture-specific details are hidden behind an application error."""
        capture = fake_capture()
        capture.grab.side_effect = RuntimeError("private screen detail")
        with self.assertRaises(screen_tips.ScreenCaptureError) as raised:
            screen_tips.capture_screen_frame(capture)
        self.assertEqual(str(raised.exception), "The screen capture failed.")
        self.assertNotIn("private screen detail", str(raised.exception))

    def test_capture_uses_no_real_backend_or_file_writes(self):
        """The capture boundary uses only the supplied backend and memory."""
        capture = fake_capture()
        fake_import = Mock(return_value=FakeNumpy)
        with patch.object(screen_tips, "import_module", fake_import):
            with patch("builtins.open") as file_open:
                screen_tips.capture_screen_frame(capture)

        fake_import.assert_called_once_with("numpy")
        file_open.assert_not_called()
        capture.grab.assert_called_once_with(FIRST_MONITOR)


class RapidOCRAdapterTests(unittest.TestCase):
    """Verify lazy OpenVINO setup and RapidOCR 3.9 result adaptation."""

    def test_factory_uses_all_three_openvino_engine_settings(self):
        """The unified RapidOCR constructor receives the exact required params."""
        openvino = object()
        engine = Mock(name="reusable_engine")
        constructor = Mock(return_value=engine)
        rapidocr = SimpleNamespace(
            EngineType=SimpleNamespace(OPENVINO=openvino),
            RapidOCR=constructor,
        )

        with patch.object(
            screen_tips, "import_module", return_value=rapidocr
        ) as loader:
            created_engine = screen_tips.create_ocr_engine()

        self.assertIs(created_engine, engine)
        loader.assert_called_once_with("rapidocr")
        constructor.assert_called_once_with(
            params={
                "Det.engine_type": openvino,
                "Cls.engine_type": openvino,
                "Rec.engine_type": openvino,
            }
        )

    def test_ordered_lines_preserve_confidence_and_position(self):
        """Text, low confidence, and quadrilaterals retain detector order."""
        numpy = import_module("numpy")
        frame = synthetic_frame()
        result = ocr_result(
            ("def first():", "    return 1"),
            (0.42, 0.99),
            numpy.asarray((FIRST_TEXT_BOX, SECOND_TEXT_BOX)),
        )
        engine = Mock(return_value=result)

        with patch.object(screen_tips, "_convert_bgra_screenshot") as conversion:
            lines = screen_tips.extract_ocr_lines(frame, engine)

        self.assertIs(engine.call_args.args[0], frame)
        conversion.assert_not_called()
        self.assertEqual(
            lines,
            (
                screen_tips.OCRLine(
                    "def first():",
                    0.42,
                    tuple(tuple(float(value) for value in point)
                          for point in FIRST_TEXT_BOX),
                ),
                screen_tips.OCRLine(
                    "    return 1",
                    0.99,
                    tuple(tuple(float(value) for value in point)
                          for point in SECOND_TEXT_BOX),
                ),
            ),
        )

    def test_missing_optional_confidence_and_position_are_preserved(self):
        """Recognition-only data remains useful when geometry is unavailable."""
        engine = Mock(return_value=ocr_result(("print('ok')",), None, None))

        lines = screen_tips.extract_ocr_lines(synthetic_frame(), engine)

        self.assertEqual(
            lines,
            (screen_tips.OCRLine("print('ok')", None, None),),
        )

    def test_valid_no_text_results_return_an_empty_collection(self):
        """Both installed empty-output shapes adapt to an empty tuple."""
        empty_results = (
            ocr_result(None, None, None),
            ocr_result((), (), ()),
        )
        for result in empty_results:
            with self.subTest(result=result):
                engine = Mock(return_value=result)
                self.assertEqual(
                    screen_tips.extract_ocr_lines(synthetic_frame(), engine), ()
                )

    def test_malformed_results_are_rejected_safely(self):
        """Missing fields, length mismatches, and invalid values fail safely."""
        malformed_results = (
            None,
            SimpleNamespace(txts=("line",), scores=(0.9,)),
            ocr_result(None, (0.9,), None),
            ocr_result(("line",), (), None),
            ocr_result((123,), (0.9,), None),
            ocr_result(("line",), ("high",), None),
            ocr_result(("line",), (0.9,), (((0, 0),),)),
        )
        for result in malformed_results:
            with self.subTest(result=result):
                engine = Mock(return_value=result)
                with self.assertRaises(screen_tips.OCRProcessingError) as raised:
                    screen_tips.extract_ocr_lines(synthetic_frame(), engine)
                self.assertEqual(
                    str(raised.exception),
                    "RapidOCR returned malformed recognition data.",
                )

    def test_initialization_failures_do_not_expose_details(self):
        """Import and constructor failures become one safe local OCR error."""
        private_detail = "private OCR initialization detail"
        with patch.object(
            screen_tips, "import_module", side_effect=ImportError(private_detail)
        ):
            with self.assertRaises(screen_tips.OCRProcessingError) as raised:
                screen_tips.create_ocr_engine()
        self.assertNotIn(private_detail, str(raised.exception))

        rapidocr = SimpleNamespace(
            EngineType=SimpleNamespace(OPENVINO=object()),
            RapidOCR=Mock(side_effect=RuntimeError(private_detail)),
        )
        with patch.object(screen_tips, "import_module", return_value=rapidocr):
            with self.assertRaises(screen_tips.OCRProcessingError) as raised:
                screen_tips.create_ocr_engine()
        self.assertNotIn(private_detail, str(raised.exception))

    def test_recognition_failure_does_not_expose_frame_or_details(self):
        """Backend recognition exceptions never reveal captured input."""
        private_detail = "private recognized screen detail"
        engine = Mock(side_effect=RuntimeError(private_detail))

        with self.assertRaises(screen_tips.OCRProcessingError) as raised:
            screen_tips.extract_ocr_lines(synthetic_frame(), engine)

        self.assertEqual(
            str(raised.exception), "Local OCR could not process the screen frame."
        )
        self.assertNotIn(private_detail, str(raised.exception))

    def test_one_engine_can_be_reused_without_reconstruction(self):
        """The factory constructs once while the adapter accepts repeated frames."""
        result = ocr_result(("x = 1",), (0.95,), (FIRST_TEXT_BOX,))
        engine = Mock(side_effect=(result, result))
        constructor = Mock(return_value=engine)
        rapidocr = SimpleNamespace(
            EngineType=SimpleNamespace(OPENVINO=object()),
            RapidOCR=constructor,
        )
        frame = synthetic_frame()

        with patch.object(screen_tips, "import_module", return_value=rapidocr):
            reusable_engine = screen_tips.create_ocr_engine()
        first_lines = screen_tips.extract_ocr_lines(frame, reusable_engine)
        second_lines = screen_tips.extract_ocr_lines(frame, reusable_engine)

        constructor.assert_called_once()
        self.assertEqual(engine.call_count, 2)
        self.assertIs(engine.call_args_list[0].args[0], frame)
        self.assertIs(engine.call_args_list[1].args[0], frame)
        self.assertEqual(first_lines, second_lines)

    def test_adapter_has_no_capture_file_network_or_openai_side_effects(self):
        """OCR adaptation uses only the injected engine and in-memory frame."""
        engine = Mock(return_value=ocr_result(("value = 1",), (0.9,), None))
        frame = synthetic_frame()
        with patch.object(screen_tips, "capture_screen_frame") as capture:
            with patch("builtins.open") as file_open:
                screen_tips.extract_ocr_lines(frame, engine)

        capture.assert_not_called()
        file_open.assert_not_called()
        self.assertNotIn("openai", vars(screen_tips))
        self.assertNotIn("socket", vars(screen_tips))


if __name__ == "__main__":
    unittest.main()
