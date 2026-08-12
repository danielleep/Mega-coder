"""Tests for Mega Coder's optional fault injection."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import mega_coder


SUCCESS_CODE = "assert 2 + 3 == 5\nprint('success')\n"
FAILURE_CODE = "assert False, 'expected repair failure'\n"
INVALID_CODE = "import socket\nassert True\n"
DESCRIPTION = "Create a deterministic calculator"
WARNING_TEXT = "Testing repair behavior: a deliberate error was added"


class FaultInjectionTests(unittest.TestCase):
    """Verify fault injection and its interaction with the repair loop."""

    def run_develop(self, output_file, responses):
        """Run option 1 with mocked API responses and a temporary output file."""
        with patch("mega_coder.OUTPUT_FILE", output_file):
            with patch(
                "mega_coder.ask_for_program_description",
                return_value=DESCRIPTION,
            ):
                with patch("mega_coder.call_openai", side_effect=responses) as api:
                    with redirect_stdout(StringIO()) as captured:
                        mega_coder.develop_program()
        return api, captured.getvalue()

    def test_disabled_fault_injection_leaves_normal_flow_unchanged(self):
        """Disabled injection leaves code unchanged and causes no repair."""
        code, was_injected = mega_coder.corrupt_generated_code(
            SUCCESS_CODE,
            enabled=False,
            probability=1.0,
            random_number_function=lambda: 0.0,
        )
        self.assertEqual(code, SUCCESS_CODE)
        self.assertFalse(was_injected)

        with TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "generated-code-openai.py"
            with patch("mega_coder.ENABLE_FAULT_INJECTION", False):
                with patch("mega_coder.repair_generated_program") as repair:
                    api, output = self.run_develop(output_file, [SUCCESS_CODE])

        self.assertEqual(api.call_count, 1)
        repair.assert_not_called()
        self.assertNotIn(WARNING_TEXT, output)

    def test_enabled_but_not_selected_does_not_warn(self):
        """A random value at the probability boundary does not inject."""
        code, was_injected = mega_coder.corrupt_generated_code(
            SUCCESS_CODE,
            enabled=True,
            probability=0.25,
            random_number_function=lambda: 0.25,
        )
        self.assertEqual(code, SUCCESS_CODE)
        self.assertFalse(was_injected)

        with TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "generated-code-openai.py"
            with patch("mega_coder.ENABLE_FAULT_INJECTION", True):
                with patch("mega_coder.FAULT_INJECTION_PROBABILITY", 0.25):
                    with patch("mega_coder.random.random", return_value=0.25):
                        api, output = self.run_develop(output_file, [SUCCESS_CODE])

        self.assertEqual(api.call_count, 1)
        self.assertNotIn(WARNING_TEXT, output)

    def test_enabled_and_selected_appends_valid_runtime_failure(self):
        """Selected injection preserves assertions and valid Python syntax."""
        code, was_injected = mega_coder.corrupt_generated_code(
            SUCCESS_CODE,
            enabled=True,
            probability=0.25,
            random_number_function=lambda: 0.24,
        )

        self.assertTrue(was_injected)
        self.assertIn(mega_coder.INJECTED_FAILURE, code)
        self.assertIn("assert 2 + 3 == 5", code)
        self.assertTrue(code.endswith("\n"))
        mega_coder.validate_generated_code(code)

    def test_probability_boundaries_and_invalid_values(self):
        """Zero never injects, one always injects, and invalid values fail."""
        _, injected_at_zero = mega_coder.corrupt_generated_code(
            SUCCESS_CODE,
            enabled=True,
            probability=0.0,
            random_number_function=lambda: 0.0,
        )
        _, injected_at_one = mega_coder.corrupt_generated_code(
            SUCCESS_CODE,
            enabled=True,
            probability=1.0,
            random_number_function=lambda: 0.999,
        )

        self.assertFalse(injected_at_zero)
        self.assertTrue(injected_at_one)
        for invalid_probability in (-0.01, 1.01):
            with self.assertRaises(ValueError):
                mega_coder.corrupt_generated_code(
                    SUCCESS_CODE,
                    enabled=True,
                    probability=invalid_probability,
                )

    def test_full_flow_injects_once_and_first_repair_succeeds(self):
        """The initial candidate fails, then an uncorrupted repair succeeds."""
        with TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "generated-code-openai.py"
            execution_results = []
            real_runner = mega_coder.run_generated_program

            def record_execution(file_path):
                result = real_runner(file_path)
                execution_results.append(result)
                return result

            with patch("mega_coder.ENABLE_FAULT_INJECTION", True):
                with patch("mega_coder.FAULT_INJECTION_PROBABILITY", 1.0):
                    with patch("mega_coder.random.random", return_value=0.0):
                        with patch(
                            "mega_coder.corrupt_generated_code",
                            wraps=mega_coder.corrupt_generated_code,
                        ) as corrupt:
                            with patch(
                                "mega_coder.run_generated_program",
                                side_effect=record_execution,
                            ):
                                api, output = self.run_develop(
                                    output_file,
                                    [SUCCESS_CODE, SUCCESS_CODE],
                                )

            self.assertEqual(api.call_count, 2)
            self.assertEqual(corrupt.call_count, 1)
            self.assertEqual(len(execution_results), 2)
            self.assertFalse(execution_results[0].success)
            self.assertIn(
                "Deliberately injected test failure",
                execution_results[0].stderr,
            )
            self.assertTrue(execution_results[1].success)
            self.assertEqual(output_file.read_text(encoding="utf-8"), SUCCESS_CODE)

        self.assertIn(WARNING_TEXT, output)
        self.assertIn("Repair attempt 1 of 5", output)

    def test_natural_validation_failure_is_not_injected(self):
        """Naturally invalid initial code goes directly to normal repair."""
        with TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "generated-code-openai.py"
            with patch("mega_coder.ENABLE_FAULT_INJECTION", True):
                with patch(
                    "mega_coder.corrupt_generated_code",
                    wraps=mega_coder.corrupt_generated_code,
                ) as corrupt:
                    api, output = self.run_develop(
                        output_file,
                        [INVALID_CODE, SUCCESS_CODE],
                    )

            self.assertEqual(output_file.read_text(encoding="utf-8"), SUCCESS_CODE)

        self.assertEqual(api.call_count, 2)
        corrupt.assert_not_called()
        self.assertNotIn(WARNING_TEXT, output)
        self.assertNotIn("Deliberately injected test failure", output)

    def test_persistent_repairs_are_never_reinjected(self):
        """Only the initial candidate is corrupted across five failed repairs."""
        with TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "generated-code-openai.py"
            responses = [SUCCESS_CODE] + [FAILURE_CODE] * 5
            with patch("mega_coder.ENABLE_FAULT_INJECTION", True):
                with patch("mega_coder.FAULT_INJECTION_PROBABILITY", 1.0):
                    with patch("mega_coder.random.random", return_value=0.0):
                        with patch(
                            "mega_coder.corrupt_generated_code",
                            wraps=mega_coder.corrupt_generated_code,
                        ) as corrupt:
                            api, output = self.run_develop(output_file, responses)

            final_code = output_file.read_text(encoding="utf-8")

        self.assertEqual(api.call_count, 1 + mega_coder.MAX_REPAIR_ATTEMPTS)
        self.assertEqual(corrupt.call_count, 1)
        self.assertEqual(output.count("Repair attempt"), 5)
        self.assertNotIn(mega_coder.INJECTED_FAILURE, final_code)


if __name__ == "__main__":
    unittest.main()
