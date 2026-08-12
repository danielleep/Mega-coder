"""Tests for Mega Coder's optional fault injection."""

from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mega_coder


SUCCESS_CODE = "assert 2 + 3 == 5\nprint('success')\n"
FAILURE_CODE = "assert False, 'expected repair failure'\n"
INVALID_CODE = "import socket\nassert True\n"
WORKING_CODE = """def total(values):
    return sum(values)

assert total([1, 2, 3]) == 6
print("total:", total([1, 2, 3]))
"""
OPTIMIZED_CODE = """def total(values):
    return sum(values)

assert total([1, 2, 3]) == 6
print("total:", total((1, 2, 3)))
"""
DESCRIPTION = "Create a deterministic calculator"
WARNING_TEXT = "Testing repair behavior: a deliberate error was added"


def execution_result(**overrides):
    """Create a concise execution result for mocked program runs."""
    values = {
        "success": True,
        "returncode": 0,
        "stdout": "success\n",
        "stderr": "",
        "timed_out": False,
        "duration_ms": 10.0,
    }
    values.update(overrides)
    return mega_coder.ExecutionResult(**values)


class FaultInjectionTests(unittest.TestCase):
    """Verify fault injection and its interaction with the repair loop."""

    def run_develop(self, output_file, responses, execution_results=None):
        """Run option 1 with mocked API responses and a temporary output file."""
        if execution_results is None:
            execution_results = [execution_result()]

        with patch("mega_coder.OUTPUT_FILE", output_file):
            with patch(
                "mega_coder.ask_for_program_description",
                return_value=DESCRIPTION,
            ):
                with patch("mega_coder.call_openai", side_effect=responses) as api:
                    with patch(
                        "mega_coder.run_generated_program",
                        side_effect=execution_results,
                    ):
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
                    with patch("mega_coder.optimize_generated_program"):
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
                        with patch("mega_coder.optimize_generated_program"):
                            api, output = self.run_develop(
                                output_file,
                                [SUCCESS_CODE],
                            )

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
            with patch("mega_coder.ENABLE_FAULT_INJECTION", True):
                with patch("mega_coder.FAULT_INJECTION_PROBABILITY", 1.0):
                    with patch("mega_coder.random.random", return_value=0.0):
                        with patch(
                            "mega_coder.corrupt_generated_code",
                            wraps=mega_coder.corrupt_generated_code,
                        ) as corrupt:
                            with patch("mega_coder.optimize_generated_program"):
                                api, output = self.run_develop(
                                    output_file,
                                    [SUCCESS_CODE, SUCCESS_CODE],
                                    [
                                        execution_result(
                                            success=False,
                                            stderr=(
                                                "Deliberately injected test failure"
                                            ),
                                            returncode=1,
                                        ),
                                        execution_result(),
                                    ],
                                )

            self.assertEqual(api.call_count, 2)
            self.assertEqual(corrupt.call_count, 1)
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
                    with patch("mega_coder.optimize_generated_program"):
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
                            with patch(
                                "mega_coder.optimize_generated_program"
                            ) as optimize:
                                api, output = self.run_develop(
                                    output_file,
                                    responses,
                                    [
                                        execution_result(
                                            success=False,
                                            returncode=1,
                                        )
                                    ]
                                    * 6,
                                )

            final_code = output_file.read_text(encoding="utf-8")

        self.assertEqual(api.call_count, 1 + mega_coder.MAX_REPAIR_ATTEMPTS)
        self.assertEqual(corrupt.call_count, 1)
        self.assertEqual(output.count("Repair attempt"), 5)
        self.assertNotIn(mega_coder.INJECTED_FAILURE, final_code)
        optimize.assert_not_called()


class OptimizationTests(unittest.TestCase):
    """Verify optimization candidates cannot replace good code prematurely."""

    def run_optimization(self, candidate, candidate_result=None, api_error=None):
        """Evaluate a mocked candidate while keeping all files temporary."""
        if candidate_result is None:
            candidate_result = execution_result(
                stdout="total: 6\n",
                duration_ms=5.0,
            )

        with TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "generated-code-openai.py"
            output_file.write_text(WORKING_CODE, encoding="utf-8")
            working_result = execution_result(
                stdout="total: 6\n",
                duration_ms=10.0,
            )
            call_arguments = {"side_effect": api_error}
            if api_error is None:
                call_arguments = {"return_value": candidate}

            with patch("mega_coder.call_openai", **call_arguments) as api:
                with patch(
                    "mega_coder.run_generated_program",
                    return_value=candidate_result,
                ) as runner:
                    with redirect_stdout(StringIO()) as captured:
                        accepted = mega_coder.optimize_generated_program(
                            DESCRIPTION,
                            WORKING_CODE,
                            working_result,
                            output_file,
                        )
            final_source = output_file.read_text(encoding="utf-8")

        return accepted, captured.getvalue(), final_source, api, runner

    def test_faster_candidate_replaces_file_and_prints_exact_message(self):
        """A valid, equivalent, strictly faster candidate is accepted."""
        candidate_result = execution_result(
            stdout="total: 6\n",
            duration_ms=5.1254,
        )
        accepted, output, final_source, api, runner = self.run_optimization(
            OPTIMIZED_CODE,
            candidate_result,
        )

        self.assertTrue(accepted)
        self.assertEqual(api.call_count, 1)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(final_source, OPTIMIZED_CODE)
        self.assertIn(
            "Code running time optimized! It now runs in 5.125 milliseconds, "
            "while before it was 10.000 milliseconds",
            output,
        )

    def test_equal_timing_keeps_working_file(self):
        """Equal duration is not a measured improvement."""
        result = execution_result(stdout="total: 6\n", duration_ms=10.0)
        accepted, output, final_source, _, _ = self.run_optimization(
            OPTIMIZED_CODE,
            result,
        )

        self.assertFalse(accepted)
        self.assertEqual(final_source, WORKING_CODE)
        self.assertIn("candidate was not faster", output)

    def test_slower_candidate_keeps_working_file(self):
        """A slower candidate cannot replace the current program."""
        result = execution_result(stdout="total: 6\n", duration_ms=11.0)
        accepted, output, final_source, _, _ = self.run_optimization(
            OPTIMIZED_CODE,
            result,
        )

        self.assertFalse(accepted)
        self.assertEqual(final_source, WORKING_CODE)
        self.assertNotIn("Code running time optimized!", output)

    def test_invalid_syntax_is_rejected_without_execution(self):
        """Invalid optimized source never reaches the subprocess runner."""
        accepted, _, final_source, _, runner = self.run_optimization("def bad(")

        self.assertFalse(accepted)
        self.assertEqual(final_source, WORKING_CODE)
        runner.assert_not_called()

    def test_changed_or_removed_assertion_is_rejected_before_execution(self):
        """Changing or removing an assertion prevents candidate execution."""
        changed = WORKING_CODE.replace("== 6", "== 7")
        removed = WORKING_CODE.replace("assert total([1, 2, 3]) == 6\n", "")

        for candidate in (changed, removed):
            with self.subTest(candidate=candidate):
                accepted, _, final_source, _, runner = self.run_optimization(
                    candidate
                )
                self.assertFalse(accepted)
                self.assertEqual(final_source, WORKING_CODE)
                runner.assert_not_called()

    def test_added_assertion_is_rejected_before_execution(self):
        """Adding an assertion also violates exact assertion preservation."""
        candidate = WORKING_CODE.replace(
            "print(\"total:\", total([1, 2, 3]))",
            "assert total([]) == 0\nprint(\"total:\", total([1, 2, 3]))",
        )
        accepted, _, final_source, _, runner = self.run_optimization(candidate)

        self.assertFalse(accepted)
        self.assertEqual(final_source, WORKING_CODE)
        runner.assert_not_called()

    def test_nonzero_candidate_keeps_working_file(self):
        """A candidate with a nonzero exit status is rejected."""
        result = execution_result(
            success=False,
            stdout="total: 6\n",
            stderr="failure",
            returncode=1,
            duration_ms=5.0,
        )
        accepted, _, final_source, _, _ = self.run_optimization(
            OPTIMIZED_CODE,
            result,
        )

        self.assertFalse(accepted)
        self.assertEqual(final_source, WORKING_CODE)

    def test_timed_out_candidate_keeps_working_file(self):
        """A timed-out candidate is rejected."""
        result = execution_result(
            success=False,
            stdout="",
            timed_out=True,
            returncode=None,
            duration_ms=30_000.0,
        )
        accepted, _, final_source, _, _ = self.run_optimization(
            OPTIMIZED_CODE,
            result,
        )

        self.assertFalse(accepted)
        self.assertEqual(final_source, WORKING_CODE)

    def test_candidate_with_different_stdout_keeps_working_file(self):
        """Successful execution is insufficient when console output changes."""
        result = execution_result(stdout="different\n", duration_ms=5.0)
        accepted, output, final_source, _, _ = self.run_optimization(
            OPTIMIZED_CODE,
            result,
        )

        self.assertFalse(accepted)
        self.assertEqual(final_source, WORKING_CODE)
        self.assertIn("changed the program output", output)

    def test_candidate_with_stderr_keeps_working_file(self):
        """A candidate that reports an error on stderr is rejected."""
        result = execution_result(
            stdout="total: 6\n",
            stderr="warning treated as an error",
            duration_ms=5.0,
        )
        accepted, _, final_source, _, _ = self.run_optimization(
            OPTIMIZED_CODE,
            result,
        )

        self.assertFalse(accepted)
        self.assertEqual(final_source, WORKING_CODE)

    def test_api_failure_is_safe_and_keeps_working_file(self):
        """An SDK failure neither crashes nor prints its potentially secret text."""
        secret_text = "private-token-should-not-appear"
        api_error = mega_coder.OpenAIError(secret_text)
        accepted, output, final_source, api, runner = self.run_optimization(
            None,
            api_error=api_error,
        )

        self.assertFalse(accepted)
        self.assertEqual(api.call_count, 1)
        runner.assert_not_called()
        self.assertEqual(final_source, WORKING_CODE)
        self.assertNotIn(secret_text, output)

    def test_optimization_prompt_contains_behavior_and_safety_rules(self):
        """The optimization prompt treats supplied data as untrusted."""
        prompt = mega_coder.build_optimization_prompt(DESCRIPTION, WORKING_CODE)

        self.assertIn(DESCRIPTION, prompt)
        self.assertIn(WORKING_CODE, prompt)
        for requirement in (
            "untrusted data",
            "same assert statements",
            "same program behavior, results, and console output",
            "command-line arguments",
            "input()",
            "internet",
            "web requests",
            "downloads",
            "external data",
            "install packages",
            "standard library",
            "deterministic",
            "Markdown fences",
        ):
            self.assertIn(requirement, prompt)

    def test_optimization_runs_after_initial_success(self):
        """Initial success is followed by exactly one optimization request."""
        with TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "generated-code-openai.py"
            results = [
                execution_result(stdout="total: 6\n", duration_ms=10.0),
                execution_result(stdout="total: 6\n", duration_ms=5.0),
            ]
            with patch("mega_coder.OUTPUT_FILE", output_file):
                with patch(
                    "mega_coder.ask_for_program_description",
                    return_value=DESCRIPTION,
                ):
                    with patch(
                        "mega_coder.call_openai",
                        side_effect=[WORKING_CODE, OPTIMIZED_CODE],
                    ) as api:
                        with patch(
                            "mega_coder.run_generated_program",
                            side_effect=results,
                        ):
                            with redirect_stdout(StringIO()):
                                mega_coder.develop_program()

            self.assertEqual(api.call_count, 2)
            self.assertEqual(
                output_file.read_text(encoding="utf-8"),
                OPTIMIZED_CODE,
            )

    def test_optimization_runs_after_successful_repair(self):
        """A repaired success is followed by one optimization request."""
        with TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "generated-code-openai.py"
            results = [
                execution_result(success=False, returncode=1),
                execution_result(stdout="total: 6\n", duration_ms=10.0),
                execution_result(stdout="total: 6\n", duration_ms=5.0),
            ]
            with patch("mega_coder.OUTPUT_FILE", output_file):
                with patch(
                    "mega_coder.ask_for_program_description",
                    return_value=DESCRIPTION,
                ):
                    with patch(
                        "mega_coder.call_openai",
                        side_effect=[FAILURE_CODE, WORKING_CODE, OPTIMIZED_CODE],
                    ) as api:
                        with patch(
                            "mega_coder.run_generated_program",
                            side_effect=results,
                        ):
                            with redirect_stdout(StringIO()):
                                mega_coder.develop_program()

            self.assertEqual(api.call_count, 3)
            self.assertEqual(
                output_file.read_text(encoding="utf-8"),
                OPTIMIZED_CODE,
            )


class ExecutionTimingTests(unittest.TestCase):
    """Verify elapsed timing and child-process environment handling."""

    def test_timing_wraps_only_the_subprocess_execution(self):
        """The elapsed clock is sampled immediately around subprocess.run."""
        events = []

        def timer():
            events.append("timer")
            return 2.0 if len(events) == 1 else 2.025

        def run_process(*_args, **_kwargs):
            events.append("subprocess")
            return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

        with TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "candidate.py"
            with patch("mega_coder.time.perf_counter", side_effect=timer):
                with patch(
                    "mega_coder.subprocess.run",
                    side_effect=run_process,
                ):
                    result = mega_coder.run_generated_program(file_path)

        self.assertEqual(events, ["timer", "subprocess", "timer"])
        self.assertAlmostEqual(result.duration_ms, 25.0)

    def test_generated_child_does_not_receive_openai_api_key(self):
        """The child environment omits the OpenAI credential."""
        completed = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        with TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "candidate.py"
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder"}):
                with patch(
                    "mega_coder.subprocess.run",
                    return_value=completed,
                ) as run_process:
                    mega_coder.run_generated_program(file_path)

        child_environment = run_process.call_args.kwargs["env"]
        self.assertNotIn("OPENAI_API_KEY", child_environment)
        self.assertFalse(run_process.call_args.kwargs["shell"])


if __name__ == "__main__":
    unittest.main()
