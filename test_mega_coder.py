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
LINT_CLEAN_CODE = '''def total(values):
    """Return the sum of the supplied values."""
    return sum(values)

assert total([1, 2, 3]) == 6
print("total:", total((1, 2, 3)))
'''
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


def pylint_result(issue_count=0, **overrides):
    """Create a deterministic Pylint result without running Pylint."""
    diagnostics = "".join(
        f"candidate.py:{index + 1}:0: C0114: Issue {index + 1} (test-issue)\n"
        for index in range(issue_count)
    )
    values = {
        "command_succeeded": True,
        "returncode": 0 if issue_count == 0 else 16,
        "stdout": diagnostics,
        "stderr": "",
        "timed_out": False,
        "issue_count": issue_count,
        "operational_error": "",
    }
    values.update(overrides)
    return mega_coder.PylintResult(**values)


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
                    with patch("mega_coder.finish_working_program"):
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
                        with patch("mega_coder.finish_working_program"):
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
                            with patch("mega_coder.finish_working_program"):
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
                    with patch("mega_coder.finish_working_program"):
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
                                "mega_coder.finish_working_program"
                            ) as finish:
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
        finish.assert_not_called()


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
                            with patch("mega_coder.check_and_repair_lint"):
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
                            with patch("mega_coder.check_and_repair_lint"):
                                with redirect_stdout(StringIO()):
                                    mega_coder.develop_program()

            self.assertEqual(api.call_count, 3)
            self.assertEqual(
                output_file.read_text(encoding="utf-8"),
                OPTIMIZED_CODE,
            )


class LintRepairTests(unittest.TestCase):
    """Verify safe Pylint checks and the independent three-attempt loop."""

    def run_lint_stage(self, responses, pylint_results, execution_results=None):
        """Run the lint stage with temporary files and deterministic mocks."""
        execution_results = execution_results or []
        with TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "generated-code-openai.py"
            output_file.write_text(WORKING_CODE, encoding="utf-8")
            working_result = execution_result(stdout="total: 6\n")
            with patch("mega_coder.call_openai", side_effect=responses) as api:
                with patch(
                    "mega_coder.run_pylint", side_effect=pylint_results
                ) as pylint_check:
                    with patch(
                        "mega_coder.run_generated_program",
                        side_effect=execution_results,
                    ) as runner:
                        with redirect_stdout(StringIO()) as captured:
                            mega_coder.check_and_repair_lint(
                                DESCRIPTION,
                                WORKING_CODE,
                                working_result,
                                output_file,
                            )
            final_source = output_file.read_text(encoding="utf-8")
        return api, pylint_check, runner, captured.getvalue(), final_source

    def test_initially_clean_code_stops_without_api_request(self):
        """A clean initial report prints the exact success message."""
        api, pylint_check, runner, output, final_source = self.run_lint_stage(
            [], [pylint_result()]
        )

        self.assertEqual(pylint_check.call_count, 1)
        api.assert_not_called()
        runner.assert_not_called()
        self.assertEqual(final_source, WORKING_CODE)
        self.assertIn("Amazing. No lint errors/warnings", output.splitlines())

    def test_one_successful_lint_repair_replaces_file(self):
        """One safe candidate can remove every diagnostic and stop the loop."""
        api, _, runner, output, final_source = self.run_lint_stage(
            [LINT_CLEAN_CODE],
            [pylint_result(2), pylint_result()],
            [execution_result(stdout="total: 6\n")],
        )

        self.assertEqual(api.call_count, 1)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(final_source, LINT_CLEAN_CODE)
        self.assertIn("Amazing. No lint errors/warnings", output.splitlines())

    def test_gradual_improvement_uses_latest_source_and_report(self):
        """An accepted partial repair becomes the basis of the next request."""
        api, _, _, _, final_source = self.run_lint_stage(
            [OPTIMIZED_CODE, LINT_CLEAN_CODE],
            [pylint_result(3), pylint_result(1), pylint_result()],
            [
                execution_result(stdout="total: 6\n"),
                execution_result(stdout="total: 6\n"),
            ],
        )

        second_prompt = api.call_args_list[1].args[0]
        self.assertEqual(api.call_count, 2)
        self.assertIn(OPTIMIZED_CODE, second_prompt)
        self.assertIn("Issue count: 1", second_prompt)
        self.assertEqual(final_source, LINT_CLEAN_CODE)

    def test_persistent_diagnostics_stop_after_exactly_three_requests(self):
        """Three non-improving responses cannot cause a fourth request."""
        api, _, _, output, final_source = self.run_lint_stage(
            [OPTIMIZED_CODE] * 3,
            [pylint_result(2)] * 4,
            [execution_result(stdout="total: 6\n")] * 3,
        )

        self.assertEqual(api.call_count, mega_coder.MAX_LINT_REPAIR_ATTEMPTS)
        self.assertEqual(output.count("Lint repair attempt"), 3)
        self.assertIn("There are still lint errors/warnings", output.splitlines())
        self.assertEqual(final_source, WORKING_CODE)

    def test_invalid_and_assertion_changing_candidates_are_not_executed(self):
        """Syntax and exact-assertion failures are rejected before execution."""
        changed = WORKING_CODE.replace("== 6", "== 7")
        removed = WORKING_CODE.replace("assert total([1, 2, 3]) == 6\n", "")
        added = WORKING_CODE.replace(
            "print(\"total:\", total([1, 2, 3]))",
            "assert total([]) == 0\nprint(\"total:\", total([1, 2, 3]))",
        )
        cases = (["def broken("] * 3, [changed, removed, added])

        for responses in cases:
            with self.subTest(responses=responses):
                _, _, runner, _, final_source = self.run_lint_stage(
                    responses, [pylint_result(2)]
                )
                runner.assert_not_called()
                self.assertEqual(final_source, WORKING_CODE)

    def test_execution_failure_timeout_and_output_change_are_rejected(self):
        """Runtime regressions cannot replace or lint the working source."""
        results = [
            execution_result(success=False, returncode=1, stderr="failure"),
            execution_result(success=False, returncode=None, timed_out=True),
            execution_result(stdout="changed\n"),
        ]
        _, pylint_check, _, _, final_source = self.run_lint_stage(
            [OPTIMIZED_CODE] * 3,
            [pylint_result(2)],
            results,
        )

        self.assertEqual(pylint_check.call_count, 1)
        self.assertEqual(final_source, WORKING_CODE)

    def test_same_or_greater_diagnostic_count_is_not_better(self):
        """Only a strict issue-count reduction is accepted."""
        _, _, _, _, final_source = self.run_lint_stage(
            [OPTIMIZED_CODE] * 3,
            [pylint_result(2), pylint_result(2), pylint_result(3), pylint_result(2)],
            [execution_result(stdout="total: 6\n")] * 3,
        )

        self.assertEqual(final_source, WORKING_CODE)

    def test_operational_pylint_failures_stop_safely(self):
        """Timeout, missing command, malformed output, and usage errors are not clean."""
        failures = [
            pylint_result(
                command_succeeded=False,
                timed_out=True,
                operational_error="Pylint exceeded its execution timeout.",
            ),
            pylint_result(
                command_succeeded=False,
                returncode=None,
                operational_error="Pylint could not start.",
            ),
            pylint_result(
                command_succeeded=False,
                returncode=1,
                stdout="unrecognized output",
                operational_error="Pylint output could not be interpreted safely.",
            ),
            pylint_result(
                command_succeeded=False,
                returncode=32,
                operational_error="Pylint reported a command usage error.",
            ),
        ]

        for failure in failures:
            with self.subTest(error=failure.operational_error):
                api, _, _, output, final_source = self.run_lint_stage([], [failure])
                api.assert_not_called()
                self.assertNotIn("Amazing. No lint errors/warnings", output)
                self.assertEqual(final_source, WORKING_CODE)

    def test_openai_failure_is_safe_and_keeps_working_file(self):
        """The SDK exception text is not exposed by lint repair."""
        secret_text = "private-lint-token-should-not-appear"
        api, _, runner, output, final_source = self.run_lint_stage(
            [mega_coder.OpenAIError(secret_text)],
            [pylint_result(1)],
        )

        self.assertEqual(api.call_count, 1)
        runner.assert_not_called()
        self.assertNotIn(secret_text, output)
        self.assertEqual(final_source, WORKING_CODE)

    def test_lint_prompt_contains_complete_report_and_restrictions(self):
        """The prompt includes trusted rules and all untrusted input sections."""
        result = pylint_result(1, stderr="complete stderr")
        prompt = mega_coder.build_lint_repair_prompt(
            DESCRIPTION, WORKING_CODE, result, 2
        )

        for required_text in (
            DESCRIPTION,
            WORKING_CODE,
            result.stdout,
            result.stderr,
            "Return code: 16",
            "Lint repair attempt 2 of 3",
            "untrusted",
            "same behavior",
            "console output",
            "assertion",
            "# pylint: disable",
            "sys.argv",
            "input()",
            "internet",
            "external APIs",
            "standard library",
            "subprocesses",
            "files",
        ):
            self.assertIn(required_text, prompt)


class PylintPipelineTests(unittest.TestCase):
    """Verify one lint handoff after each successful development path."""

    def test_optimization_acceptance_and_rejection_each_lint_once(self):
        """The best source after either optimization decision is linted once."""
        for accepted in (True, False):
            with self.subTest(accepted=accepted):
                with TemporaryDirectory() as temporary_directory:
                    output_file = Path(temporary_directory) / "generated-code-openai.py"
                    output_file.write_text(WORKING_CODE, encoding="utf-8")

                    def optimize(
                        *_args,
                        accepted_result=accepted,
                        candidate_path=output_file,
                    ):
                        if accepted_result:
                            candidate_path.write_text(
                                OPTIMIZED_CODE, encoding="utf-8"
                            )
                        return accepted_result

                    with patch(
                        "mega_coder.optimize_generated_program",
                        side_effect=optimize,
                    ):
                        with patch(
                            "mega_coder.check_and_repair_lint",
                            return_value=True,
                        ) as lint_stage:
                            mega_coder.finish_working_program(
                                DESCRIPTION,
                                WORKING_CODE,
                                execution_result(stdout="total: 6\n"),
                                output_file,
                            )

                lint_stage.assert_called_once()
                expected_source = OPTIMIZED_CODE if accepted else WORKING_CODE
                self.assertEqual(lint_stage.call_args.args[1], expected_source)

    def test_initial_success_reaches_lint_once(self):
        """Initial execution success has one post-optimization lint stage."""
        with TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "generated-code-openai.py"
            with patch("mega_coder.OUTPUT_FILE", output_file):
                with patch(
                    "mega_coder.ask_for_program_description", return_value=DESCRIPTION
                ):
                    with patch("mega_coder.call_openai", return_value=WORKING_CODE):
                        with patch(
                            "mega_coder.run_generated_program",
                            return_value=execution_result(stdout="total: 6\n"),
                        ):
                            with patch("mega_coder.optimize_generated_program"):
                                with patch(
                                    "mega_coder.check_and_repair_lint"
                                ) as lint_stage:
                                    with redirect_stdout(StringIO()):
                                        mega_coder.develop_program()

        lint_stage.assert_called_once()

    def test_successful_execution_repair_reaches_lint_once(self):
        """A working repaired file is optimized and then linted once."""
        with TemporaryDirectory() as temporary_directory:
            output_file = Path(temporary_directory) / "generated-code-openai.py"
            with patch("mega_coder.call_openai", return_value=WORKING_CODE):
                with patch(
                    "mega_coder.run_generated_program",
                    return_value=execution_result(stdout="total: 6\n"),
                ):
                    with patch("mega_coder.optimize_generated_program"):
                        with patch(
                            "mega_coder.check_and_repair_lint"
                        ) as lint_stage:
                            with redirect_stdout(StringIO()):
                                mega_coder.repair_generated_program(
                                    DESCRIPTION,
                                    FAILURE_CODE,
                                    "failure",
                                    output_file,
                                )

        lint_stage.assert_called_once()


class PylintRunnerTests(unittest.TestCase):
    """Verify the real Pylint subprocess wrapper without running Pylint."""

    def test_command_parsing_and_sanitized_environment(self):
        """Pylint uses the active interpreter and counts diagnostic lines."""
        stdout = (
            "candidate.py:1:0: C0114: First issue (first-issue)\n"
            "candidate.py:2:0: W0611: Second issue (second-issue)\n"
        )
        completed = SimpleNamespace(returncode=20, stdout=stdout, stderr="")
        with TemporaryDirectory() as temporary_directory:
            file_path = Path(temporary_directory) / "candidate.py"
            file_path.write_text(WORKING_CODE, encoding="utf-8")
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-placeholder"}):
                with patch(
                    "mega_coder.subprocess.run", return_value=completed
                ) as run_process:
                    result = mega_coder.run_pylint(file_path)

        expected_command = [
            mega_coder.sys.executable,
            "-m",
            "pylint",
            "--reports=no",
            "--score=no",
            "--persistent=no",
            str(file_path.resolve()),
        ]
        self.assertEqual(run_process.call_args.args[0], expected_command)
        self.assertEqual(result.issue_count, 2)
        self.assertTrue(result.command_succeeded)
        self.assertNotIn("OPENAI_API_KEY", run_process.call_args.kwargs["env"])
        self.assertTrue(run_process.call_args.kwargs["capture_output"])
        self.assertTrue(run_process.call_args.kwargs["text"])
        self.assertFalse(run_process.call_args.kwargs["check"])
        self.assertFalse(run_process.call_args.kwargs["shell"])
        self.assertEqual(
            run_process.call_args.kwargs["timeout"],
            mega_coder.RUN_TIMEOUT_SECONDS,
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
