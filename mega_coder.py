"""Mega Coder console application."""

import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import OpenAI, OpenAIError


MENU = """I’m Mega Coder. What would you like me to do today?

1. Develop a python program.
2. Fix/change something in a Github repository.
3. Look at my screen and give me realtime coding tips."""

MODEL = "gpt-5-nano"
OUTPUT_FILE = Path(__file__).with_name("generated-code-openai.py")
RUN_TIMEOUT_SECONDS = 30

FORBIDDEN_MODULES = {
    "aiohttp",
    "builtins",
    "fileinput",
    "ftplib",
    "getpass",
    "glob",
    "http.client",
    "httpx",
    "importlib",
    "os",
    "pathlib",
    "requests",
    "runpy",
    "shutil",
    "smtplib",
    "socket",
    "subprocess",
    "tempfile",
    "urllib",
    "webbrowser",
}
FORBIDDEN_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "getenv",
    "getpass",
    "input",
    "open",
    "os.getenv",
    "os.popen",
    "os.system",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.run",
}
FORBIDDEN_ATTRIBUTES = {
    "os.environ",
    "os.getenv",
    "os.popen",
    "os.system",
    "subprocess.Popen",
    "subprocess.call",
    "subprocess.check_call",
    "subprocess.check_output",
    "subprocess.run",
    "sys.argv",
}


class GeneratedCodeValidationError(ValueError):
    """Raised when generated source code fails the static safety checks."""


@dataclass
class ExecutionResult:
    """Store complete details from one generated-program execution."""

    success: bool
    returncode: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool


def show_menu():
    """Display the Mega Coder menu."""
    print(MENU)


def ask_for_program_description():
    """Ask for and return a non-empty program description."""
    print("Describe me which python program you want me to develop")

    while True:
        description = input().strip()
        if description:
            return description
        print("The program description cannot be empty. Please try again.")


def build_generation_prompt(program_description):
    """Build the instructions for generating the requested Python program."""
    return f"""Create a Python program with this description:

{program_description}

Follow every requirement below:
- Return only raw, runnable Python source code.
- Do not include Markdown fences.
- Do not include explanations outside the code.
- Do not accept command-line arguments or use sys.argv.
- Do not call input() or otherwise pause or wait for user interaction.
- Use embedded example data so the program runs from beginning to end.
- Include meaningful assert statements that test the program's core logic.
- Ensure the assertions execute automatically when the generated file runs.
- Print a short, useful demonstration after the assertions pass.
- Keep the program self-contained and use only the Python standard library.
- Do not access the internet, network, URLs, websites, APIs, remote services,
  or external sources.
- Do not import networking modules or packages such as requests, httpx,
  aiohttp, urllib, or socket.
- Do not read environment variables, credentials, secrets, or API keys.
- Do not run subprocesses, shell commands, or other programs.
- Do not read, create, modify, or delete files.
"""


def call_openai(prompt):
    """Send a generation prompt to OpenAI and return its text response."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is not set. Set it in your environment before "
            "using option 1."
        )

    client = OpenAI()
    response = client.responses.create(
        model=MODEL,
        input=prompt,
    )
    return response.output_text


def clean_generated_code(model_output):
    """Remove an optional Markdown fence and return normalized source code."""
    code = model_output.strip()
    if not code:
        raise ValueError("OpenAI returned an empty response.")

    lines = code.splitlines()
    opening_fences = {"```python", "```py", "```"}
    if lines[0].strip().lower() in opening_fences:
        if lines[-1].strip() != "```":
            raise ValueError("OpenAI returned code with an unclosed Markdown fence.")
        code = "\n".join(lines[1:-1]).strip()

    if not code:
        raise ValueError("OpenAI returned an empty response.")

    return code.rstrip() + "\n"


def _is_forbidden_module(module_name):
    """Return whether a module or one of its submodules is forbidden."""
    return any(
        module_name == forbidden or module_name.startswith(f"{forbidden}.")
        for forbidden in FORBIDDEN_MODULES
    )


def _dotted_name(node):
    """Return a dotted name such as os.environ from an AST expression."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value

    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _resolve_import_alias(dotted_name, import_aliases):
    """Replace the first part of a dotted name with its imported module name."""
    first_part, separator, remaining_parts = dotted_name.partition(".")
    resolved_first_part = import_aliases.get(first_part, first_part)
    if separator:
        return f"{resolved_first_part}.{remaining_parts}"
    return resolved_first_part


def _validate_imports(tree):
    """Reject forbidden imports and return names used as import aliases."""
    import_aliases = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules = [alias.name for alias in node.names]
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[0]
                import_aliases[local_name] = alias.name if alias.asname else local_name
        elif isinstance(node, ast.ImportFrom):
            parent_module = node.module or ""
            imported_modules = [parent_module]
            for alias in node.names:
                imported_name = (
                    f"{parent_module}.{alias.name}" if parent_module else alias.name
                )
                imported_modules.append(imported_name)
                if imported_name in FORBIDDEN_ATTRIBUTES:
                    raise GeneratedCodeValidationError(
                        f"Importing '{imported_name}' is not allowed."
                    )
                if alias.name != "*":
                    import_aliases[alias.asname or alias.name] = imported_name
        else:
            continue

        for module_name in imported_modules:
            if module_name and _is_forbidden_module(module_name):
                raise GeneratedCodeValidationError(
                    f"Importing '{module_name}' is not allowed."
                )

    return import_aliases


def validate_generated_code(code):
    """Apply best-effort checks; this validation is not a security sandbox."""
    if not code.strip():
        raise GeneratedCodeValidationError("Generated source code is empty.")

    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        raise GeneratedCodeValidationError(
            f"Invalid Python syntax on line {error.lineno}: {error.msg}."
        ) from None

    import_aliases = _validate_imports(tree)

    has_assertion = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            has_assertion = True

        if isinstance(node, ast.Call):
            call_name = _resolve_import_alias(
                _dotted_name(node.func), import_aliases
            )
            if call_name in FORBIDDEN_CALLS:
                raise GeneratedCodeValidationError(
                    f"Calling '{call_name}' is not allowed."
                )

        if isinstance(node, ast.Attribute):
            attribute_name = _resolve_import_alias(
                _dotted_name(node), import_aliases
            )
            if any(
                attribute_name == forbidden
                or attribute_name.startswith(f"{forbidden}.")
                for forbidden in FORBIDDEN_ATTRIBUTES
            ):
                raise GeneratedCodeValidationError(
                    f"Accessing '{attribute_name}' is not allowed."
                )

    if not has_assertion:
        raise GeneratedCodeValidationError(
            "Generated code must contain at least one assert statement."
        )


def write_generated_code(code):
    """Overwrite the generated Python file using UTF-8."""
    OUTPUT_FILE.write_text(code, encoding="utf-8")


def _output_as_text(output):
    """Normalize captured subprocess output to text."""
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def run_generated_program(file_path=OUTPUT_FILE, timeout=RUN_TIMEOUT_SECONDS):
    """Run generated code in a child process and return all result details."""
    program_path = Path(file_path).resolve()
    child_environment = os.environ.copy()
    child_environment.pop("OPENAI_API_KEY", None)

    try:
        completed_process = subprocess.run(
            [sys.executable, str(program_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
            cwd=str(program_path.parent),
            env=child_environment,
        )
    except subprocess.TimeoutExpired as error:
        return ExecutionResult(
            success=False,
            returncode=None,
            stdout=_output_as_text(error.stdout),
            stderr=_output_as_text(error.stderr),
            timed_out=True,
        )
    except OSError as error:
        return ExecutionResult(
            success=False,
            returncode=None,
            stdout="",
            stderr=str(error),
            timed_out=False,
        )

    return ExecutionResult(
        success=completed_process.returncode == 0,
        returncode=completed_process.returncode,
        stdout=completed_process.stdout,
        stderr=completed_process.stderr,
        timed_out=False,
    )


def _print_captured_output(heading, output):
    """Print captured output under a heading without adding extra blank lines."""
    if output:
        print(heading)
        print(output, end="" if output.endswith("\n") else "\n")


def report_execution_result(result, timeout=RUN_TIMEOUT_SECONDS):
    """Print a clear summary of a generated program's execution result."""
    _print_captured_output("Generated program output:", result.stdout)

    if result.success:
        print("Generated program and its assertions completed successfully.")
        return

    if result.timed_out:
        print(f"Generated program failed: execution exceeded {timeout} seconds.")
    elif result.returncode is not None:
        print(f"Generated program failed with return code {result.returncode}.")
    else:
        print("Generated program could not be started.")

    _print_captured_output("Generated program errors:", result.stderr)


def develop_program():
    """Generate, validate, save, run, and report the requested program."""
    program_description = ask_for_program_description()
    prompt = build_generation_prompt(program_description)

    try:
        model_output = call_openai(prompt)
        generated_code = clean_generated_code(model_output)
        validate_generated_code(generated_code)
        write_generated_code(generated_code)
    except OpenAIError:
        print("The OpenAI request failed. Check your connection and API settings.")
    except GeneratedCodeValidationError as error:
        print(f"Generated code failed validation: {error}")
    except ValueError as error:
        print(f"Could not generate code: {error}")
    except OSError as error:
        print(f"Could not write {OUTPUT_FILE.name}: {error}")
    else:
        print(f"Generated code saved to {OUTPUT_FILE.name}")
        print("Running the generated program...")
        execution_result = run_generated_program(OUTPUT_FILE)
        report_execution_result(execution_result)


def main():
    """Run the Mega Coder menu."""
    try:
        while True:
            show_menu()
            choice = input().strip()

            if choice == "1":
                develop_program()
                return

            if choice in {"2", "3"}:
                print("Not implemented yet")
            else:
                print("Please choose 1, 2, or 3.")
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
