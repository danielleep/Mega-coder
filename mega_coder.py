"""Mega Coder console application."""

import os
from pathlib import Path

from openai import OpenAI, OpenAIError


MENU = """I’m Mega Coder. What would you like me to do today?

1. Develop a python program.
2. Fix/change something in a Github repository.
3. Look at my screen and give me realtime coding tips."""

MODEL = "gpt-5-nano"
OUTPUT_FILE = Path(__file__).with_name("generated-code-openai.py")


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


def write_generated_code(code):
    """Overwrite the generated Python file using UTF-8."""
    OUTPUT_FILE.write_text(code, encoding="utf-8")


def develop_program():
    """Generate, clean, and save the Python program requested by the user."""
    program_description = ask_for_program_description()
    prompt = build_generation_prompt(program_description)

    try:
        model_output = call_openai(prompt)
        generated_code = clean_generated_code(model_output)
        write_generated_code(generated_code)
    except OpenAIError:
        print("The OpenAI request failed. Check your connection and API settings.")
    except ValueError as error:
        print(f"Could not generate code: {error}")
    except OSError as error:
        print(f"Could not write {OUTPUT_FILE.name}: {error}")
    else:
        print(f"Generated code saved to {OUTPUT_FILE.name}")


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
