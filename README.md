# Mega Coder

Mega Coder is a Python console application that uses the OpenAI Responses API
to build and improve Python programs, analyze public GitHub repositories, and
give near-real-time coding tips from the screen.

## Features

1. **Develop Python programs** — generates, validates, runs, repairs, optimizes,
   and lints noninteractive Python code.
2. **Analyze public GitHub repositories** — uses Gitingest to explain a project
   and suggest changes without modifying the repository.
3. **Give screen-based coding tips** — captures an editor region in memory,
   recognizes code locally with OCR, and requests short suggestions from
   OpenAI.

## Technology

Python 3.12, OpenAI Python SDK, Gitingest, MSS, NumPy, RapidOCR, OpenVINO,
Pylint, Colorama, tqdm, python-dotenv, and unittest.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a local `.env` file:

```text
OPENAI_API_KEY=your-api-key
```

Never commit `.env` or `.venv`. Then start the app:

```bash
python mega_coder.py
```

OpenAI API calls may incur usage charges.

## Model choice

I chose `gpt-5-nano` because it is fast and inexpensive. For stronger code
analysis and smarter tips, consider a newer model available to your API
account, such as `gpt-5.6-terra`; it may cost more and respond more slowly.

The model is defined in constants. Change the values in both application
files if you want every feature to use another model:

```python
# mega_coder.py
MODEL = "gpt-5-nano"
REPOSITORY_MODEL = "gpt-5-nano"

# screen_coding_tips.py
SCREEN_TIPS_MODEL = "gpt-5-nano"
```

For example, replace the values with `"gpt-5.6-terra"`. Use an exact model ID
supported by your account; Mega Coder does not automatically fall back to a
different model.

## Screen-capture setup

For Option 3, open the IDE full screen, hide the sidebar, and place the editor
above the integrated terminal. On macOS, also allow the terminal or VS Code to
record the screen in **System Settings → Privacy & Security → Screen & System
Audio Recording**.

Run this command from the project folder. It reads the monitor size and prints
a ready-to-paste editor region with 120-pixel margins and a maximum height of
500 pixels:

```bash
python - <<'PY'
import mss

with mss.MSS() as capture:
    monitor = capture.primary_monitor

horizontal_margin = min(120, max(0, (monitor["width"] - 1) // 2))
top_offset = min(120, max(0, monitor["height"] - 1))
region_width = max(1, monitor["width"] - (2 * horizontal_margin))
region_height = max(1, min(500, monitor["height"] - top_offset))

print("SCREEN_CAPTURE_REGION = {")
print(f'    "left": {monitor["left"] + horizontal_margin},')
print(f'    "top": {monitor["top"] + top_offset},')
print(f'    "width": {region_width},')
print(f'    "height": {region_height},')
print("}")
PY
```

For a `1710 × 1107` monitor starting at `(0, 0)`, it prints:

```python
SCREEN_CAPTURE_REGION = {
    "left": 120,
    "top": 120,
    "width": 1470,
    "height": 500,
}
```

Paste the result over `SCREEN_CAPTURE_REGION = None` in
`screen_coding_tips.py`. It is a starting point: reduce `height` if the region
reaches the terminal, and adjust `left` or `top` if it misses part of the
editor. Personal monitor coordinates normally should not be committed.

## Safety and limitations

- Screenshots stay in memory; Option 3 sends filtered, redacted OCR text rather
  than image files.
- Generated code runs in a separate subprocess, but this is not a complete
  security sandbox. Review generated code before relying on it.
- Option 2 supports public repositories and provides advice only.
- OCR can miss small edits, while larger stable changes are more reliable.

## Tests

External services, capture, OCR, and time are mocked in automated tests:

```bash
python -m unittest discover -v
python -m pylint --persistent=no \
  mega_coder.py screen_coding_tips.py \
  test_mega_coder.py test_screen_coding_tips.py
```

## References

- [GPT-5 nano](https://developers.openai.com/api/docs/models/gpt-5-nano)
- [GPT-5.6 Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra)
- [OpenAI Responses API](https://developers.openai.com/api/docs/guides/migrate-to-responses)
