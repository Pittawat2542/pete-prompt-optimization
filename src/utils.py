import json
import time


def sleep(seconds: int) -> None:
    """Sleep for a given number of seconds."""
    time.sleep(seconds)


def parse_json_output(raw_text: str) -> dict:
    """Parse the raw text output from OpenAI API into a dictionary."""
    if "```json" in raw_text:
        raw_text = raw_text.split("```json")[1].split("```")[0]
    try:
        return json.loads(raw_text, strict=False)
    except Exception as e:
        print(f"Unexpected error: {e}")
