import json
import os
import time

import pandas as pd

from src.config import OUTPUTS_FOLDER, LOGS_FOLDER, STATS_FOLDER, PARSED_FOLDER, RAW_FOLDER, EVALUATION_FOLDER, \
    DATASET_PATH


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


def prepare_output_folders():
    """Create the output folders if they don't exist."""
    if not os.path.exists(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER)

    if not os.path.exists(LOGS_FOLDER):
        os.makedirs(LOGS_FOLDER)

    if not os.path.exists(STATS_FOLDER):
        os.makedirs(STATS_FOLDER)

    if not os.path.exists(PARSED_FOLDER):
        os.makedirs(PARSED_FOLDER)

    if not os.path.exists(RAW_FOLDER):
        os.makedirs(RAW_FOLDER)

    if not os.path.exists(EVALUATION_FOLDER):
        os.makedirs(EVALUATION_FOLDER)


def prepare_dataset():
    """Prepare the dataset for the classification task."""
    df = pd.read_csv(DATASET_PATH)
    df['label'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values.tolist()
    df['label'] = df['label'].apply(
        lambda x: 'clean' if x == [0, 0, 0, 0, 0, 0] else
        'toxic' if x == [1, 0, 0, 0, 0, 0] else
        'severe_toxic' if x == [0, 1, 0, 0, 0, 0] else
        'obscene' if x == [0, 0, 1, 0, 0, 0] else
        'threat' if x == [0, 0, 0, 1, 0, 0] else
        'insult' if x == [0, 0, 0, 0, 1, 0] else
        'identity_hate')

    return df
