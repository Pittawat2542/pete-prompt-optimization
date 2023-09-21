import argparse
import json
import os
import time
import tiktoken

import pandas as pd

from src.config import OUTPUTS_FOLDER, LOGS_FOLDER, STATS_FOLDER, PARSED_FOLDER, RAW_FOLDER, EVALUATION_FOLDER, \
    DATASET_PATH, PROMPTS_PATH, ORIGINAL_PROMPT_PATH, MODIFYING_PROMPT_PATH


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


def parse_arguments():
    """Parse the arguments passed to the program."""
    parser = argparse.ArgumentParser(
        prog='Prompt Optimization with LLMs')

    parser.add_argument("-mt", "--task-model", type=str,
                        default="gpt-3.5-turbo",
                        help='The model to be used for the task.',
                        choices=["gpt-3.5-turbo", "gpt-4"], )
    parser.add_argument("-mm", "--modifying-model", type=str,
                        default="gpt-3.5-turbo",
                        help='The model to be used for modifying the prompt.',
                        choices=["gpt-3.5-turbo", "gpt-4"], )
    parser.add_argument("-p", "--patience", type=int,
                        help='The number of rounds to wait before stopping the optimization.')
    parser.add_argument("-t", "--threshold", type=float,
                        help='The accuracy threshold to stop the optimization.')
    parser.add_argument("-n", "--n", type=int,
                        help='The number of evolution round to run. (Either "number of evolution" or (threshold and '
                             'patience) must be specified.)')

    _args = parser.parse_args()

    if _args.n is None and (_args.threshold is None or _args.patience is None):
        raise ValueError("Either the number of evolution round or the threshold and patience must be specified.")

    if _args.n is not None and (_args.threshold is not None or _args.patience is not None):
        raise ValueError("Either the number of evolution round or the threshold and patience must be specified.")

    if _args.n is not None and _args.n <= 0:
        raise ValueError("The number of evolution round must be greater than 0.")

    if _args.threshold is not None and _args.threshold <= 0:
        raise ValueError("The threshold must be greater than 0.")

    if _args.patience is not None and _args.patience <= 0:
        raise ValueError("The patience must be greater than 0.")

    return _args


def load_previous_values(args):
    """Load the previous values from the stat.json file."""
    last_accuracy = 0
    last_prompt_version = 0
    last_round = -1

    if os.path.exists(f'{STATS_FOLDER}/stat.json'):
        stat_obj = json.load(open(f'{STATS_FOLDER}/stat.json', 'r'))
        last_accuracy = stat_obj['last_accuracy']
        last_prompt_version = stat_obj['last_prompt_version']

        if stat_obj.get('patience', None) is not None:
            args.patience = stat_obj['patience']
        if stat_obj.get('threshold', None) is not None:
            args.threshold = stat_obj['threshold']
        if stat_obj.get('n', None) is not None:
            args.threshold = stat_obj['n']
            last_round = stat_obj['last_round']
        args.task_model = stat_obj['task_model']
        args.modifying_model = stat_obj['modifying_model']

        with open(PROMPTS_PATH, 'r') as f:
            prompts = json.load(f)["prompts"]
            last_prompt = [p for p in prompts if p["version"] == last_prompt_version][0]["prompt"]
    else:
        with open(f'{STATS_FOLDER}/stat.json', 'w') as f:
            stat_obj = {
                "last_accuracy": last_accuracy,
                "best_accuracy": last_accuracy,
                "last_prompt_version": last_prompt_version,
                "best_prompt_version": last_prompt_version,
                "task_model": args.task_model,
                "modifying_model": args.modifying_model,
            }

            if args.threshold is not None and args.patience is not None:
                stat_obj["threshold"] = args.threshold
                stat_obj["patience"] = args.patience
            else:
                stat_obj["n"] = args.n
                last_round = 0
                stat_obj["last_round"] = last_round

            f.write(json.dumps(stat_obj))

            last_prompt = open(ORIGINAL_PROMPT_PATH, 'r').read()

            with open(PROMPTS_PATH, 'w') as f2:
                f2.write(json.dumps({
                    "prompts": [{
                        "version": last_prompt_version,
                        "prompt": last_prompt
                    }]
                }))

    return last_accuracy, last_prompt_version, last_prompt, last_round


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text."""
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    return len(tokens)


def get_prompt_by_version(prompt_version: int) -> str:
    """Get the prompt by its version."""
    with open(PROMPTS_PATH, 'r') as f:
        prompts = json.load(f)["prompts"]
        return [p for p in prompts if p["version"] == prompt_version][0]["prompt"]


def get_modifying_prompt() -> str:
    """Get the modifying prompt."""
    with open(MODIFYING_PROMPT_PATH, 'r') as f:
        prompt = f.read()
        return prompt


def get_evaluation_results_by_version(prompt_version: int) -> dict:
    """Get the evaluation by its version."""
    with open(EVALUATION_FOLDER / f'{prompt_version}.json', 'r') as f:
        return json.load(f)
