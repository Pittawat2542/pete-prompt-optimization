import argparse
import os
import json
import time
import logging

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from src.config import LOGS_FOLDER, STATS_FOLDER, PROMPTS_PATH, ORIGINAL_PROMPT_PATH
from src.utils import prepare_output_folders

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def parse_arguments():
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


def load_previous_values():
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


if __name__ == "__main__":
    prepare_output_folders()

    FORMAT = '[%(asctime)s] %(message)s'
    logging.basicConfig(format=FORMAT, filename=f'{LOGS_FOLDER}/{time.strftime("%Y%m%d-%H%M%S")}.log',
                        level=logging.INFO,
                        filemode='a', datefmt='%Y-%m-%d %H:%M:%S')

    args = parse_arguments()

    current_accuracy, prompt_version, prompt, current_round = load_previous_values()

    if args.n is not None and current_round != -1:
        for i in tqdm(range(args.n - current_round)):
            logging.info(f"Starting round {i + 1} of {args.n}.")
    else:
        stagnant_count = 0
        while stagnant_count < args.patience:
            accuracy = 0
            if accuracy - current_accuracy > args.threshold:
                current_accuracy = accuracy
                stagnant_count = 0
            else:
                stagnant_count += 1

                if stagnant_count >= args.patience:
                    break

# TODO: When return, return best performing prompt, not the last one when finish the program
# TODO: Compare performance optimization via GPT-3.5 and GPT-4 and base performance of GPT-3.5 and GPT-4 without optimization
# TODO: Add logging
# 1. Load the prompt [/]
# 2. Predict the result [/]
# 3. Evaluate (accuracy) [/]
# 4. Sample the result for prompt optimization
# 5. Get new prompt
# 6. Loop until reach specified n or acc or stagnant [/]

# Main prompt: Replace "<message>"
# Modifying prompt: Replace: "<sample>"
# Each entry in <sample>
# |- Message: <message>
# |- Predicted class: <prediction>
# |- Predicted reason: <reason>
# |- Ground truth: <ground_truth>
