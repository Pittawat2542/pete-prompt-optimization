import os
import time
import logging

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from src.config import LOGS_FOLDER
from src.utils import prepare_output_folders, parse_arguments, load_previous_values

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if __name__ == "__main__":
    prepare_output_folders()

    FORMAT = '[%(asctime)s] %(message)s'
    logging.basicConfig(format=FORMAT, filename=f'{LOGS_FOLDER}/{time.strftime("%Y%m%d-%H%M%S")}.log',
                        level=logging.INFO,
                        filemode='a', datefmt='%Y-%m-%d %H:%M:%S')

    args = parse_arguments()
    current_accuracy, prompt_version, prompt, current_round = load_previous_values(args)

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
