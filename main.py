import os
import time
import logging

import openai
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

from src.config import LOGS_FOLDER, DATASET_PATH
from src.tasks import classification, evaluation, evolution
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
    dataset = pd.read_csv(DATASET_PATH)

    if args.n is not None and current_round != -1:
        for i in tqdm(range(args.n - current_round)):
            logging.info(f"Starting round {i + 1} of {args.n}.")
            classification(prompt, prompt_version, dataset, args.task_model)
            accuracy = evaluation(prompt_version)
            new_prompt, new_version = evolution(prompt_version, args.modifying_model)
            prompt = new_prompt
            prompt_version = new_version
            if accuracy > current_accuracy:
                current_accuracy = accuracy
    else:
        stagnant_count = 0
        logging.info(f"Starting with patience {args.patience} and threshold {args.threshold}.")
        while stagnant_count < args.patience:
            classification(prompt, prompt_version, dataset, args.task_model)
            accuracy = evaluation(prompt_version)
            new_prompt, new_version = evolution(prompt_version, args.modifying_model)
            prompt = new_prompt
            prompt_version = new_version
            logging.info(f"Current accuracy: {accuracy}")
            if accuracy - current_accuracy > args.threshold:
                current_accuracy = accuracy
                stagnant_count = 0
                logging.info(f"Accuracy improved. Resetting stagnant count.")
            else:
                stagnant_count += 1
                logging.info(f"Accuracy did not improve. Stagnant count: {stagnant_count}")

    logging.info(f"Finished. Best accuracy: {current_accuracy}. Last prompt version: {prompt_version}.")
