import json
import logging
import os
import random

import pandas as pd
from tqdm import tqdm

from src.config import RAW_FOLDER, PARSED_FOLDER, EVALUATION_FOLDER, MAX_TOKENS, PROMPTS_PATH, STATS_FILE_PATH
from src.models import chat_model
from src.utils import parse_json_output, sleep, get_prompt_by_version, count_tokens, get_modifying_prompt, \
    get_evaluation_results_by_version, get_stat


def classification(prompt: str, prompt_version: int, dataset: pd.DataFrame, model=None):
    output_raw_path = os.path.join(RAW_FOLDER, str(prompt_version))
    output_parsed_path = os.path.join(PARSED_FOLDER, str(prompt_version))

    if not os.path.exists(output_raw_path):
        os.makedirs(output_raw_path)

    if not os.path.exists(output_parsed_path):
        os.makedirs(output_parsed_path)

    logging.info(f"Starting classification for prompt version: {prompt_version}.")
    for index, message in tqdm(zip(dataset['index'], dataset['text']), desc="Classification", total=len(dataset)):
        logging.info(f"Classifying message {index} of {len(dataset)}.")
        raw_file_path = os.path.join(output_raw_path, f'{index}.txt')
        parsed_file_path = os.path.join(output_parsed_path, f'{index}.json')

        if os.path.exists(raw_file_path) and os.path.exists(parsed_file_path):
            logging.info(f"Skipping message {index} of {len(dataset)} because it was already classified.")
            continue

        logging.info(f"Interacting with OpenAI API for message {index} of {len(dataset)}.")
        response = chat_model(prompt.replace('<message>', message), model, 0)
        logging.info(f"Finished interacting with OpenAI API for message {index} of {len(dataset)}.")
        with open(raw_file_path, 'w', encoding="utf-8") as raw_file:
            logging.info(f"Saving raw response for message {index} of {len(dataset)}.")
            raw_file.write(response)

        parsed_response = parse_json_output(response)
        if parsed_response is None:
            logging.info(f"Retrying message {index} of {len(dataset)} because it was not parsed.")
            sleep(3)
            response = chat_model(prompt.replace('<message>', message), model, 0)
            parsed_response = parse_json_output(response)

        with open(parsed_file_path, 'w', encoding="utf-8") as parsed_file:
            parsed_response['prompt_version'] = prompt_version
            parsed_response['label'] = dataset[dataset['index'] == index]['label'].values[0]
            parsed_response['model'] = model
            logging.info(f"Saving parsed response for message {index} of {len(dataset)}.")
            json.dump(parsed_response, parsed_file, indent=2)

        logging.info(f"Finished classifying message {index} of {len(dataset)}.")
        sleep(3)


def evaluation(prompt_version: int):
    parsed_folder_path = os.path.join(PARSED_FOLDER, str(prompt_version))
    files = [file for file in os.listdir(parsed_folder_path) if file.endswith('.json')]
    total = len(files)
    correct_messages = []
    incorrect_messages = []

    logging.info(f"Starting evaluation for prompt version: {prompt_version}.")
    for file in tqdm(files, desc="Evaluation"):
        parsed_file_path = os.path.join(parsed_folder_path, file)
        logging.info(f"Evaluating message {file} of {len(files)}.")
        with open(parsed_file_path, 'r') as f:
            try:
                data = json.load(f)
                if data['predicted'] == data['label']:
                    correct_messages.append(data)
                else:
                    incorrect_messages.append(data)
            except json.decoder.JSONDecodeError:
                logging.info(f"Evaluating message {file} of {len(files)} failed because it was not parsed.")
                incorrect_messages.append({
                    'message': '<ERROR_COULD_NOT_PARSE_JSON>',
                    'predicted': '<ERROR_PREDICTED_LABEL>',
                    'reason': '<ERROR>',
                    'label': '<ERROR_LABEL>',
                    'prompt_version': prompt_version,
                    'model': '<ERROR>'
                })
                continue

    with open(os.path.join(EVALUATION_FOLDER, f'{prompt_version}.json'), 'w', encoding="utf-8") as f:
        json.dump({
            'total': total,
            'correct': len(correct_messages),
            'incorrect': len(incorrect_messages),
            'accuracy': len(correct_messages) / total,
            'correct_messages': correct_messages,
            'incorrect_messages': incorrect_messages
        }, f, indent=2)

    stat_obj = get_stat()
    stat_obj["last_prompt_version"] = prompt_version
    stat_obj["last_accuracy"] = len(correct_messages) / total
    if stat_obj["last_accuracy"] > stat_obj["best_accuracy"]:
        stat_obj["best_accuracy"] = stat_obj["last_accuracy"]
        stat_obj["best_prompt_version"] = prompt_version

    with open(STATS_FILE_PATH, 'w', encoding="utf-8") as f:
        json.dump(stat_obj, f, indent=2)

    logging.info(f"Finished evaluation for prompt version: {prompt_version}.")
    return len(correct_messages) / total


def evolution(prompt_version: int, modifying_model: str = "gpt-3.5-turbo"):
    best_prompt_version = get_stat()["best_prompt_version"]
    logging.info(f"Starting evolution for prompt version: {prompt_version} (best prompt used: v{best_prompt_version}).")

    with open(PROMPTS_PATH, 'r+', encoding="utf-8") as f:
        prompts = json.loads(f.read())
        for prompt in prompts["prompts"]:
            if prompt["version"] == prompt_version + 1:
                logging.info(f"Prompt version: {prompt_version + 1} already exists in prompts.")
                return prompt["prompt"], prompt_version + 1

    max_tokens = MAX_TOKENS[modifying_model]
    logging.info(f"Model: {modifying_model} has max tokens: {max_tokens}.")

    current_prompt = get_prompt_by_version(best_prompt_version)
    current_prompt_token_count = count_tokens(current_prompt)
    new_prompt_token_count = current_prompt_token_count * 1.5
    logging.info(
        f"Current prompt version: {prompt_version}, Current prompt length: {len(current_prompt)}, Current prompt "
        f"token count: {current_prompt_token_count}")

    modifying_prompt = get_modifying_prompt()
    modifying_prompt = modifying_prompt.replace("<prompt>", current_prompt)
    modifying_prompt_token_count = count_tokens(modifying_prompt)

    left_tokens = max_tokens - modifying_prompt_token_count - new_prompt_token_count

    incorrect_messages = get_evaluation_results_by_version(prompt_version)["incorrect_messages"]
    incorrect_messages = [m for m in incorrect_messages if m['message'] != '<ERROR_COULD_NOT_PARSE_JSON>']

    samples = ""
    while left_tokens > 0 and len(incorrect_messages) > 0:
        sample = incorrect_messages.pop(random.randrange(len(incorrect_messages)))
        sample_text = f"---\nMessage: {sample['message']}\nPredicted class: {sample['predicted']}\nPredicted reason: {sample['reason']}\nGround truth: {sample['label']}\n---\n"
        sample_token_count = count_tokens(sample_text)
        if (left_tokens - sample_token_count) > 0:
            samples += sample_text
            left_tokens -= sample_token_count
        else:
            break
    samples = samples.strip()

    modifying_prompt.replace("<sample>", samples)

    logging.info(f"Interacting with OpenAI API for prompt version: {prompt_version}.")
    new_prompt = chat_model(modifying_prompt, modifying_model, 1)
    logging.info(f"Finished interacting with OpenAI API for prompt version: {prompt_version}.")

    new_version = prompt_version + 1
    logging.info(f"New version: {new_version}, New prompt length: {len(new_prompt)}")

    with open(PROMPTS_PATH, 'r+', encoding="utf-8") as f:
        prompts = json.loads(f.read())
        prompts["prompts"].append({
            "version": new_version,
            "prompt": new_prompt
        })
        f.seek(0)
        f.write(json.dumps(prompts, indent=2))
        f.truncate()

    logging.info(f"Finished evolution for prompt version: {prompt_version} with new prompt version: {new_version}.")
    return new_prompt, new_version
