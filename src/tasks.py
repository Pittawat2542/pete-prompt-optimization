import json
import os

import pandas as pd
from tqdm import tqdm

from src.config import RAW_FOLDER, PARSED_FOLDER, EVALUATION_FOLDER
from src.models import chat_model
from src.utils import parse_json_output, sleep


def classification(prompt: str, prompt_version: int, dataset: pd.DataFrame, model=None):
    output_raw_path = os.path.join(RAW_FOLDER, str(prompt_version))
    output_parsed_path = os.path.join(PARSED_FOLDER, str(prompt_version))

    if not os.path.exists(output_raw_path):
        os.makedirs(output_raw_path)

    if not os.path.exists(output_parsed_path):
        os.makedirs(output_parsed_path)

        for index, message in tqdm(zip(dataset['index'], dataset['text']), desc="Classification"):
            raw_file_path = os.path.join(output_raw_path, f'{index}.txt')
            parsed_file_path = os.path.join(output_parsed_path, f'{index}.json')

            if os.path.exists(raw_file_path) and os.path.exists(parsed_file_path):
                continue

            response = chat_model(prompt.replace('<message>', message), model)
            with open(raw_file_path, 'w') as raw_file:
                raw_file.write(response)

            parsed_response = parse_json_output(response)
            with open(parsed_file_path, 'w') as parsed_file:
                parsed_response['prompt_version'] = str(prompt_version)
                parsed_response['label'] = dataset[dataset['index'] == index]['label'].values[0]
                parsed_response['model'] = model
                json.dump(parsed_response, parsed_file, indent=2)

            sleep(3)


def evaluation(prompt_version: int):
    parsed_folder_path = os.path.join(PARSED_FOLDER, str(prompt_version))
    files = [file for file in os.listdir(parsed_folder_path) if file.endswith('.json')]
    total = len(files)
    correct_messages = []
    incorrect_messages = []

    for file in tqdm(files, desc="Evaluation"):
        parsed_file_path = os.path.join(parsed_folder_path, file)
        with open(parsed_file_path, 'r') as f:
            data = json.load(f)
            if data['predicted'] == data['label']:
                correct_messages.append(data)
            else:
                incorrect_messages.append(data)

    with open(os.path.join(EVALUATION_FOLDER, f'{prompt_version}.json'), 'w') as f:
        json.dump({
            'total': total,
            'correct': len(correct_messages),
            'incorrect': len(incorrect_messages),
            'accuracy': len(correct_messages) / total,
            'correct_messages': correct_messages,
            'incorrect_messages': incorrect_messages
        }, f, indent=2)


def evolution():
    pass
