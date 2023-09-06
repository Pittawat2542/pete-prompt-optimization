import os
import json
import time
import pandas as pd
import openai
from dotenv import load_dotenv
from tqdm import tqdm

from src.models import chat_model
from src.utils import sleep, parse_json_output

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# TODO: Compare performance optimization via GPT-3.5 and GPT-4 and base performance of GPT-3.5 and GPT-4 without optimization
# TODO: Save raw data, logging, add resume-ability, handle OpenAI error
# 1. Load the prompt
# 2. Predict the result
# 3. Evaluate (accuracy)
# 4. Sample the result for prompt optimization
# 5. Get new prompt
# 6. Loop until reach specified n or acc or stagnant

# Main prompt: Replace "<message>"
# Modifying prompt: Replace: "<sample>"
# Each entry in <sample>
# |- Message: <message>
# |- Predicted class: <prediction>
# |- Predicted reason: <reason>
# |- Ground truth: <ground_truth>


def classification():
    prompt = original_prompt
    prompt_version = 'v1'

    for index, message in zip(df['index'], df['text']):
        if os.path.exists(f'outputs/raw/{prompt_version}/{index}.txt') and os.path.exists(
                f'outputs/parsed/{prompt_version}/{index}.json'):
            continue

        response = chat_model(prompt.replace('<message>', message))
        with open(f'outputs/raw/{prompt_version}/{index}.txt', 'w') as f:
            f.write(response)

        parsed_response = parse_json_output(response)
        with open(f'outputs/parsed/{prompt_version}/{index}.json', 'w') as f:
            parsed_response['prompt_version'] = prompt_version
            parsed_response['label'] = df[df['index'] == index]['label'].values[0]
            json.dump(parsed_response, f, indent=2)

        sleep(5)


def evaluation(prompt_version: str):
    files = [file for file in os.listdir(f'outputs/parsed/{prompt_version}') if file.endswith('.json')]
    total = len(files)
    correct = 0

    for file in tqdm(files):
        with open(f'outputs/parsed/{prompt_version}/{file}', 'r') as f:
            data = json.load(f)
            if data['predicted'] == data['label']:
                correct += 1


def evolution():
    pass


if __name__ == "__main__":
    df = pd.read_csv('GGWP-Toxic-Behavior/data/labeled/combined.csv')
    df['label'] = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values.tolist()
    df['label'] = df['label'].apply(
        lambda x: 'clean' if x == [0, 0, 0, 0, 0, 0] else 'toxic' if x == [1, 0, 0, 0, 0,
                                                                           0] else 'severe_toxic' if x == [0,
                                                                                                           1,
                                                                                                           0,
                                                                                                           0,
                                                                                                           0,
                                                                                                           0] else 'obscene' if x == [
            0, 0, 1, 0, 0, 0] else 'threat' if x == [0, 0, 0, 1, 0, 0] else 'insult' if x == [0, 0, 0, 0, 1,
                                                                                              0] else 'identity_hate')

    with open('prompts/original_prompt.txt', 'r') as f:
        original_prompt = f.read()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    if not os.path.exists('outputs/parsed'):
        os.makedirs('outputs/parsed')

    if not os.path.exists('outputs/raw'):
        os.makedirs('outputs/raw')
