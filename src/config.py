from pathlib import Path

DATASET_PATH = Path('data/sampled.csv')
ORIGINAL_PROMPT_PATH = Path('prompts/original_prompt.txt')
MODIFYING_PROMPT_PATH = Path('prompts/modifying_prompt.txt')
PROMPTS_PATH = Path('prompts') / "prompts.json"
OUTPUTS_FOLDER = Path('outputs')
STATS_FOLDER = OUTPUTS_FOLDER / 'stats'
STATS_FILE_PATH = STATS_FOLDER / 'stats.json'
LOGS_FOLDER = OUTPUTS_FOLDER / 'logs'
RAW_FOLDER = OUTPUTS_FOLDER / 'raw'
PARSED_FOLDER = OUTPUTS_FOLDER / 'parsed'
EVALUATION_FOLDER = OUTPUTS_FOLDER / 'evaluation'
MAX_TOKENS = {
    "gpt-3.5-turbo": 4097,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-3.5-turbo-instruct": 4097,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
}
