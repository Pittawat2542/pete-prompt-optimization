from pathlib import Path

DATASET_PATH = Path('GGWP-Toxic-Behavior/data/labeled/combined.csv')
ORIGINAL_PROMPT_PATH = Path('prompts/original_prompt.txt')
MODIFIED_PROMPT_PATH = Path('prompts/modifying_prompt.txt')
PROMPTS_PATH = Path('prompts') / "prompts.json"
OUTPUTS_FOLDER = Path('outputs')
STATS_FOLDER = OUTPUTS_FOLDER / 'stats'
LOGS_FOLDER = OUTPUTS_FOLDER / 'logs'
RAW_FOLDER = OUTPUTS_FOLDER / 'raw'
PARSED_FOLDER = OUTPUTS_FOLDER / 'parsed'
EVALUATION_FOLDER = OUTPUTS_FOLDER / 'evaluation'
