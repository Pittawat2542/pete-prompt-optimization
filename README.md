# Prompt Evolution Through Examples for Large Language Models–A Case Study in Game Comment Toxicity Classification ([Paper](https://ieeexplore.ieee.org/document/10584130))

This repository contains the code and datasets for the paper "Prompt Evolution Through Examples for Large Language Models–A Case Study in Game Comment Toxicity Classification" accepted at [IEEE MetroInd4.0&IoT 2024](https://www.metroind40iot.org).

## Authors
Pittawat Taveekitworachai, Febri Abdullah, Mustafa Can Gursesli, Antonio Lanata, Andrea Guazzini, and Ruck Thawonmas

## Abstract

This paper presents a novel approach for automatic prompt optimization (APO) using a large language model (LLM) as an optimizer, named Prompt Evolution Through Examples (PETE). The approach draws inspiration from evolutionary computation for the prompt evolution stages. We aim to aid in developing prompts for use in systems classifying toxic content including game community moderator-assist tools. While traditional approaches are useful for developing these tools, they have various shortcomings where LLMs can potentially mitigates these issues. LLMs accept prompts as inputs to condition generated outputs. However, to design a prompt with the best performance in this task, fine-grained adjustments are usually required and should be automated through the APO process instead of a manual approach, which is often time-consuming. In this study, ChatGPT and GPT-4 are utilized as both task performers and prompt optimizers for comparisons across models. The results indicate that PETE improves the performance of the target task up to 56.14% from a performance of an initial prompt, compared to only up to 49.15% using a standard mutation evolution. Optimized prompts are provided for future utilization in other game community moderation tools. We also recommend that future studies explore more cost-effective approaches for evaluation using LLMs to enhance the benefits of APO.
## File structure
```
.
├── data # A folder containing the sampled game comment toxicity dataset.
├── experiment_results # Results obtained from the experiment
│   ├── best_prompts
│   ├── random
│   └── with-reasons
├── experiment_results.zip # Zipped version of the experimental results
├── main.py # Index file of the program
├── prompts # Prompts used for the optimization process
│   ├── original_prompt.txt
│   ├── random_modifying_prompt.txt
│   └── with_reasons_modifying_prompt.txt
├── requirements.txt # Dependency list
├── scripts # Scripts for data analysis
│   ├── data_analysis.ipynb
│   ├── data_preparation.ipynb
│   ├── dataset_distribution.png
│   ├── dataset_distribution_pie.png
│   ├── random_optimization_graph.png
│   └── reasons_optimization_graph.png
└── src # Main logic of the program
    ├── __init__.py
    ├── config.py # Change additional configuration here
    ├── models.py
    ├── tasks.py
    └── utils.py
```

## Installation and Usage
0. Create a virtual environment (if needed):
```bash
conda create -n pete python=3.11
```
and activate it:
```bash
conda activate pete
```
1. Copy `.env.example` and rename it to `.env`. Follow instructions on [this page](https://platform.openai.com/docs/api-reference/authentication) to obtain your own OpenAI API key.
2. Install the requirements:
```bash
pip install -r requirements.txt
```
3. For Python file, run it by executing `python <filename>.py`. For Jupyter Notebook (`.ipynb`), open it with Jupyter Notebook and run it.

Options for the main program are
- `-mt`, `--task-model`: a task model used to perform such a task. (Default: `gpt-3.5-turbo` or `gpt-4`)
- `-mm`, `--modifying-model`: a model responsible for optimizing a prompt (Default: `gpt-3.5-turbo` or `gpt-4`)
- `-p`, `--patience`: The number of rounds to wait before stopping the optimization. (Applicable when not using `n`.)
- `-t`, `--threshold`: The accuracy threshold to stop the optimization. (Applicable when not using `n`.)
- `-n`, `--n`: The number of evolution round to run. (Applicable when not using both `-p` and `-t`.) This is a default mode used for the experiment.
- `-e`, `--evolution`: Evolution strategy. (Deafult: `with-reasons` or `random`)
