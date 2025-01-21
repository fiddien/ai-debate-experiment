"""
Main script to run the debate experiments with the BoardgameQA dataset.
"""

import json
import logging
import multiprocessing as mp
import os
import random
import re
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Tuple

from src.debate.baseline import BaselineManager
from src.debate.debate import DebateTwoPlayers
from src.debate.judge import JudgeManager
from src.debate.types import DebateScenario

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


# Dataset difficulty levels
LEVELS = [
    "ZeroConflict",
    "LowConflict",
    "Main",  # Medium
    "HighConflict",
]


def load_boardgame_qa(base_path: str = "BoardgameQA") -> dict:
    """Load BoardgameQA dataset from json files."""
    ds = {}
    for level in LEVELS:
        path = Path(base_path) / f"BoardgameQA-{level}-depth2/test.json"
        with open(path, "r", encoding="utf-8") as f:
            ds[level] = json.load(f)
    return ds


def sample_data(ds: dict, sample_size: float = 0.052) -> dict:
    """Sample data while maintaining label distribution."""
    sampled_data = {}
    for level, data in ds.items():
        label_to_examples = defaultdict(list)

        for example in data:
            label_to_examples[example["label"]].append(example)

        sample = []
        for exs in label_to_examples.values():
            sample_size_per_label = int(len(exs) * sample_size)
            sample.extend(random.sample(exs, sample_size_per_label))

        sampled_data[level] = sample
    display_total(sampled_data)
    return sampled_data


def convert_to_scenarios(examples: list) -> list[DebateScenario]:
    """Convert BoardgameQA examples to DebateScenario objects."""

    scenarios = []
    for ex in examples:
        situation, question = split_question(ex["example"])
        scenario = DebateScenario(
            situation=situation,
            question=question,
            answer_options=["proved", "disproved", "unknown"],
            label=ex["label"],
        )
        scenarios.append(scenario)
    return scenarios


def display_total(ds: dict):
    """Display the total number of examples per level."""
    print("Total examples per level:")
    for level, data in ds.items():
        print(f"{level}: {len(data)}")


def save_sampled_data(data: dict, filepath: str):
    """Save sampled dataset to JSONL file."""
    Path(filepath).parent.mkdir(exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for level, examples in data.items():
            for example in examples:
                example["level"] = level  # Add level info to each example
                f.write(json.dumps(example) + "\n")


def load_sampled_data(filepath: str) -> dict:
    """Load sampled dataset from JSONL file."""
    try:
        data = defaultdict(list)
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line.strip())
                level = example.pop("level")  # Remove and get level info
                data[level].append(example)
        return dict(data)
    except FileNotFoundError:
        return None


def load_scenarios_stream(filepath: str) -> Generator[DebateScenario, None, None]:
    """Load and yield scenarios one at a time from JSONL file."""
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line.strip())
            situation, question = split_question(example["example"])
            yield DebateScenario(
                situation=situation,
                question=question,
                answer_options=["proved", "disproved", "unknown"],
                label=example["label"],
            )


def split_question(example: str) -> Tuple[str, str]:
    """Split the example text into the game and the question."""
    example_text = example
    question = re.split(r"\.\s+", example_text)[-1]
    game = example_text.replace(question, "").strip()
    return game, question


def process_single_scenario(
    scenario: DebateScenario,
    debater_models: List[str],
    judge_models: List[str],
) -> Dict:
    """Process a single scenario and return all results."""
    results = {"baseline": {}, "debates": [], "judgments": {}}

    # Run baseline
    baseline = BaselineManager(scenario=scenario, models=judge_models)
    baseline.run()
    results["baseline"] = baseline.get_results()

    # Run debate with different variants
    debate = DebateTwoPlayers(
        scenario=scenario,
        debater_models=debater_models,
        word_limit=300,
        max_debate_rounds=3,
    )

    for variant in [
        {"swap": False, "all_wrong": False},
        {"swap": True, "all_wrong": False},
        {"swap": False, "all_wrong": True},
    ]:
        # Run debate and get record
        record = debate.run(**variant)
        results["debates"].append(record.to_dict())

        # Run judgment immediately for this record
        judge = JudgeManager(record=record, judge_models=judge_models)
        judge_results = judge.run()

        # Merge judgment results
        for model, judgments in judge_results.items():
            if model not in results["judgments"]:
                results["judgments"][model] = []
            results["judgments"][model].extend(judgments)

    return results


def save_batch_results(results: List[Dict], batch_num: int, output_dir: str):
    """Save batch results to files."""
    Path(output_dir).mkdir(exist_ok=True)

    # Combine results from all scenarios in batch
    combined = {"baseline": {}, "debates": [], "judgments": {}}

    for r in results:
        combined["debates"].extend(r["debates"])
        # Merge baseline and judgment results
        for key in ["baseline", "judgments"]:
            for model, items in r[key].items():
                if model not in combined[key]:
                    combined[key][model] = []
                combined[key][model].extend(items)

    # Save each result type
    for key, content in combined.items():
        filename = Path(output_dir) / f"{key}_batch_{batch_num}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(content, f)


def process_batch(
    scenarios: List[DebateScenario],
    debater_models: List[str],
    judge_models: List[str],
    batch_num: int,
    output_dir: str,
    num_workers: int = None,
) -> None:
    """Process a batch of scenarios in parallel."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    # Process scenarios in parallel
    with mp.Pool(num_workers) as pool:
        process_func = partial(
            process_single_scenario,
            debater_models=debater_models,
            judge_models=judge_models,
        )
        results = pool.map(process_func, scenarios)

    # Save combined results
    save_batch_results(results, batch_num, output_dir)


def process_scenarios_in_batches(
    scenarios: Iterator[DebateScenario],
    debater_models: List[str],
    judge_models: List[str],
    batch_size: int = 5,
    output_dir: str = "results",
    num_workers: int = None,
) -> None:
    """Process scenarios in small batches with parallel processing."""
    Path(output_dir).mkdir(exist_ok=True)

    batch = []
    batch_num = 0

    for scenario in scenarios:
        batch.append(scenario)

        if len(batch) >= batch_size:
            process_batch(
                batch, debater_models, judge_models, batch_num, output_dir, num_workers
            )
            batch = []
            batch_num += 1

    # Process remaining scenarios
    if batch:
        process_batch(
            batch, debater_models, judge_models, batch_num, output_dir, num_workers
        )


# Model configurations
MODEL_CONFIGS = {
    "config1": {  # Same capability
        "debater_models": ["claude-3-haiku-20240307", "claude-3-haiku-20240307"],
        "judge_models": ["claude-3-haiku-20240307"]
    },
    # "config2": {  # Stronger judge
    #     "debater_models": ["claude-3-haiku-20240307", "claude-3-haiku-20240307"],
    #     "judge_models": ["claude-3-sonnet-20240229"]
    # },
    # "config3": {  # Stronger debaters
    #     "debater_models": ["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"],
    #     "judge_models": ["claude-3-haiku-20240307"]
    # }
}

if __name__ == "__main__":

    SAMPLED_DATA_PATH = "data/sampled_boardgame_qa.jsonl"  # Changed extension to .jsonl
    # Check if sampled data exists
    if not os.path.exists(SAMPLED_DATA_PATH):
        save_sampled_data(sample_data(load_boardgame_qa()), SAMPLED_DATA_PATH)

    # Run experiments for each configuration
    for config_name, models in MODEL_CONFIGS.items():
        print(f"\nRunning experiments for {config_name}")

        # Set up models for this configuration
        DEBATER_MODELS = models["debater_models"]
        JUDGE_MODELS = models["judge_models"]

        # Set up output directory for this configuration
        OUTPUT_DIR = f"results/{config_name}"


        # Process scenarios in streaming fashion with parallel processing
        scenarios_stream = load_scenarios_stream(SAMPLED_DATA_PATH)
        process_scenarios_in_batches(
            scenarios_stream,
            DEBATER_MODELS,
            JUDGE_MODELS,
            batch_size=5,
            output_dir=OUTPUT_DIR,
            num_workers=4,
        )
