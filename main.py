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
from enum import Enum, Flag, auto
from tqdm import tqdm

from src.debate.baseline import BaselineManager
from src.debate.debate import DebateTwoPlayers
from src.debate.judge import JudgeManager
from src.debate.types import DebateScenario

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.WARNING
)

logger = logging.getLogger(__name__)


# Dataset difficulty levels
LEVELS = [
    # "ZeroConflict",
    "LowConflict",
    # "Main",  # Medium
    "HighConflict",
]


class RunMode(Flag):
    """Run modes for the debate experiments."""

    BASELINE = auto()
    DEBATE = auto()
    JUDGE = auto()
    ALL = BASELINE | DEBATE | JUDGE


def load_boardgame_qa(base_path: str = "BoardgameQA") -> dict:
    """Load BoardgameQA dataset from json files."""
    ds = {}
    for level in LEVELS:
        path = Path(base_path) / f"BoardgameQA-{level}-depth2/test.json"
        with open(path, "r", encoding="utf-8") as f:
            ds[level] = json.load(f)
    return ds


def sample_data(ds: dict, samples_per_label: int = 20) -> dict:
    """Sample data with fixed size per label.

    Args:
        ds: Dataset dictionary with difficulty levels as keys
        samples_per_label: Number of examples to sample for each label
    """
    sampled_data = {}
    for level, data in ds.items():
        label_to_examples = defaultdict(list)
        for example in data:
            label_to_examples[example["label"]].append(example)

        sample = []
        for exs in label_to_examples.values():
            actual_samples = min(
                samples_per_label, len(exs)
            )  # Handle if we have fewer examples
            sample.extend(random.sample(exs, actual_samples))

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
    """Display the total number of examples per label and per level."""
    print("Total examples per level:")
    for level, data in ds.items():
        print(f"{level}: {len(data)}")
        for label in set([ex["label"] for ex in data]):
            print(f"  - {label}: {len([ex for ex in data if ex['label'] == label])}")


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
        # Count total lines first for tqdm
        total_lines = sum(1 for _ in f)
        f.seek(0)  # Reset file pointer

        for line in tqdm(f, total=total_lines, desc="Loading scenarios"):
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
    run_mode: RunMode,
) -> Dict:
    """Process a single scenario and return all results."""
    results = {"baseline": {}, "debates": [], "judgments": {}}

    if RunMode.BASELINE in run_mode:
        baseline = BaselineManager(scenario=scenario, models=judge_models)
        results["baseline"] = baseline.run()

    if RunMode.DEBATE in run_mode:
        debate = DebateTwoPlayers(
            scenario=scenario,
            debater_models=debater_models,
            word_limit=150,
            max_debate_rounds=3,
        )

        for variant in [
            {"swap": False, "all_wrong": False},
            {"swap": True, "all_wrong": False},
            {"swap": False, "all_wrong": True},
        ]:
            record = debate.run(**variant)
            results["debates"].append(record.to_dict())

            if RunMode.JUDGE in run_mode:
                judge = JudgeManager(record=record, judge_models=judge_models)
                judge_results = judge.run()
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
        # Ensure all objects are converted to dictionaries
        combined["debates"].extend(
            [
                debate if isinstance(debate, dict) else debate.to_dict()
                for debate in r["debates"]
            ]
        )

        # Merge baseline and judgment results
        for key in ["baseline", "judgments"]:
            for model, items in r[key].items():
                if model not in combined[key]:
                    combined[key][model] = []
                combined[key][model].extend(
                    [
                        item if isinstance(item, dict) else item.to_dict()
                        for item in items
                    ]
                )

    # Save each result type
    for key, content in combined.items():
        filename = Path(output_dir) / f"{key}_batch_{batch_num}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(content, f)


def process_batch(
    scenarios: List[DebateScenario],
    debater_models: List[str],
    judge_models: List[str],
    run_mode: RunMode,
    batch_num: int,
    output_dir: str,
    num_workers: int = None,
) -> None:
    """Process a batch of scenarios in parallel."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    # Process scenarios in parallel with progress bar
    with mp.Pool(num_workers) as pool:
        process_func = partial(
            process_single_scenario,
            debater_models=debater_models,
            judge_models=judge_models,
            run_mode=run_mode,
        )
        results = list(tqdm(
            pool.imap(process_func, scenarios),
            total=len(scenarios),
            desc=f"Processing batch {batch_num}"
        ))

    # Save combined results
    save_batch_results(results, batch_num, output_dir)


def process_scenarios_in_batches(
    scenarios: Iterator[DebateScenario],
    debater_models: List[str],
    judge_models: List[str],
    run_mode: RunMode = RunMode.ALL,
    batch_size: int = 5,
    output_dir: str = "results",
    num_workers: int = None,
) -> None:
    """Process scenarios in small batches with parallel processing."""
    Path(output_dir).mkdir(exist_ok=True)

    batch = []
    batch_num = 0

    # Convert iterator to list to get total count for tqdm
    scenarios = list(scenarios)
    total_batches = (len(scenarios) + batch_size - 1) // batch_size

    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for scenario in scenarios:
            batch.append(scenario)

            if len(batch) >= batch_size:
                process_batch(
                    batch,
                    debater_models,
                    judge_models,
                    run_mode,
                    batch_num,
                    output_dir,
                    num_workers,
                )
                batch = []
                batch_num += 1
                pbar.update(1)

        # Process remaining scenarios
        if batch:
            process_batch(
                batch,
                debater_models,
                judge_models,
                run_mode,
                batch_num,
                output_dir,
                num_workers,
            )
            pbar.update(1)


# Model configurations
MODEL_CONFIGS = {
    "self-play-claude-3.5-haiku": {
        "debater_models": ["claude-3-5-haiku-20241022", "claude-3-5-haiku-20241022"],
        "judge_models": ["claude-3-5-haiku-20241022"],
    },
    # "self-play-claude-3.5-sonnet": {
    #     "debater_models": ["claude-3-5-sonnet-20241022", "claude-3-5-sonnet-20241022"],
    #     "judge_models": ["claude-3-5-sonnet-20241022"]
    # },
    # "config2": {  # Stronger judge
    #     "debater_models": ["claude-3-5-haiku-20241022", "claude-3-haiku-20240307"],
    #     "judge_models": ["claude-3-sonnet-20240229"]
    # },
    # "config3": {  # Stronger debaters
    #     "debater_models": ["claude-3-sonnet-20240229", "claude-3-sonnet-20240229"],
    #     "judge_models": ["claude-3-haiku-20240307"]
    # }
}

if __name__ == "__main__":
    # Add run configuration
    RUN_MODE = RunMode.DEBATE | RunMode.JUDGE
    # For example:
    # RUN_MODE = RunMode.BASELINE  # Only baseline
    # RUN_MODE = RunMode.DEBATE | RunMode.JUDGE  # Only debate and judge
    # RUN_MODE = RunMode.BASELINE | RunMode.DEBATE  # Only baseline and debate

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
            RUN_MODE,
            batch_size=5,
            output_dir=OUTPUT_DIR,
            num_workers=4,
        )
