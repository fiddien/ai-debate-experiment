"""
Main script to run the debate experiments with the BoardgameQA dataset.
"""

import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import re
from collections import defaultdict
from enum import Flag, auto
from functools import partial
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Tuple

from tqdm import tqdm

from src.debate.baseline import BaselineManager
from src.debate.debate import DebateTwoPlayers
from src.debate.judge import JudgeManager
from src.debate.types import DebateScenario

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.WARNING
)

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_PER_LABEL = 20
# Dataset difficulty levels
LEVELS = [
    # "ZeroConflict",
    "LowConflict",
    # "Main",  # Medium
    "HighConflict",
]
N = DEFAULT_SAMPLE_PER_LABEL * len(LEVELS) * 3  # 3 labels
# N = 2  # For testing

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


def sample_data(ds: dict, samples_per_label: int = DEFAULT_SAMPLE_PER_LABEL) -> dict:
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


def display_total(ds: dict):
    """Display the total number of examples per label and per level."""
    print("Total examples per level:")
    for level, data in ds.items():
        print(f"{level}: {len(data)}")
        for label in set([ex["label"] for ex in data]):
            print(f"  - {label}: {len([ex for ex in data if ex['label'] == label])}")


def save_sampled_data(data: dict, result_paths: Dict[str, Path]):
    """Save sampled dataset directly as scenarios."""
    scenarios_path = result_paths["scenarios"]
    scenarios_path.mkdir(exist_ok=True)

    for level, examples in data.items():
        for example in examples:
            situation, question = split_question(example["example"])
            scenario = DebateScenario(
                situation=situation,
                question=question,
                answer_options=["proved", "disproved", "unknown"],
                label=example["label"],
                id=hashlib.md5(example["example"].encode()).hexdigest(),
            )
            save_scenario(scenario, result_paths)


def get_scenarios_stream(result_paths: Dict[str, Path]) -> Generator[DebateScenario, None, None]:
    """Stream existing scenarios from the organized directory."""
    scenario_files = result_paths["scenarios"].glob("*.json")

    for file_path in tqdm(list(scenario_files), desc="Loading scenarios"):
        yield load_scenario(file_path.stem, result_paths)


def load_scenarios_stream(
    filepath: str, result_paths: Dict[str, Path]
) -> Generator[DebateScenario, None, None]:
    """Load and yield scenarios one at a time, saving them to organized files."""
    with open(filepath, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
        f.seek(0)

        for line in tqdm(f, total=total_lines, desc="Loading scenarios"):
            game = json.loads(line.strip())
            situation, question = split_question(game["example"])
            scenario = DebateScenario(
                situation=situation,
                question=question,
                answer_options=["proved", "disproved", "unknown"],
                label=game["label"],
                id=hashlib.md5(game["example"].encode()).hexdigest(),
            )
            # Save scenario to organized structure
            save_scenario(scenario, result_paths)
            yield scenario


def split_question(example: str) -> Tuple[str, str]:
    """Split the example text into the game and the question."""
    example_text = example
    question = re.split(r"\.\s+", example_text)[-1]
    game = example_text.replace(question, "").strip()
    return game, question


def get_result_paths(
    base_dir: str = "results",
    config_name: str = None,
    reuse_debates_from: str = None,
) -> Dict[str, Path]:
    """Get paths for different result types."""
    base = Path(base_dir)
    if config_name:
        paths = {
            "scenarios": base / "scenarios",
            "baseline": base / "baseline",
            "debates": base / "debates" / config_name,
            "judgments": base / "judgments" / config_name,
        }
        if reuse_debates_from:
            # Use the referenced config's debate path
            paths["debates"] = base / "debates" / reuse_debates_from
            # Put judgments under the same config as debates
            paths["judgments"] = base / "judgments" / reuse_debates_from
        return paths
    return {
        "scenarios": base / "scenarios",
        "baseline": base / "baseline",
        "debates": base / "debates",
        "judgments": base / "judgments",
    }


def save_scenario(scenario: DebateScenario, paths: Dict[str, Path]):
    """Save scenario to a JSON file."""
    save_path = paths["scenarios"] / f"{scenario.id}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    scenario_dict = {
        "id": scenario.id,
        "situation": scenario.situation,
        "question": scenario.question,
        "answer_options": scenario.answer_options,
        "label": scenario.label,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(scenario_dict, f)


def load_scenario(scenario_id: str, paths: Dict[str, Path]) -> DebateScenario:
    """Load scenario from a JSON file."""
    scenario_path = paths["scenarios"] / f"{scenario_id}.json"
    if not scenario_path.exists():
        raise FileNotFoundError(f"No scenario file found for ID: {scenario_id}")

    with open(scenario_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return DebateScenario(
        situation=data["situation"],
        question=data["question"],
        answer_options=data["answer_options"],
        label=data["label"],
        id=data["id"],
    )


def load_existing_results(
    scenario_id: str, paths: Dict[str, Path], config_name: str = None
) -> Dict:
    """Load existing debate records for a scenario."""
    results = {"debates": []}

    # Load debate records if they exist
    if config_name:
        debate_paths = list(
            (paths["debates"] / config_name).glob(f"{scenario_id}*.json")
        )
    else:
        debate_paths = list(paths["debates"].glob(f"**/{scenario_id}*.json"))

    for path in debate_paths:
        with open(path, "r", encoding="utf-8") as f:
            results["debates"].append(json.load(f))

    return results


def save_result(
    result: Dict,
    result_type: str,
    paths: Dict[str, Path],
    scenario_id: str,
    model_name: str = None,
    record_id: str = None,
    config_name: str = None,
):
    """Save a single result to the appropriate location."""
    if result_type == "baseline":
        save_path = paths["baseline"] / model_name / f"{scenario_id}.json"
    elif result_type == "debates":
        save_path = paths["debates"] / config_name / f"{scenario_id}_{record_id}.json"
    elif result_type == "judgments":
        save_path = (
            paths["judgments"]
            / config_name
            / model_name
            / f"{scenario_id}_{record_id}.json"
        )
    else:
        raise ValueError(f"Unknown result type: {result_type}")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


def process_single_scenario(
    scenario: DebateScenario,
    debater_models: List[str],
    judge_models: List[str],
    run_mode: RunMode,
    result_paths: Dict[str, Path] = None,
    config_name: str = None,
) -> Dict:
    """Process a single scenario, optionally using existing debate results."""
    results = {"baseline": {}, "debates": [], "judgments": {}}

    # Only load existing debate results if we're reusing them
    if result_paths and RunMode.DEBATE not in run_mode:
        existing_results = load_existing_results(scenario.id, result_paths, config_name)
        results["debates"] = existing_results["debates"]

    if RunMode.BASELINE in run_mode:
        baseline = BaselineManager(scenario=scenario, models=judge_models)
        results["baseline"] = baseline.run()
        if result_paths:
            for model, result in results["baseline"].items():
                save_result(result, "baseline", result_paths, scenario.id, model)

    if RunMode.DEBATE in run_mode and not results["debates"]:
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
            results["debates"].append(record)
            if result_paths:
                save_result(
                    record,
                    "debates",
                    result_paths,
                    scenario.id,
                    record_id=record.id,
                    config_name=config_name,
                )

    if RunMode.JUDGE in run_mode:
        for record in results["debates"]:
            judge = JudgeManager(record=record, judge_models=judge_models)
            judge_results = judge.run()
            for model, judgments in judge_results.items():
                if model not in results["judgments"]:
                    results["judgments"][model] = []
                results["judgments"][model].extend(judgments)
                if result_paths:
                    for judgment in judgments:
                        save_result(
                            judgment,
                            "judgments",
                            result_paths,
                            scenario.id,
                            model,
                            record["id"],
                            config_name=config_name,
                        )

    return results


def save_batch_results(results: List[Dict], result_paths: Dict[str, Path]):
    """Save batch results according to the organized directory structure."""

    for result in results:
        # Save baseline results
        for model, b_result in result.get("baseline", {}).items():
            scenario_id = b_result["id"]
            save_path = result_paths["baseline"] / model / f"{scenario_id}.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(b_result, f)

        record2scenario = {}
        # Save debate results
        for debate in result.get("debates", []):
            scenario_id = debate["scenario_id"]
            record_id = debate["id"]
            record2scenario[record_id] = scenario_id
            save_path = result_paths["debates"] / f"{scenario_id}_{record_id}.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(debate, f)

        # Save judgment results
        for model, judgment in result.get("judgments", {}).items():
            record_id = judgment["id"]
            scenario_id = record2scenario[record_id]
            save_path = (
                result_paths["judgments"]
                / model
                / f"{scenario_id}_{record_id}.json"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(judgment, f)


def process_batch(
    scenarios: List[DebateScenario],
    debater_models: List[str],
    judge_models: List[str],
    run_mode: RunMode,
    batch_num: int,
    result_paths: Dict[str, Path] = None,
    num_workers: int = None,
    config_name: str = None,
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
            result_paths=result_paths,
            config_name=config_name,
        )
        results = list(
            tqdm(
                pool.imap(process_func, scenarios),
                total=N,
                desc=f"Processing batch {batch_num}",
            )
        )

    # Save combined results
    save_batch_results(results, result_paths)


def process_scenarios_in_batches(
    scenarios: Iterator[DebateScenario],
    debater_models: List[str],
    judge_models: List[str],
    run_mode: RunMode = RunMode.ALL,
    batch_size: int = 5,
    result_paths: Dict[str, Path] = None,
    num_workers: int = None,
    config_name: str = None,
) -> None:
    """Process scenarios in small batches with parallel processing."""

    batch = []
    batch_num = 0

    total_batches = (N + batch_size - 1) // batch_size

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
                    result_paths,
                    num_workers,
                    config_name,
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
                result_paths,
                num_workers,
                config_name,
            )
            pbar.update(1)


MODEL_CONFIGS = {
    # "claude-3.5-haiku vs claude-3.5-haiku": {  # This is the config name
    #     "debater_models": ["claude-3-5-haiku-20241022", "claude-3-5-haiku-20241022"],
    #     "judge_models": ["claude-3-5-haiku-20241022"],
    # },
    "claude-3.5-sonnet on haikus": {
        "debater_models": None,
        "judge_models": ["claude-3-5-sonnet-20241022"],
        "reuse_debates_from": "claude-3.5-haiku vs claude-3.5-haiku",  # Reference to another config name
    },
}

if __name__ == "__main__":
    # Add run configuration
    RUN_MODE = RunMode.BASELINE | RunMode.JUDGE
    # For example:
    # RUN_MODE = RunMode.BASELINE  # Only baseline
    # RUN_MODE = RunMode.DEBATE | RunMode.JUDGE  # Only debate and judge
    # RUN_MODE = RunMode.BASELINE | RunMode.DEBATE  # Only baseline and debate

    SAMPLED_DATA_PATH = "data/sampled_boardgame_qa.jsonl"
    # Check if sampled data exists
    if not os.path.exists(SAMPLED_DATA_PATH):
        save_sampled_data(sample_data(load_boardgame_qa()), SAMPLED_DATA_PATH)

    # Set up common paths first
    base_paths = get_result_paths("results")

    # Run experiments for each configuration
    for config_name, models in MODEL_CONFIGS.items():
        print(f"\nRunning experiments for {config_name}")

        # Set up models for this configuration
        DEBATER_MODELS = models.get("debater_models")
        JUDGE_MODELS = models["judge_models"]
        REUSE_DEBATES = models.get("reuse_debates_from")

        # Set up paths for results with reuse_debates_from
        result_paths = get_result_paths(
            "results",
            config_name,
            reuse_debates_from=REUSE_DEBATES
        )
        result_paths["scenarios"] = base_paths["scenarios"]
        print(result_paths)

        # Determine run mode based on configuration
        current_run_mode = RUN_MODE
        if REUSE_DEBATES:
            # Remove DEBATE from run mode if we're reusing debates
            current_run_mode &= ~RunMode.DEBATE

        # Set up output directory for this configuration
        OUTPUT_DIR = f"results/{config_name}"
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Add debate records path if needed
        DEBATE_RECORDS_PATH = None
        if RunMode.JUDGE in RUN_MODE and RunMode.DEBATE not in RUN_MODE:
            DEBATE_RECORDS_PATH = "path/to/debate/records"  # User should set this

        # Process scenarios in streaming fashion with parallel processing
        scenarios_stream = get_scenarios_stream(result_paths)

        process_scenarios_in_batches(
            scenarios_stream,
            DEBATER_MODELS,
            JUDGE_MODELS,
            current_run_mode,
            batch_size=5,
            result_paths=result_paths,
            num_workers=4,
            config_name=config_name,
        )
