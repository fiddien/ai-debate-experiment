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
import argparse
from collections import defaultdict
from enum import Flag, auto
from functools import partial
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Tuple

from tqdm import tqdm

from src.debate.baseline import BaselineManager
from src.debate.debate import DebateTwoPlayers
from src.debate.judge import JudgeManager
from src.debate.types import DebateScenario, DebateRecord, DebateResponse

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)


class RunMode(Flag):
    """Run modes for the debate experiments."""

    BASELINE = auto()
    DEBATE = auto()
    JUDGE = auto()
    ALL = BASELINE | DEBATE | JUDGE


SAMPLED_DATA_PATH = "data/sampled_boardgame_qa.jsonl"
DEFAULT_SAMPLE_PER_LABEL = 20
LEVELS = [
    # "ZeroConflict",
    "LowConflict",
    # "Main",  # Medium
    "HighConflict",
]
EXCLUDED_LABELS = ["unknown"]
N = DEFAULT_SAMPLE_PER_LABEL * len(LEVELS) * 3  # 3 labels

OUTPUT_DIR = "results"
VARIATION = ""  # "P2"


def load_boardgame_qa(base_path: str = "BoardgameQA") -> dict:
    """Load BoardgameQA dataset from json files."""
    ds = {}
    for level in LEVELS:
        path = Path(base_path) / f"BoardgameQA-{level}-depth2/test.json"
        with open(path, "r", encoding="utf-8") as f:
            ds[level] = json.load(f)
    return ds


def display_total(ds: dict):
    """Display the total number of examples per label and per level."""
    print("Total examples per level:")
    for level, data in ds.items():
        print(f"{level}: {len(data)}")
        for label in set([ex["label"] for ex in data]):
            print(f"  - {label}: {len([ex for ex in data if ex['label'] == label])}")


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


def save_sampled_data(data: dict, result_paths: Dict[str, Path]):
    """Save sampled dataset directly as scenarios."""
    scenarios_path = result_paths["scenarios"]
    scenarios_path.mkdir(exist_ok=True)

    for examples in data.values():
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


def split_question(example: str) -> Tuple[str, str]:
    """Split the example text into the game and the question."""
    example_text = example
    question = re.split(r"\.\s+", example_text)[-1]
    game = example_text.replace(question, "").strip()
    return game, question


def get_scenarios_stream(
    result_paths: Dict[str, Path],
) -> Generator[DebateScenario, None, None]:
    """Stream existing scenarios from the organized directory."""
    scenario_files = result_paths["scenarios"].glob("*.json")
    for file_path in list(scenario_files):
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
            if game["label"] in EXCLUDED_LABELS:
                continue
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


def load_existing_results(scenario_id: str, paths: Dict[str, Path]) -> Dict:
    """Load existing debate records for a scenario."""
    results = {"debates": []}
    debate_paths = list(paths["debates"].glob(f"**/{scenario_id}*.json"))
    logger.info(
        "Loading existing results for scenario: %s (%s records)",
        scenario_id,
        len(debate_paths),
    )
    for path in debate_paths:
        with open(path, "r", encoding="utf-8") as f:
            results["debates"].append(json.load(f))
    return results


def get_result_paths(
    base_dir: str = "results",
    config: str = None,
    reuse_debates_from: str = None,
    variation: str = "",
) -> Dict[str, Path]:
    """Get paths for different result types."""
    os.makedirs(base_dir, exist_ok=True)
    base = Path(base_dir)
    variation = f"_{variation}" if variation else ""
    paths = {
        "scenarios": base / "scenarios",
        "baselines": base / f"baselines{variation}",
        "debates": base / "debates",
        "judgements": base / f"judgements{variation}",
    }
    if reuse_debates_from:
        paths["debates"] = base / "debates" / reuse_debates_from
        paths["judgements"] = base / f"judgements{variation}" / reuse_debates_from
    elif config:
        paths["debates"] = base / "debates" / config
        paths["judgements"] = base / f"judgements{variation}" / config
    return paths


def save_result(
    result: Dict,
    result_type: str,
    paths: Dict[str, Path],
    scenario_id: str,
    model_name: str = None,
    record_id: str = None,
):
    """Save a single result to the appropriate location."""
    if result_type == "baselines":
        save_path = paths["baselines"] / model_name / f"{scenario_id}.json"
    elif result_type == "debates":
        save_path = paths["debates"] / f"{scenario_id}_{record_id}.json"
    elif result_type == "judgements":
        save_path = paths["judgements"] / model_name / f"{scenario_id}_{record_id}.json"
    else:
        raise ValueError(f"Unknown result type: {result_type}")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


def process_batch(
    scenarios: List[DebateScenario],
    debater_models: List[str],
    judge_models: List[str],
    run_mode: RunMode,
    result_paths: Dict[str, Path],
    num_workers: int = None,
) -> None:
    """Process a batch of scenarios in parallel."""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 2)

    with mp.Pool(num_workers) as pool:
        process_func = partial(
            process_single_scenario,
            debater_models=debater_models,
            judge_models=judge_models,
            run_mode=run_mode,
            result_paths=result_paths,
        )
        list(pool.imap(process_func, scenarios))


def process_scenarios_in_batches(
    scenarios: Iterator[DebateScenario],
    debater_models: List[str],
    judge_models: List[str],
    run_mode: RunMode,
    result_paths: Dict[str, Path],
    batch_size: int = 5,
    num_workers: int = None,
) -> None:
    """Process scenarios in batches."""

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
                    result_paths,
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
                result_paths,
                num_workers,
            )
            pbar.update(1)


def process_single_scenario(
    scenario: DebateScenario,
    debater_models: List[str],
    judge_models: List[str],
    run_mode: RunMode,
    result_paths: Dict[str, Path],
) -> Dict:
    """Process a single scenario, optionally using existing debate results."""
    debate_results = []

    # Only load existing debate results if we're reusing them
    if RunMode.DEBATE not in run_mode:
        existing_results = load_existing_results(scenario.id, result_paths)
        debate_results = existing_results["debates"]

    if RunMode.BASELINE in run_mode:
        baseline = BaselineManager(scenario=scenario, models=judge_models)
        for model, result in baseline.run().items():
            save_result(result, "baselines", result_paths, scenario.id, model)

    if RunMode.DEBATE in run_mode and not debate_results:
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
            record = debate.run(cooldown=3, **variant)
            debate_results.append(record)
            save_result(
                record,
                "debates",
                result_paths,
                scenario.id,
                record_id=record["id"],
            )

    if RunMode.JUDGE in run_mode:
        for record in debate_results:
            if not record.get("record_id"):
                record["record_id"] = record["id"]
            judge = JudgeManager(
                record=DebateRecord(
                    scenario=scenario,
                    debater_positions=record["debater_positions"],
                    debater_models=record["debater_models"],
                    swap=record["swap"],
                    all_wrong=record["all_wrong"],
                    id=record["record_id"],
                    transcript=[DebateResponse(**r) for r in record["transcript"]],
                ),
                judge_models=judge_models,
            )
            judge_results = judge.run()
            for model, judgements in judge_results.items():
                save_result(
                    judgements,
                    "judgements",
                    result_paths,
                    scenario.id,
                    model,
                    record["record_id"],
                )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run debate experiments with BoardgameQA dataset."
    )
    parser.add_argument(
        "--sampled-data-path",
        default="data/sampled_boardgame_qa.jsonl",
        help="Path to sampled data file",
    )
    parser.add_argument(
        "--samples-per-label", type=int, default=20, help="Number of samples per label"
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["LowConflict", "HighConflict"],
        help="Difficulty levels to use",
    )
    parser.add_argument(
        "--excluded-labels",
        nargs="+",
        default=["unknown"],
        help="Labels to exclude from processing",
    )
    parser.add_argument(
        "--output-dir", default="results", help="Output directory for results"
    )
    parser.add_argument(
        "--variation",
        default="",
        help="Optional variation suffix for output directories",
    )
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to experiment configurations JSON file",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes (defaults to half of CPU cores)",
    )
    return parser.parse_args()


def load_experiment_configs(config_file):
    """Load experiment configurations from JSON file."""
    with open(config_file, "r") as f:
        return json.load(f)


def parse_run_mode(mode_config) -> RunMode:
    """Convert config run_mode value to RunMode flag."""
    if isinstance(mode_config, str):
        mode_config = [mode_config]

    run_mode = RunMode(0)  # Start with empty flag
    for mode in mode_config:
        try:
            run_mode |= RunMode[mode.upper()]
        except KeyError:
            raise ValueError(f"Invalid run mode: {mode}")

    return run_mode if run_mode else RunMode.ALL


def main():
    """Main execution function for the debate experiments."""
    args = parse_args()

    # Update global variables with command line arguments
    global SAMPLED_DATA_PATH, DEFAULT_SAMPLE_PER_LABEL, LEVELS, EXCLUDED_LABELS
    global OUTPUT_DIR, VARIATION, N

    SAMPLED_DATA_PATH = args.sampled_data_path
    DEFAULT_SAMPLE_PER_LABEL = args.samples_per_label
    LEVELS = args.levels
    EXCLUDED_LABELS = args.excluded_labels
    OUTPUT_DIR = args.output_dir
    VARIATION = args.variation
    N = DEFAULT_SAMPLE_PER_LABEL * len(LEVELS) * 3  # 3 labels

    EXPERIMENT_CONFIGS = load_experiment_configs(args.config_file)

    # Check if sampled data exists
    if not os.path.exists(SAMPLED_DATA_PATH):
        save_sampled_data(sample_data(load_boardgame_qa()), SAMPLED_DATA_PATH)

    for config_name, config in EXPERIMENT_CONFIGS.items():
        print(f"Running experiments for '{config_name}'")

        # Set up models for this configuration
        run_mode = parse_run_mode(config.get("run_mode", "ALL"))
        debater_models = config.get("debater_models")
        judge_models = config["judge_models"]
        reuse_debates = config.get("reuse_debates_from")

        # Set up paths for results with reuse_debates_from
        result_paths = get_result_paths(
            OUTPUT_DIR,
            config_name,
            reuse_debates_from=reuse_debates,
            variation=VARIATION,
        )

        if reuse_debates:
            # Remove DEBATE from run mode if we're reusing debates
            run_mode &= ~RunMode.DEBATE
        logging.info("Current run mode: %s", run_mode)

        process_scenarios_in_batches(
            get_scenarios_stream(result_paths),
            debater_models,
            judge_models,
            run_mode,
            result_paths,
            batch_size=5,
            num_workers=args.num_workers,
        )
        print()


if __name__ == "__main__":
    main()
