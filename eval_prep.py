"""
Module to process judge results.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Literal, List

import pandas as pd

from src.debate.types import DebateRecord, DebaterNames, DebateScenario


def get_available_models(judgement_path: Path) -> List[str]:
    """Get list of model names from judgements folder structure."""
    if not judgement_path.exists():
        return []
    return [p.name for p in judgement_path.iterdir() if p.is_dir()]

def main(args):
    debate_name = args.debate_name
    b_variation = args.b_variation if args.b_variation else ""
    j_variation = args.j_variation if args.j_variation else ""

    # Setup base paths
    result_path = Path("./results")
    judgement_base = result_path / f"judgements{j_variation}" / debate_name

    # Get models to process
    models_to_process = []
    if args.model_name:
        models_to_process = [args.model_name]
    else:
        models_to_process = get_available_models(judgement_base)
        if not models_to_process:
            print(f"No models found in {judgement_base}")
            return

    for model_name in models_to_process:
        print(f"Processing model: {model_name}")

        # Setup model-specific paths
        output_csv_name = (
            f"results_{model_name}_on_{debate_name}{j_variation}{b_variation}.csv"
        )

        scenarios_path = result_path / "scenarios"
        baseline_path = result_path / f"baselines{b_variation}" / model_name
        debate_path = result_path / "debates" / debate_name
        judgement_path = judgement_base / model_name

        # Verify paths exist
        if not all(p.exists() for p in [scenarios_path, baseline_path, debate_path, judgement_path]):
            print(f"Skipping {model_name} - missing required folders")
            continue

        scenario_paths = list(scenarios_path.glob("*.json"))
        baseline_paths = list(baseline_path.glob("*.json"))
        debate_paths = list(debate_path.glob("*.json"))
        judgement_paths = list(judgement_path.glob("*.json"))

        # Load data
        scenarios = {}
        baseline_records = {}
        debate_records = {}
        judgement_records = {}
        scenario2records = {}

        for path in scenario_paths:
            with open(path, "r") as f:
                scenarios[path.stem] = DebateScenario(**json.load(f))

        for path in baseline_paths:
            with open(path, "r") as f:
                baseline_records[path.stem] = json.load(f)

        for path in debate_paths:
            with open(path, "r") as f:
                n = path.stem.split("_")
                scenario_id = n[0]
                record_id = n[1]
                if scenario_id not in scenario2records:
                    scenario2records[scenario_id] = []
                scenario2records[scenario_id].append(record_id)
                debate_records[record_id] = json.load(f)

        for path in judgement_paths:
            with open(path, "r") as f:
                n = path.stem.split("_")
                record_id = n[1]
                judgement_records[record_id] = json.load(f)

        # Helper functions
        def get_answer(text: str) -> Literal["A", "B", "C"]:
            """Get the answer from the text."""
            answer = re.findall(r"Answer: (A|B|C)", text)
            if answer:
                return answer[0]
            answer = re.findall(r"Answer: <(A|B|C)>", text)
            if answer:
                return answer[0]
            return "invalid"

        def answer_to_label(answer: Literal["A", "B", "C"]) -> str:
            """Convert the answer to label."""
            if answer == "A":
                return "proved"
            elif answer == "B":
                return "disproved"
            elif answer == "C":
                return "unknown"
            return "invalid"

        def get_num_correct_quote(text: str) -> int:
            """Get the number of correct quote from the text."""
            pattern = re.compile(r"<v_quote>(.*?)</v_quote>")
            return len(re.findall(pattern, text))

        def get_num_incorrect_quote(text: str) -> int:
            """Get the number of incorrect quote from the text."""
            pattern = re.compile(r"<u_quote>(.*?)</u_quote>")
            return len(re.findall(pattern, text))

        # Process data
        answers = []

        for scenario_id, scenario in scenarios.items():

            if scenario_id not in baseline_records:
                continue

            baseline_judgement = baseline_records[scenario_id]["judgement"]
            baseline_answer = answer_to_label(get_answer(baseline_judgement))

            for record_id in scenario2records[scenario_id]:

                if record_id not in debate_records or record_id not in judgement_records:
                    continue

                debate_records[record_id].pop("scenario_id", None)
                debate_records[record_id].pop("record_id", None)
                try:
                    debate = DebateRecord(**debate_records[record_id])
                except TypeError:
                    debate = DebateRecord(scenario=scenario, **debate_records[record_id])

                correct_debater_pos: Literal[1, 2, -1] = -1
                if debate.debater_positions[DebaterNames.A.value] == scenario.label:
                    correct_debater_pos = 1
                elif debate.debater_positions[DebaterNames.B.value] == scenario.label:
                    correct_debater_pos = 2

                # Get the number of validated quote from each debater argument
                debater_a_args = ""
                debater_b_args = ""
                for t in debate.transcript:
                    if t["name"] == DebaterNames.A.value:
                        debater_a_args += t["validated_response_arguments"]
                    elif t["name"] == DebaterNames.B.value:
                        debater_b_args += t["validated_response_arguments"]

                num_correct_quote_a = get_num_correct_quote(debater_a_args)
                num_correct_quote_b = get_num_correct_quote(debater_b_args)
                num_incorrect_quote_a = get_num_incorrect_quote(debater_a_args)
                num_incorrect_quote_b = get_num_incorrect_quote(debater_b_args)

                debater_a_quote_acc = 0
                if num_correct_quote_a + num_incorrect_quote_a != 0:
                    debater_a_quote_acc = num_correct_quote_a / (
                        num_correct_quote_a + num_incorrect_quote_a
                    )
                debater_b_quote_acc = 0
                if num_correct_quote_b + num_incorrect_quote_b != 0:
                    debater_b_quote_acc = num_correct_quote_b / (
                        num_correct_quote_b + num_incorrect_quote_b
                    )

                # Get the judge answer
                judge_judgement = judgement_records[record_id]["judgement"]
                judge_chosen_debater = get_answer(judge_judgement)
                if judge_chosen_debater == "invalid":
                    print(f"Invalid answer for record {record_id} by model {model_name}")
                    print(judge_judgement)
                    continue

                try:
                    chosen_debater_name = getattr(DebaterNames, judge_chosen_debater)
                except AttributeError:
                    print(f"Invalid answer for record {record_id} by model {model_name}")
                    continue

                judge_answer = debate.debater_positions[chosen_debater_name.value]

                # Get the accuracy of the chosen debater
                chosen_debater_quote_acc = 0
                other_debater_quote_acc = 0
                if chosen_debater_name == DebaterNames.A:
                    chosen_debater_quote_acc = debater_a_quote_acc
                    other_debater_quote_acc = debater_b_quote_acc
                elif chosen_debater_name == DebaterNames.B:
                    chosen_debater_quote_acc = debater_b_quote_acc
                    other_debater_quote_acc = debater_a_quote_acc
                chosen_debater_quote_acc_diff = (
                    chosen_debater_quote_acc - other_debater_quote_acc
                )

                answers.append(
                    {
                        "scenario_id": scenario_id,
                        "record_id": record_id,
                        "level": scenario.level,
                        "correct_debater_pos": correct_debater_pos,
                        "label": scenario.label,
                        "baseline": baseline_answer,
                        "judge": judge_answer,
                        "chosen_debater": 1 if chosen_debater_name == DebaterNames.A else 2,
                        "a_quote_acc": debater_a_quote_acc,
                        "a_quote_num": (num_correct_quote_a + num_incorrect_quote_a),
                        "b_quote_acc": debater_b_quote_acc,
                        "b_quote_num": (num_correct_quote_b + num_incorrect_quote_b),
                        "chosen_debater_quote_acc_diff": chosen_debater_quote_acc_diff,
                    }
                )

        # Save results
        answers_df = pd.DataFrame.from_dict(answers)
        csv_result_path = result_path / output_csv_name
        answers_df.to_csv(csv_result_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process debate results")
    parser.add_argument("--debate-name", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=False)
    parser.add_argument("--b-variation", type=str, default="")
    parser.add_argument("--j-variation", type=str, default="")

    args = parser.parse_args()
    main(args)
