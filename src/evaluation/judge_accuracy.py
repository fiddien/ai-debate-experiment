import json
from pathlib import Path
from collections import defaultdict

ANSWER_MAP = {
    "A": "proved",
    "B": "disproved",
    "C": "unknown"
}

def evaluate_judge_accuracy(
    qa_path="data/sampled_boardgame_qa.jsonl",
    debates_file="results/self-play-claude-3.5-haiku/debates_batch_0.json",
    judgements_file="results/self-play-claude-3.5-haiku/judgements_batch_0.json",
):
    # Load scenarios keyed by 'example'
    scenarios = {}
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            scenarios[ex["example"]] = {"label": ex["label"], "level": ex["level"]}

    # Load debates: ID → { "example": str, "correct_side": "A"/"B"/"None", ... }
    debates = {}
    try:
        with open(debates_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for debate in data:
            debate_id = debate["id"]
            # ...existing code to find scenario, ground truth side...
            # Suppose we store the substring in debate["scenario"] → "example"
            debater_ground_truth = "A"  # or "B" or "None" as needed
            debates[debate_id] = {
                "example": debate.get("scenario", ""),
                "correct_side": debater_ground_truth,
            }
    except FileNotFoundError:
        pass

    # Load judgements and evaluate
    correct = 0
    total = 0
    try:
        with open(judgements_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            debate_id = entry["id"]
            judgement_text = entry["judgement"]
            # Extract the final "Answer: X"
            # Example: "...Answer: A"
            if "Answer:" in judgement_text:
                ans_letter = judgement_text.split("Answer:")[-1].strip()[0]
                predicted_label = ANSWER_MAP.get(ans_letter, "unknown")

                # Match scenario label
                d = debates.get(debate_id, {})
                scenario_info = scenarios.get(d.get("example", ""), {})
                ground_label = scenario_info.get("label", "")
                if ground_label == predicted_label:
                    correct += 1
                total += 1
    except FileNotFoundError:
        pass

    accuracy = correct / total if total else 0.0
    print(f"Judge accuracy: {accuracy*100:.2f}% (based on {total} judgements)")
