# AI Debate Experiment

This repository contains code and configurations for running AI debate experiments. The experiments involve different AI models debating on BoardgameQA[^1] and being judged by other AI models. The goal is to explore the capabilities of AI in generating and evaluating arguments in a persuasive debate format.

## Running Experiments

Each experiment can be run using the following command pattern:
```bash
python main.py --config-file configs/<config_file>.json [additional options]
```

### Available Experiments

1. Claude Haiku Self-Play:
```bash
python main.py --config-file configs/claude_haiku_self_play.json
```

2. Claude Sonnet Self-Play:
```bash
python main.py --config-file configs/claude_sonnet_self_play.json
```

3. Claude Sonnet Judging Haiku Debates:
```bash
python main.py --config-file configs/claude_sonnet_on_haikus.json
```

4. Claude Sonnet Judging with Prompt 2:
```bash
python main.py --config-file configs/claude_sonnet_on_haikus_prompt2.json --variation USER_PROMPT2
```

5. Deepseek Judging Haiku Debates:
```bash
python main.py --config-file configs/deepseek_on_haikus.json
```

6. Deepseek Judging with Prompt 2:
```bash
python main.py --config-file configs/deepseek_on_haikus_prompt2.json --variation USER_PROMPT2
```

7. GPT-4 Judging Haiku Debates:
```bash
python main.py --config-file configs/gpt4o_on_haikus.json
```

### Optional Arguments

- `--sampled-data-path`: Path to sampled data file (default: "data/sampled_boardgame_qa.jsonl")
- `--samples-per-label`: Number of samples per label (default: 20)
- `--levels`: Difficulty levels to use (default: ["LowConflict", "HighConflict"])
- `--excluded-labels`: Labels to exclude from processing (default: ["unknown"])
- `--output-dir`: Output directory for results (default: "results")
- `--variation`: Optional variation suffix for output directories (default: "")

Example with custom options:
```bash
python main.py --config-file configs/gpt4_on_haikus.json --samples-per-label 30 --output-dir custom_results
```

[^1]: Kazemi, M., Yuan, Q., Bhatia, D., Kim, N., Xu, X., Imbrasaite, V., & Ramachandran, D. (2024). Boardgameqa: A dataset for natural language reasoning with contradictory information. *Advances in Neural Information Processing Systems*, 36. [[Link](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7adce80e86aa841490e6307109094de5-Abstract-Datasets_and_Benchmarks.html)]

[^2]: Khan, A., Hughes, J., Valentine, D., Ruis, L., Sachan, K., Radhakrishnan, A., ... & Perez, E. (2024). Debating with more persuasive llms leads to more truthful answers. *arXiv preprint arXiv:2402.06782*. [[Link](https://arxiv.org/abs/2402.06782)]