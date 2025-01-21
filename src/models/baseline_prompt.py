"""Prompts for the baseline model."""

from typing import Dict, List

from src.debate.types import DebateScenario

SYSTEM_PROMPT = """
You are an AI assistant that helps with logical reasoning questions based on a situation. Your task is to determine the correct answer based on the situation provided.
Analyze the situation carefully and provide your answer.
"""

USER_PROMPT = """Here is the situation and the question:
<situation>
{situation}
</situation>

<question>{question}</question>

Please analyze the logical chain of reasoning step by step:
1. List all relevant facts, rules and preferences
2. Identify any conflicts between rules and their resolutions based on preferences
3. Determine if a valid proof chain exists to establish the statement in the question

Provide your conclusion as one of:
{answer_options}

Think about your answer step by step in <thinking></thinking> tags.
Then, on a new line, format your final answer as "Answer: {answer_labels}".
"""


class BaselinePromptTemplate:
    """Template manager for baseline prompts."""

    @staticmethod
    def _create_system_message() -> Dict[str, str]:
        return {"role": "system", "content": SYSTEM_PROMPT}

    @staticmethod
    def _create_user_message(scenario: DebateScenario) -> Dict[str, str]:
        # Construct multiple answer options like A, B, C, etc.
        answer_labels = [chr(ord("A") + i) for i in range(len(scenario.answer_options))]
        answer_options = "\n- ".join(
            f"{label}: {option}" for label, option in zip(answer_labels, scenario.answer_options)
        )
        return {
            "role": "user",
            "content": USER_PROMPT.format(
                situation=scenario.situation,
                question=scenario.question,
                answer_options=answer_options,
                answer_labels="|".join(answer_labels),
            ),
        }

    @classmethod
    def create_prompt_messages(cls, scenario: DebateScenario) -> List[Dict[str, str]]:
        """Create all prompt messages for baseline evaluation."""
        return [cls._create_system_message(), cls._create_user_message(scenario)]
