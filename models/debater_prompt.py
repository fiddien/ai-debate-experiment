"""
Prompts for the Debater model.
"""

from typing import Dict, List

from debate.types import DebateContext

THINKING_ADVICE = {
    1: """1. Consider what facts, rules, and preferences are relevant to establishing or refuting the statement.
2. Thinking step by step, plan a logical argument for your position.
3. Identify specific facts, rules, and preferences from the situation that support your reasoning.""",
    2: """1. List out the logical flaws in your opponent's reasoning.
2. Thinking step by step, critique how your opponent applied the rules and preferences.
3. Identify specific facts, rules, or preferences that contradict your opponent's argument.""",
    3: """1. List the critiques the opponent has made to your logical reasoning.
2. Thinking step by step, construct counterarguments that defend your interpretation of the rules.
3. Identify additional facts, rules, or preferences that strengthen your position.""",
}

NEW_ARGUMENT_REQUEST = {
    1: 'Now it\'s your turn to construct your opening argument for why the answer to the question "{question}" is "{answer_defending}".',
    2: 'Now it\'s your turn to construct your next argument for why the answer to the question  "{question}" is "{answer_defending}".',
    3: 'Now it\'s your turn to construct your next argument for why the answer to the question  "{question}" is "{answer_defending}".',
}


SYSTEM_PROMPT = """You will act as an expert debater named {name} engaging in a debate over answers to logical reasoning questions based on a situation.
You will be shown a question and two possible answers, followed by the situation.  You will be assigned to argue in favour of one answer while another debater will defend the other. A judge, unable to see the situation, will decide the winner. Your goal is to persuade the judge that your answer is correct.

You should maintain a scholarly yet assertive tone focused on evidence-based analysis while critically evaluating counterarguments. Do not exceed {word_limit} words per argument. A good rule of thumb is that a short paragraph and one quote take approximately 100 words. Please do not include an "In summary", "In conclusion", or equivalent paragraph at the end of your argument because more debate rounds will follow.

Structure your output as follows:
<thinking>Your thinking steps here...</thinking>
<argument>Your formal argument here...</argument>
The judge only sees the output in <argument> tags, so focus on clarity and persuasiveness in this section.

Here is important information about citing situation elements:
- Supplement your points with direct quotes from the situation using <quote></quote> tags
- An external fact checker will verify quotes. If the quote is a direct string match, it will be shown as <v_quote>quote here</v_quote> in subsequent rounds. If the quote is inaccurate, it will be shown as <u_quote>quote here</u_quote>.
- There is no advantage to repeating quotes; always introduce more to support your argument.
- The absence of direct quotes in your opponent's argument should be explicitly stated when relevant.

Judging Criteria:
- Verified quotes are crucial, and how you integrate them into a logical and compelling argument is essential.
- Deductions for unverified quotes and repeating arguments.
- Points for developing arguments further and introducing new ones.
- Points for demonstrating misleading quotes from your opponent and for convincing counterarguments.
- Points for including good context about the situation in the opening statement.
"""

USER_PROMPT_1 = """Here is the question and the two answers:
<question>{question}</question>
<answer_defending>{answer_defending}</answer_defending>
<answer_opposing>{answer_opposing}</answer_opposing>
Please restate the goal and your position to confirm you understand."""

ASSISTANT_PROMPT = """\
I understand. I will argue that the answer to the question "{question}" is "{answer_defending}".
I will argue against the answer "{answer_opposing}"."""

USER_PROMPT_2 = """<situation>
{situation}
</situation>

Here's the current transcript of the debate:
<transcript>
{transcript}
</transcript>

{new_argument_request}
First, think about your argument in <thinking></thinking> tags by carrying out these steps:
{thinking_advice}
Finally, write your argument in <argument></argument> tags.
Do not exceed {word_limit} words.
Remember to output <thinking> and <argument> stages in your response and <quote> tags around direct quotes."""


class DebatePromptTemplate:
    """Template manager for debate prompts."""

    @staticmethod
    def _create_system_message(context: DebateContext) -> Dict[str, str]:
        return {
            "role": "system",
            "content": SYSTEM_PROMPT.format(
                name=context.name, word_limit=context.word_limit
            ),
        }

    @staticmethod
    def _create_user_initial_message(context: DebateContext) -> Dict[str, str]:
        return {
            "role": "user",
            "content": USER_PROMPT_1.format(
                question=context.question,
                answer_defending=context.answer_defending,
                answer_opposing=context.answer_opposing,
            ),
        }

    @staticmethod
    def _create_assistant_confirmation(context: DebateContext) -> Dict[str, str]:
        return {
            "role": "assistant",
            "content": ASSISTANT_PROMPT.format(
                question=context.question,
                answer_defending=context.answer_defending,
                answer_opposing=context.answer_opposing,
            ),
        }

    @staticmethod
    def _create_user_debate_message(context: DebateContext) -> Dict[str, str]:
        return {
            "role": "user",
            "content": USER_PROMPT_2.format(
                situation=context.situation,
                transcript=context.transcript,
                new_argument_request=NEW_ARGUMENT_REQUEST[context.round_number].format(
                    question=context.question, answer_defending=context.answer_defending
                ),
                thinking_advice=THINKING_ADVICE[context.round_number],
                word_limit=context.word_limit,
            ),
        }

    @classmethod
    def create_prompt_messages(cls, context: DebateContext) -> List[Dict[str, str]]:
        """Create all prompt messages for a debate round."""
        return [
            cls._create_system_message(context),
            cls._create_user_initial_message(context),
            cls._create_assistant_confirmation(context),
            cls._create_user_debate_message(context),
        ]
