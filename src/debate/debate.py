"""
Debate class for running a debate between two players.
"""

import logging
import time
from typing import Any, Dict, List, Tuple

from typing_extensions import Union
from langfuse.decorators import observe, langfuse_context

from src.models.debater_prompt import DebatePromptTemplate
from src.models.llms import get_response

from .config import DEFAULT_MAX_ROUNDS, DEFAULT_WORD_LIMIT
from .exceptions import DebateConfigurationError
from .types import (
    DebateContext,
    DebateRecord,
    DebateResponse,
    DebaterNames,
    DebateScenario,
)
from .utils.cache import generate_cache_key, load_from_cache, save_to_cache
from .utils.response import extract_argument, format_transcript, validate_citations


class DebateTwoPlayers:
    """Run a debate between two players."""

    def __init__(
        self,
        scenario: DebateScenario,
        debater_models: List[str],
        debater_names: List[str] = None,
        word_limit: int = DEFAULT_WORD_LIMIT,
        max_debate_rounds: int = DEFAULT_MAX_ROUNDS,
    ):
        self.logger = logging.getLogger(__name__)
        self.scenario = scenario
        self.debater_models = debater_models
        self.logger.info("Initializing debate with %d models", len(debater_models))

        # Validate and set debater names
        if debater_names is None:
            debater_names = [str(n) for n in list(DebaterNames)]
            self.logger.debug("Using default debater names")

        if len(debater_names) < len(debater_models):
            self.logger.error("Insufficient debater names for models")
            raise DebateConfigurationError(
                f"Number of debater names ({len(debater_names)}) must be >= "
                f"number of models ({len(debater_models)})"
            )

        # Only use as many names as models
        self.debater_names = debater_names[: len(debater_models)]
        self.name_to_model = dict(zip(self.debater_names, self.debater_models))

        self.word_limit = word_limit
        self.max_debate_rounds = max_debate_rounds
        self._setup_answers(scenario.answer_options)

        self.cache_key = generate_cache_key(
            "debate",
            scenario_id=scenario.id,
            models=debater_models,
            names=debater_names,
            word_limit=word_limit,
            max_rounds=max_debate_rounds,
        )

        self.logger.info("Debate initialized for scenario ID: %s", self.scenario.id)
        self.current_record = None

    def _setup_answers(self, labels: List[str]) -> None:
        """Initialize the answer positions for each label."""
        n = len(labels)
        self.answer_pos: Dict[str, Tuple[str, str]] = {}
        self.wrong_answer_pos: Dict[str, Tuple[str, str]] = {}

        for i, label in enumerate(labels):
            next_idx = (i + 1) % n
            next_next_idx = (i + 2) % n
            self.answer_pos[label] = (label, labels[next_idx])
            self.wrong_answer_pos[label] = (labels[next_idx], labels[next_next_idx])

    def _get_labels(
        self,
        name: str,
        swap: bool = False,
        all_wrong: bool = False,
    ) -> Tuple[str, str]:
        label = self.scenario.label
        pos = self.wrong_answer_pos if all_wrong else self.answer_pos
        result = pos[label] if not swap else pos[label][::-1]
        return result if name == self.debater_names[0] else result[::-1]

    def _create_messages(
        self,
        scenario: DebateScenario,
        name: Union[DebaterNames, str],
        debate_round: int,
        transcript: List[DebateResponse],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        answer_defending, answer_opposing = self._get_labels(name, **kwargs)

        context = DebateContext(
            situation=scenario.situation,
            question=scenario.question,
            name=name,
            answer_defending=answer_defending,
            answer_opposing=answer_opposing,
            word_limit=self.word_limit,
            round_number=debate_round,
            transcript=format_transcript(transcript),
        )

        return DebatePromptTemplate.create_prompt_messages(context)

    def _get_debater_response(
        self, name: str, messages: List[Dict[str, Any]], **kwargs
    ) -> str:
        """Get response from the LLM model for the given debater."""
        model = self.name_to_model[name]
        self.logger.debug("Getting response from model %s for debater %s", model, name)
        try:
            response = get_response(
                model,
                messages,
                tags=["debater"],
                session_id=kwargs.get("record_id", None),
                user_id=self.scenario.id,
                name=name,
            )
            return response
        except Exception as e:
            self.logger.error("Error getting response from %s: %s", model, str(e))
            raise

    def get_debater_positions(self, swap: bool, all_wrong: bool) -> Dict[str, str]:
        """Get the debater positions for the debate."""
        debater_positions = {}
        for name in self.debater_names:
            debater_positions[name] = self._get_labels(
                name, swap=swap, all_wrong=all_wrong
            )[0]
        return debater_positions

    @observe()
    def run(
        self, swap: bool = False, all_wrong: bool = False, cooldown: int = 10
    ) -> DebateRecord:
        """Run the debate with caching and return the record."""
        run_cache_key = generate_cache_key(
            self.cache_key.split(".")[0],
            swap=swap,
            all_wrong=all_wrong,
        )

        # Try loading from cache
        cached_record = load_from_cache(run_cache_key)
        if cached_record:
            return DebateRecord(**cached_record)

        # Create new record
        record = DebateRecord(
            scenario=self.scenario,
            debater_positions=self.get_debater_positions(swap, all_wrong),
            debater_models=self.name_to_model,
            swap=swap,
            all_wrong=all_wrong,
        )

        langfuse_context.update_current_trace(
            tags=["debate"],
            user_id=self.scenario.id,
            session_id=record.id,
        )

        # Run debate rounds
        for debate_round in range(1, self.max_debate_rounds + 1):
            self.logger.debug("Starting round %d", debate_round)
            for turn, name in enumerate(self.debater_names, start=1):
                self.logger.debug("Turn %d: %s's move", turn, name)
                messages = self._create_messages(
                    self.scenario,
                    name,
                    debate_round,
                    record.transcript,
                    swap=swap,
                    all_wrong=all_wrong,
                )
                self.logger.debug("Prompt messages: %s", messages)
                response = self._get_debater_response(
                    name, messages, record_id=record.id
                )
                arguments = extract_argument(response)
                validated = validate_citations(arguments, self.scenario.situation)
                record.transcript.append(
                    DebateResponse(
                        debate_round=debate_round,
                        turn=turn,
                        name=name,
                        response=response,
                        response_arguments=arguments,
                        validated_response_arguments=validated,
                    )
                )
                self.logger.debug("Completed turn %d for %s", turn, name)
            self.logger.debug("Completed round %d", debate_round)
            time.sleep(cooldown)

        # Cache and return
        save_to_cache(run_cache_key, record.to_dict())
        return record
