"""Judge management functionality."""

import logging
from typing import Any, Dict, List, Optional

from src.debate.utils.response import format_transcript
from src.models.judge_prompt import JudgePromptTemplate
from src.models.llms import get_response
from langfuse.decorators import observe, langfuse_context

from .utils.cache import generate_cache_key, load_from_cache, save_to_cache
from .types import (
    DebateRecord,
    JudgeContext,
    JudgementResult,
)


class JudgeManager:
    """Manages judgement operations for debates."""

    def __init__(self, record: DebateRecord, judge_models: Optional[List[str]] = None):
        self.record = record
        self.judge_models = list(set(judge_models)) or []
        self.results: Dict[str, JudgementResult] = {}
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "Initialized JudgeManager for debate %s with %d models",
            record.id,
            len(self.judge_models),
        )
        self.cache_key = generate_cache_key(
            "judge", record_id=record.id, models=judge_models or []
        )

    def _create_judge_messages(self) -> List[Dict[str, Any]]:
        """Create messages for the judge."""
        names = list(self.record.debater_positions.keys())
        answer_a = self.record.debater_positions[names[0]]
        answer_b = self.record.debater_positions[names[1]]
        context = JudgeContext(
            question=self.record.scenario.question,
            answer_a=answer_a,
            answer_b=answer_b,
            name_a=names[0],
            name_b=names[1],
            transcript=format_transcript(self.record.transcript),
        )

        return JudgePromptTemplate.create_prompt_messages(context)

    def _get_judge_response(self, model: str, messages: List[Dict[str, Any]]) -> str:
        """
        Get response from specified judge model.
        """
        self.logger.debug("Getting judge response from model: %s", model)
        try:
            response = get_response(
                model,
                messages
            )
            self.logger.debug("Received response from %s", model)
            return response
        except Exception as e:
            self.logger.error("Error getting response from %s: %s", model, str(e))
            raise


    @observe()
    def run(self) -> Dict[str, dict]:
        """Run judgement with caching and return results."""
        self.logger.debug("Starting judgement phase")

        # Try loading from cache
        cached_results = load_from_cache(self.cache_key)
        if cached_results:
            self.logger.debug("Loading results from cache")
            self.results = cached_results
            return cached_results

        langfuse_context.update_current_trace(
            tags=["judge"],
            session_id=self.record.id,
            user_id=self.record.scenario.id,
        )

        # Run normal judgement if not cached
        messages = self._create_judge_messages()

        for model in self.judge_models:
            self.logger.debug("Processing model: %s", model)
            judgement = self._get_judge_response(model, messages)
            self.results[model] = JudgementResult(
                id=self.record.id,
                judgement=judgement,
                model=model,
            )
            self.logger.info("Completed judgement from model %s", model)

        # Save results to cache
        save_to_cache(
            self.cache_key,
            {model: result.to_dict() for model, result in self.results.items()},
        )

        return {model: result.to_dict() for model, result in self.results.items()}

    def get_results(
        self, model: Optional[str] = None
    ) -> Dict[str, JudgementResult]:
        """Get judgement results for specified model or all models."""
        self.logger.debug("Retrieving results for model: %s", model if model else "all")
        if model:
            result = self.results.get(model)
            return {model: result} if result else {}
        return self.results
