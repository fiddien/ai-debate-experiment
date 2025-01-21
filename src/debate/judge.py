"""Judge management functionality."""

import logging
from typing import Any, Dict, List, Optional

from src.debate.utils.response import format_transcript
from src.models.judge_prompt import JudgePromptTemplate
from src.models.llms import get_response

from .utils.cache import generate_cache_key, load_from_cache, save_to_cache
from .types import (
    DebateRecord,
    JudgeContext,
    JudgementResult,
)


class JudgeManager:
    """Manages judgment operations for debates."""

    def __init__(self, record: DebateRecord, judge_models: Optional[List[str]] = None):
        self.record = record
        self.judge_models = list(set(judge_models)) or []
        self.results: Dict[str, List[JudgementResult]] = {}
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
                messages,
                "judge",
                scenario_id=self.record.scenario.id,
                record_id=self.record.id,
            )
            self.logger.debug("Received response from %s", model)
            return response
        except Exception as e:
            self.logger.error("Error getting response from %s: %s", model, str(e))
            raise

    def run(self) -> Dict[str, List[JudgementResult]]:
        """Run judgment with caching and return results."""
        self.logger.info("Starting judgment phase")

        # Try loading from cache
        cached_results = load_from_cache(self.cache_key)
        if cached_results:
            self.logger.info("Loading results from cache")
            self.results = {
                model: [JudgementResult(**r) for r in results]
                for model, results in cached_results.items()
            }
            return self.results

        # Run normal judgment if not cached
        messages = self._create_judge_messages()

        for model in self.judge_models:
            self.logger.info("Processing model: %s", model)
            judgment = self._get_judge_response(model, messages)
            self.results[model] = [
                JudgementResult(
                    id=self.record.id,
                    judgment=judgment,
                    model=model,
                )
            ]
            self.logger.info("Completed judgment for model %s", model)

        # Save results to cache
        save_to_cache(
            self.cache_key,
            {
                model: [r.to_dict() for r in results]
                for model, results in self.results.items()
            },
        )

        return self.results

    def get_results(
        self, model: Optional[str] = None
    ) -> Dict[str, List[JudgementResult]]:
        """Get judgment results for specified model or all models."""
        self.logger.debug("Retrieving results for model: %s", model if model else "all")
        if model:
            return {model: self.results.get(model, [])}
        return self.results
