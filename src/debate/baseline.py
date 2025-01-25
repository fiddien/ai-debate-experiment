"""Baseline evaluation functionality."""

import logging
from typing import Any, Dict, List
from langfuse.decorators import observe, langfuse_context

from src.models.baseline_prompt import BaselinePromptTemplate
from src.models.llms import get_response
from .types import DebateScenario, JudgementResult
from .utils.cache import generate_cache_key, load_from_cache, save_to_cache


class BaselineManager:
    """Manages direct model evaluations without debate."""

    def __init__(self, scenario: DebateScenario, models: List[str]):
        self.scenario = scenario
        self.models = list(set(models))
        self.results: Dict[str, JudgementResult] = {}  # model -> result
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "Initialized BaselineManager for scenario %s with %d models",
            scenario.id,
            len(self.models),
        )
        self.cache_key = generate_cache_key(
            "baseline", scenario_id=scenario.id, models=self.models
        )

    @observe()
    def run(self) -> Dict[str, dict]:
        """Run baseline evaluation with caching and return results."""
        self.logger.debug("Starting baseline evaluation")

        # Try loading from cache
        cached_results = load_from_cache(self.cache_key)
        if cached_results:
            self.logger.debug("Loading results from cache")
            self.results = cached_results
            return cached_results

        langfuse_context.update_current_trace(
            tags=["baseline"],
            user_id=self.scenario.id,
        )

        # Run evaluation
        messages = BaselinePromptTemplate.create_prompt_messages(self.scenario)

        for model in self.models:
            self.logger.debug("Processing model: %s", model)
            judgement = self._get_response(model, messages)
            self.results[model] = JudgementResult(
                id=self.scenario.id,
                judgement=judgement,
                model=model,
            )
            self.logger.info("Completed baseline answer from model %s", model)

        # Cache results
        save_to_cache(
            self.cache_key,
            {model: result.to_dict() for model, result in self.results.items()},
        )

        return {model: result.to_dict() for model, result in self.results.items()}

    def _get_response(self, model: str, messages: List[Dict[str, Any]]) -> str:
        """Get response from specified model."""
        self.logger.debug("Getting baseline response from model: %s", model)
        try:
            response = get_response(
                model,
                messages,
                tags=["baseline"],
                user_id=self.scenario.id,
            )
            self.logger.debug("Received baseline response from %s", model)
            return response
        except Exception as e:
            self.logger.error("Error getting response from %s: %s", model, str(e))
            raise

    def get_results(self, model: str = None) -> Dict[str, JudgementResult]:
        """Return the baseline evaluation results for specified model or all models."""
        self.logger.debug("Retrieving results for model: %s", model if model else "all")
        if model:
            result = self.results.get(model)
            return {model: result} if result else {}
        return self.results
