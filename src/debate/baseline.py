"""Baseline evaluation functionality."""

import logging
from typing import Any, Dict, List

from models.baseline_prompt import BaselinePromptTemplate
from models.llms import get_response

from .utils.cache import generate_cache_key, load_from_cache, save_to_cache
from .types import DebateScenario, JudgementResult


class BaselineManager:
    """Manages direct model evaluations without debate."""

    def __init__(self, scenario: DebateScenario, models: List[str]):
        self.scenario = scenario
        self.models = set(models)  # Use set to ensure uniqueness
        self.logger = logging.getLogger(__name__)
        self.cache_key = generate_cache_key(
            "baseline", scenario_id=scenario.id, models=self.models
        )

    def run(self) -> Dict[str, List[JudgementResult]]:
        """Run baseline evaluation with caching and return results."""
        # Try loading from cache
        cached_results = load_from_cache(self.cache_key)
        if cached_results:
            return {
                model: [JudgementResult(**r) for r in results]
                for model, results in cached_results.items()
            }

        # Run evaluation
        results = {}
        messages = BaselinePromptTemplate.create_prompt_messages(self.scenario)

        for model in self.models:
            judgment = self._get_response(model, messages)
            results[f"{model}_baseline"] = [
                JudgementResult(
                    id=self.scenario.id,
                    judgment=judgment,
                    model=model,
                )
            ]

        # Cache results
        save_to_cache(
            self.cache_key,
            {model: [r.to_dict() for r in res] for model, res in results.items()},
        )

        return results

    def _get_response(self, model: str, messages: List[Dict[str, Any]]) -> str:
        """Get response from specified model."""
        self.logger.debug("Getting baseline response from model: %s", model)
        try:
            response = get_response(
                model, messages, "baseline", scenario_id=self.scenario.id
            )
            self.logger.debug("Received baseline response from %s", model)
            return response
        except Exception as e:
            self.logger.error("Error getting response from %s: %s", model, str(e))
            raise
