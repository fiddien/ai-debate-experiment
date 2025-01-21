"""
Shared type definitions for the debate system.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Literal, Dict, Tuple
from uuid import uuid4


class DebaterNames(Enum):
    """Enum for debater names."""

    A = "Debater A"
    B = "Debater B"

    def __str__(self):
        return f'{self.value}'


@dataclass(frozen=True)
class DebateScenario:
    """Structure of a debate scenario."""

    situation: str
    question: str
    answer_options: List[str]
    label: str
    id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self):
        """Return a JSON-able version of the data."""
        return asdict(self)

@dataclass
class DebateContext:
    """Context for debate prompt construction."""

    situation: str
    question: str
    name: str
    answer_defending: str
    answer_opposing: str
    word_limit: int
    transcript: str
    round_number: int = 1


@dataclass
class DebateResponse:
    """Structure of a debate transcript entry."""

    debate_round: int
    turn: int
    name: str
    response: str
    response_arguments: str
    validated_response_arguments: str


@dataclass
class DebateRecord:
    """Structure of a complete debate record."""

    scenario: DebateScenario
    debater_positions: Dict[str, Tuple[str, str]]
    debater_models: Dict[str, str]
    swap: bool = False
    all_wrong: bool = False
    id: str = field(default_factory=lambda: str(uuid4()))
    transcript: List[DebateResponse] = field(default_factory=list)

    def to_dict(self):
        """Return a JSON-able version of the data."""
        return asdict(self)


@dataclass
class JudgeContext:
    """Context for judge prompt construction."""

    question: str
    answer_a: str
    answer_b: str
    name_a: str
    name_b: str
    transcript: str


@dataclass
class JudgeResponse:
    """Structure of a judge's decision."""

    chosen_answer: Literal["A", "B"]
    reasoning: str
    confidence: float


@dataclass
class JudgementResult:
    """Structure of a judge's judgment result."""

    id: str
    judgment: str
    model: str

    def to_dict(self):
        """Return a JSON-able version of the data."""
        return asdict(self)
