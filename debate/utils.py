import re
import string
import logging
from typing import List

from .types import DebateResponse

logger = logging.getLogger(__name__)

def extract_argument(text: str) -> str:
    """Extract the argument from the response text."""
    match = re.search(r"<argument>(.*?)</argument>", text, re.DOTALL)
    if not match:
        logger.error("No argument found in response text")
        return text
    return match.group(1)

def validate_citations(text: str, situation: str) -> str:
    """Validate citations in debate text and mark them as valid/invalid."""
    def remove_punctuation(text: str) -> str:
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator).lower().strip()

    source = remove_punctuation(situation)

    def validate_quote(match: re.Match) -> str:
        content = match.group(1).strip()
        clean_content = remove_punctuation(content)
        tag = "v_quote" if clean_content in source else "u_quote"
        return f"<{tag}>{content}</{tag}>"

    return re.sub(r"<quote>(.*?)</quote>", validate_quote, text)

def format_transcript(transcript: List[DebateResponse]) -> str:
    """Format the debate transcript into a string."""
    return "\n\n".join(f"{r.name}: {r.validated_response_arguments}" for r in transcript)
