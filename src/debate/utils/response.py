import re
import string
import logging
from typing import List

from ..types import DebateResponse

logger = logging.getLogger(__name__)

def extract_argument(text: str) -> str:
    """
    Extract the argument from the response text.
    Example text:
    "<thinking>Some thinkings...</thinking>\n\n<argument>Some argument</argument>"
    """
    match = re.search(r"<argument>(.*?)</argument>", text, re.DOTALL)

    if match:
        return match.group(1).strip()

    logger.warning("Incomplete argument tags found in response text")

    match = re.search(r"<argument>(.*?)(</argument>|$)", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Extract only the argument tags, including the tags themselves
    split_ = text.split("<argument>")
    if len(split_) > 1:
        split_ = split_[1].split("</argument>")
        if len(split_) > 1:
            return split_[0].strip()
        return split_[0].strip()

    split_ = text.split("</thinking>")
    if len(split_) > 1:
        return split_[1].strip()

    logger.warning("No argument found in response text, return full text: %s", text)
    return text

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
