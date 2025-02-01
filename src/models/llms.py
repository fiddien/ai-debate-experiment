"""
TODO: Add a description of the module.
"""

import time
from functools import wraps
from os import environ

import google.generativeai as google_client
from anthropic import Anthropic
from anthropic import RateLimitError as AnthropicRateLimitError
from dotenv import load_dotenv
from langfuse.decorators import langfuse_context, observe
from openai import OpenAI
from openai import RateLimitError as OpenAIRateLimitError

load_dotenv()

google_client.configure(api_key=environ.get("GOOGLE_API_KEY"))
anthropic_client = Anthropic(api_key=environ.get("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=environ.get("OPENAI_API_KEY"))
deepseek_client = OpenAI(
    api_key=environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
)

provider_models = {
    "OpenAI": [
        "o1-2024-12-17",
        "o1-mini-2024-09-12",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini-2024-07-18",
    ],
    "Anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ],
    "Google": [
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ],
    "DeepSeek": ["deepseek-chat", "deepseek-reasoner"],
    "Groq": [
        "mixtral-8x7b-32768",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
    ],
}

provider_of = {
    model: provider for provider, models in provider_models.items() for model in models
}

__all__ = ["get_response", "provider_models", "provider_of"]


def translate_to_google_schema(messages):
    """Change the message format to Google Gemini schema."""
    google_schema = []
    for message in messages:
        if message["role"] == "system":
            role = "user"
        elif message["role"] == "assistant":
            role = "model"
        else:
            role = "user"
        content = message["content"]
        parts = [{"text": content}]
        google_schema.append({"role": role, "parts": parts})
    return google_schema


class RateLimitError(Exception):
    """Custom rate limit error."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        self.message = message
        super().__init__(f"{provider} API rate limit exceeded: {message}")


@observe(as_type="generation")
def get_response(model: str, messages, max_retries=3, initial_delay=10, **kwargs):
    """Generate a response from the model."""

    def retry_with_exponential_backoff(provider):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                delay = initial_delay
                last_exception = None

                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        is_rate_limit = (
                            isinstance(e, (OpenAIRateLimitError, AnthropicRateLimitError)) or
                            (provider == "Google" and "429" in str(e))
                        )

                        if is_rate_limit:
                            last_exception = RateLimitError(provider, str(e))
                            if attempt < max_retries - 1:
                                print(
                                    f"Rate limit hit for {provider}, retrying in {delay} seconds..."
                                )
                                time.sleep(delay)
                                delay *= 2
                                continue
                        raise e

                raise last_exception

            return wrapper

        return decorator

    langfuse_context.update_current_observation(
        model=model,
        input=messages,
        metadata=kwargs,
    )
    result = 0
    input_tokens = 0
    output_tokens = 0

    try:
        if model not in provider_of:
            raise ValueError(f"Model {model} is not supported.")

        if provider_of[model] == "Google":

            @retry_with_exponential_backoff("Google")
            def google_call():
                messages_schema = translate_to_google_schema(messages)
                model_client = google_client.GenerativeModel(model)
                chat = model_client.start_chat(history=messages_schema[:-1])
                return chat.send_message(messages_schema[-1])

            response = google_call()
            result = response.text
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count

        elif provider_of[model] == "Anthropic":

            @retry_with_exponential_backoff("Anthropic")
            def anthropic_call():
                system_message = (
                    messages[0]["content"] if messages[0]["role"] == "system" else ""
                )
                msgs = messages[1:] if messages[0]["role"] == "system" else messages
                return anthropic_client.messages.create(
                    model=model,
                    max_tokens=2000,
                    temperature=0.1,
                    system=system_message,
                    messages=msgs,
                )

            response = anthropic_call()
            result = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

        elif provider_of[model] == "OpenAI":

            @retry_with_exponential_backoff("OpenAI")
            def openai_call():
                return openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                )

            response = openai_call()
            if hasattr(response.choices[0].message, "reasoning_content"):
                reasoning = response.choices[0].message.reasoning_content
                print(f"Reasoning: {reasoning}")
            result = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        elif provider_of[model] == "DeepSeek":

            @retry_with_exponential_backoff("DeepSeek")
            def deepseek_call():
                return deepseek_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                )

            response = deepseek_call()
            if hasattr(response.choices[0].message, "reasoning_content"):
                reasoning = response.choices[0].message.reasoning_content
                print(f"Reasoning: {reasoning}")
            result = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        langfuse_context.update_current_observation(
            usage_details={"input": input_tokens, "output": output_tokens}
        )
        return result
    except Exception as e:
        print(f"Error getting response from {model}: {str(e)}")
        raise e
