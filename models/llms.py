"""
TODO: Add a description of the module.
"""

from os import environ

import google.generativeai as google_client
from anthropic import Anthropic
from openai import OpenAI

google_client.configure(api_key=environ.get("GOOGLE_API_KEY"))
anthropic_client = Anthropic(api_key=environ.get("ANTHROPIC_API_KEY"))
openai_client = OpenAI(api_key=environ.get("OPENAI_API_KEY"))

provider_models = {
    "OpenAI": ["o1", "o1-mini", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "Anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
        "claude-2.1",
    ],
    "Google": [
        "gemini-1.5-pro",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
        "gemini-1.0-pro",
    ],
    "DeepSeek": ["deepseek-chat"],
    "Groq": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
    ],
}

provider_of = {
    model: provider for provider, models in provider_models.items() for model in models
}

__all__ = ['get_response', 'provider_models', 'provider_of']

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


def get_response(model: str, messages):
    """Generate a response from the model."""
    if provider_of[model] == "Google":
        messages = translate_to_google_schema(messages)
        model = google_client.GenerativeModel(model)
        chat = model.start_chat(history=messages[:-1])
        response = chat.send_message(messages[-1])
        return response.text

    elif provider_of[model] == "Anthropic":
        if messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]
        else:
            system_message = ""
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=3094,
            temperature=0.1,
            system=system_message,
            messages=messages,
        )
        return message.content[0].text

    elif provider_of[model] == "OpenAI":
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=3094,
            temperature=0.1,
        )
        return response.choices[0].message.text
    else:
        raise NotImplementedError(f"Model {model} is not supported.")
