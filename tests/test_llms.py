import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.llms import get_response, provider_of, translate_to_google_schema


class TestLLMs(unittest.TestCase):
    def setUp(self):
        self.test_messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
        ]

    def test_translate_to_google_schema(self):
        result = translate_to_google_schema(self.test_messages)
        expected = [
            {"role": "user", "parts": [{"text": "You are a helpful assistant"}]},
            {"role": "user", "parts": [{"text": "Hello"}]},
            {"role": "model", "parts": [{"text": "Hi there"}]},
            {"role": "user", "parts": [{"text": "How are you?"}]},
        ]
        self.assertEqual(result, expected)

    @patch("google.generativeai.GenerativeModel")
    def test_google_response(self, mock_model):
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5

        mock_chat.send_message.return_value = mock_response
        mock_model.return_value.start_chat.return_value = mock_chat

        result = get_response("gemini-1.5-flash", self.test_messages)
        self.assertEqual(result, "Test response")

    @patch("models.llms.Anthropic")
    def test_anthropic_response(self, mock_anthropic_class):
        # Create mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5

        # Setup mock client
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        result = get_response("claude-3-haiku-20240307", self.test_messages)
        self.assertEqual(result, "Test response")

        # Verify the mock was called correctly
        mock_client.messages.create.assert_called_once()

    @patch("openai.OpenAI")
    def test_openai_response(self, mock_openai):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Test response"))]
        mock_response.usage.input_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = get_response("gpt-4o-mini", self.test_messages)
        self.assertEqual(result, "Test response")


if __name__ == "__main__":
    unittest.main()
