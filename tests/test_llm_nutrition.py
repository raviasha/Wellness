"""Tests for LLM nutrition parsing (standalone utility, not part of action space)."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.llm_nutrition import parse_nutrition_text


from unittest.mock import patch, MagicMock

@patch("backend.llm_nutrition.OpenAI")
def test_nutrition_parser(mock_openai):
    # Mock the OpenAI client and response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Configure mock response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"calories": 350, "protein_g": 50, "carbs_g": 10, "fat_g": 10, "nutrition_type": "high_protein", "quality_score": 9}'))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    test_cases = [
        "I ate a chicken breast and some broccoli",
        "Just a protein shake",
        "I had a huge pizza and 3 beers for dinner",
    ]

    for text in test_cases:
        result = parse_nutrition_text(text)
        assert "calories" in result
        assert "protein_g" in result


if __name__ == "__main__":
    try:
        test_nutrition_parser()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
