"""Tests for LLM nutrition parsing (standalone utility, not part of action space)."""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from backend.llm_nutrition import parse_nutrition_text


def test_nutrition_parser():
    if not os.environ.get("OPENAI_API_KEY"):
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        os.environ[key] = value

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
