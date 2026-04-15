import os
import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from backend.llm_nutrition import parse_nutrition_text
from wellness_env.models import NutritionType

def test_nutrition_parser():
    # Attempt to load .env manually if not already set
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
        "Nothing today, I fasted until dinner",
        "A balanced bowl with salmon, rice, and avocado"
    ]

    for text in test_cases:
        print(f"\nInput: {text}")
        result = parse_nutrition_text(text)
        print(f"Result: {result}")
        
        assert "calories" in result
        assert "protein_g" in result
        assert "carbs_g" in result
        assert "fat_g" in result
        assert "nutrition_type" in result
        assert result["nutrition_type"] in [t.value for t in NutritionType]

if __name__ == "__main__":
    try:
        test_nutrition_parser()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
