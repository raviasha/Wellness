import os
import json
from openai import OpenAI

# Nutrition types for LLM classification (standalone, not in action space)
_NUTRITION_TYPES = ["high_protein", "high_carb", "balanced", "processed", "skipped"]

def parse_nutrition_text(text: str) -> dict:
    """
    Parses natural language food logs into structured nutrition data using GPT-4o-mini.
    
    Returns:
        dict: {
            "calories": int,
            "protein_g": int,
            "carbs_g": int,
            "fat_g": int,
            "nutrition_type": str (one of NutritionType values)
        }
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    
    nutrition_types = _NUTRITION_TYPES
    
    system_prompt = f"""
    You are a nutrition expert. Your task is to convert natural language food logs into structured macro estimates.
    Return ONLY a JSON object with the following fields:
    - calories: total estimated calories (integer)
    - protein_g: estimated protein in grams (integer)
    - carbs_g: estimated carbohydrates in grams (integer)
    - fat_g: estimated fat in grams (integer)
    - nutrition_type: one of the following strings based on the description: {nutrition_types}
    - quality_score: an integer from 0 to 10 representing overall nutritional quality (10 = whole foods, micronutrient-rich, well-balanced; 5 = average mixed diet; 0 = skipped meals or pure junk food)
    
    Nutrition Type Guidelines:
    - high_protein: High ratio of protein (e.g., meat, eggs, shakes).
    - high_carb: High ratio of carbs (e.g., pasta, rice, bread, sugary foods).
    - balanced: Good mix of macros.
    - processed: Junk food, fast food, highly processed snacks.
    - skipped: Explicitly mentions skipping meals or fasting.

    Quality Score Guidelines:
    - 9-10: Whole foods, lots of vegetables, lean protein, healthy fats, low sugar
    - 7-8: Mostly healthy with minor indulgences
    - 5-6: Mixed diet, moderate processed food
    - 3-4: Mostly processed, fast food, low vegetables
    - 0-2: Junk food only, skipped meals, or very poor nutrition
    
    Example: 
    Input: "I had a chicken breast and some broccoli"
    Output: {{"calories": 350, "protein_g": 50, "carbs_g": 10, "fat_g": 10, "nutrition_type": "high_protein", "quality_score": 9}}
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Ensure all fields are present and typed correctly
        return {
            "calories": int(result.get("calories", 0)),
            "protein_g": int(result.get("protein_g", 0)),
            "carbs_g": int(result.get("carbs_g", 0)),
            "fat_g": int(result.get("fat_g", 0)),
            "nutrition_type": result.get("nutrition_type", "balanced"),
            "quality_score": int(result.get("quality_score", 5))
        }
    except Exception as e:
        print(f"Error parsing nutrition: {e}")
        return {
            "calories": 0,
            "protein_g": 0,
            "carbs_g": 0,
            "fat_g": 0,
            "nutrition_type": "balanced",
            "quality_score": 5,
            "error": str(e)
        }


def decide_food_action(new_text: str, existing_entries: list[dict]) -> dict:
    """
    Given the new food description and a list of existing food entries for that day,
    asks the LLM whether the new entry is a duplicate of an existing one (overwrite)
    or a new separate meal (append).

    existing_entries: list of {"id": int, "text": str}

    Returns: {"action": "overwrite", "target_id": <int>} | {"action": "append"}
    """
    if not existing_entries:
        return {"action": "append"}

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = os.environ.get("MODEL_NAME", "gpt-4o-mini")

    existing_str = "\n".join(
        f'  Entry {e["id"]}: "{e["text"]}"' for e in existing_entries
    )

    system_prompt = (
        "You are a food log deduplication assistant. "
        "A user is adding a new food entry for the same day. "
        "Decide if the new entry describes the SAME meal as one of the existing entries "
        "(same foods, just re-worded or with a minor correction) or a DIFFERENT meal "
        "(clearly different food items, different time of day, additional meal).\n\n"
        "Return ONLY a JSON object:\n"
        '- If it is the same meal as an existing entry: {"action": "overwrite", "target_id": <entry_id>}\n'
        '- If it is a brand-new separate meal: {"action": "append"}\n\n'
        "Be generous in treating entries as duplicates — if the core foods are the same "
        "even if quantities or wording differ slightly, treat it as an overwrite."
    )

    user_msg = (
        f"New entry: \"{new_text}\"\n\n"
        f"Existing entries for today:\n{existing_str}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        action = result.get("action", "append")
        if action == "overwrite" and "target_id" in result:
            return {"action": "overwrite", "target_id": int(result["target_id"])}
        return {"action": "append"}
    except Exception as e:
        print(f"decide_food_action error: {e}")
        return {"action": "append"}
