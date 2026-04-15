import os
import json
from openai import OpenAI
from wellness_env.models import NutritionType

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
    
    nutrition_types = [t.value for t in NutritionType]
    
    system_prompt = f"""
    You are a nutrition expert. Your task is to convert natural language food logs into structured macro estimates.
    Return ONLY a JSON object with the following fields:
    - calories: total estimated calories (integer)
    - protein_g: estimated protein in grams (integer)
    - carbs_g: estimated carbohydrates in grams (integer)
    - fat_g: estimated fat in grams (integer)
    - nutrition_type: one of the following strings based on the description: {nutrition_types}
    
    Nutrition Type Guidelines:
    - high_protein: High ratio of protein (e.g., meat, eggs, shakes).
    - high_carb: High ratio of carbs (e.g., pasta, rice, bread, sugary foods).
    - balanced: Good mix of macros.
    - processed: Junk food, fast food, highly processed snacks.
    - skipped: Explicitly mentions skipping meals or fasting.
    
    Example: 
    Input: "I had a chicken breast and some broccoli"
    Output: {{"calories": 350, "protein_g": 50, "carbs_g": 10, "fat_g": 10, "nutrition_type": "high_protein"}}
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
            "nutrition_type": result.get("nutrition_type", "balanced")
        }
    except Exception as e:
        print(f"Error parsing nutrition: {e}")
        return {
            "calories": 0,
            "protein_g": 0,
            "carbs_g": 0,
            "fat_g": 0,
            "nutrition_type": "balanced",
            "error": str(e)
        }
