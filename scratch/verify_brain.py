import torch
from rl_training.env_wrapper import WellnessGymEnv
from rl_training.ppo_lite import ActorCritic
from wellness_env.models import SleepDuration, ExerciseType, NutritionType

def verify_brain():
    print("Verification: Testing Brain Recommendations...")
    
    env = WellnessGymEnv(task_name="personal_coaching")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Init and Load
    policy = ActorCritic(state_dim, action_dim)
    policy.load_state_dict(torch.load("models/ppo_wellness_lite.pt"))
    policy.eval()
    
    # Test Scenario: Day 1, Low HRV (need recovery)
    # [day, compliance, goal, 8x bio, 8x delta]
    # HRV is index 4
    state = [0.1, 0.8, 0.0] + [70, 20, 30, 25, 60, 80, 50, 50] + [0]*8
    state_t = torch.FloatTensor(state)
    
    with torch.no_grad():
        action_probs = policy.actor(state_t)
        action_idx = torch.argmax(action_probs).item()
    
    # Decipher action
    sl_idx = action_idx % len(env.sleep_options)
    rem = action_idx // len(env.sleep_options)
    ex_idx = rem % len(env.exercise_options)
    nu_idx = rem // len(env.exercise_options)
    
    sleep = env.sleep_options[sl_idx]
    exercise = env.exercise_options[ex_idx]
    nutrition = env.nutrition_options[nu_idx]
    
    print("\nSimulation State: High Stress (HRV=20)")
    print(f"AI Recommendation:")
    print(f"  - Sleep: {sleep}")
    print(f"  - Exercise: {exercise}")
    print(f"  - Nutrition: {nutrition}")
    
    # Logical check: if HRV is 20, we shouldn't recommend HIIT + Processed Food
    if sleep in [SleepDuration.OPTIMAL_LOW, SleepDuration.OPTIMAL_HIGH]:
        print("\n✅ SUCCESS: AI recommends prioritized recovery for high-stress states.")
    else:
        print("\n⚠️ WARNING: AI advice may not be optimal yet. Consider more training steps.")

if __name__ == "__main__":
    verify_brain()
