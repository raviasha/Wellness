import os
import time
import torch
import numpy as np
from rl_training.env_wrapper import WellnessGymEnv
from rl_training.ppo_lite import PPOLite

def train(user_id: int = None, persona_path: str = None, distribution_path: str = None, total_steps: int = 50000):
    """Train a PPO agent. If user_id is given, trains a per-user model with calibrated persona.
    
    Pass distribution_path to use the Gaussian copula distribution simulator instead
    of the rule-based simulator. The two can be combined (persona_path and distribution_path
    can both be provided — persona determines compliance/goal/starting biomarkers, while
    the distribution replaces the per-biomarker response rules).
    """
    mode = "distribution" if distribution_path else "rules"
    print(f"Initializing environment (user_id={user_id}, simulator_mode={mode})...")
    
    env = WellnessGymEnv(
        task_name="personal_coaching",
        persona_path=persona_path,
        distribution_path=distribution_path,
    )
    
    # Environment info
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Stats: State Dim={state_dim}, Action Dim={action_dim}")
    if persona_path:
        print(f"Using calibrated persona: {persona_path}")
    
    # Hyperparameters
    update_timestep = 2048      # update policy every n timesteps
    K_epochs = 10               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor
    lr_actor = 0.0003           # learning rate for actor network
    lr_critic = 0.001           # learning rate for critic network
    
    ppo_agent = PPOLite(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    
    os.makedirs("models", exist_ok=True)
    
    time_step = 0
    episode_reward = 0
    
    state, _ = env.reset()
    
    print(f"Training PPO-Lite Agent ({total_steps} steps)...")
    
    start_time = time.time()
    
    while time_step < total_steps:
        # Select action
        action = ppo_agent.select_action(state)
        state, reward, done, truncated, info = env.step(action)
        
        # Saving reward and is_terminals
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)
        
        time_step += 1
        episode_reward += reward
        
        # update PPO agent
        if time_step % update_timestep == 0:
            ppo_agent.update()
            
        if time_step % 5000 == 0:
            elapsed = time.time() - start_time
            print(f"Step: {time_step} | Time: {elapsed:.1f}s | Latest Episode Reward: {episode_reward:.1f}")
            
        if done:
            state, _ = env.reset()
            episode_reward = 0

    # Save model
    if user_id is not None:
        model_dir = os.path.join("models", f"user_{user_id}")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "ppo_wellness_lite.pt")
    else:
        model_path = "models/ppo_wellness_lite.pt"
    
    ppo_agent.save(model_path)
    print(f"\nModel saved to {model_path}")
    print(f"Total training time: {time.time() - start_time:.1f}s")
    return model_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO wellness agent")
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--persona-path", type=str, default=None)
    parser.add_argument("--distribution-path", type=str, default=None,
                        help="Path to distribution.json from calibrate_user_distribution(). "
                             "Enables the Gaussian copula simulator.")
    parser.add_argument("--total-steps", type=int, default=50000)
    args = parser.parse_args()
    train(
        user_id=args.user_id,
        persona_path=args.persona_path,
        distribution_path=args.distribution_path,
        total_steps=args.total_steps,
    )
