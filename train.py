#!/usr/bin/env python3
import numpy as np
import torch as th
import time
import gymnasium as gym

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecEnv

from rl_with_gnns.policy import MaskableGraphActorCriticPolicy
from rl_with_gnns.util import get_clean_kwargs, change_obs_action_space


def train_ppo(train_env: VecEnv, val_env: VecEnv, config: dict, run_id: int):
    ppo_kwargs = get_clean_kwargs(
        MaskablePPO.__init__,
        warn=False,
        kwargs=config["PPO"],
    )

    # Create the PPO model
    model = MaskablePPO(
        MaskableGraphActorCriticPolicy,
        train_env,
        **ppo_kwargs,
        policy_kwargs=config["policy_kwargs"],
        tensorboard_log=f"runs/{run_id}",
    )

    # Evaluate the model periodically during training and save the best model
    eval_callback = MaskableEvalCallback(
        eval_env=val_env,
        n_eval_episodes=config["n_val_episodes"],
        eval_freq=max(config["val_freq"] // train_env.num_envs, 1),
        best_model_save_path=f"models/{run_id}",
        deterministic=True,
        render=False,
    )

    # Train the model
    model.learn(
        total_timesteps=config["PPO"]["timesteps"],
        progress_bar=True,
        callback=eval_callback,
    )


def evaluate(run_id, test_env: VecEnv, config: dict):
    # Load the best model
    model = MaskablePPO.load(f"models/{run_id}/best_model.zip")

    # Update the action/observation spaces of the model to match the eval env
    model.policy = change_obs_action_space(model.policy, test_env)

    # Evaluate the trained model
    ep_rewards, ep_lengths = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        render=False,
        use_masking=True,
        return_episode_rewards=True,
    )

    print("Test Results: ")
    print(f"Mean Reward: {np.mean(ep_rewards)} +/- {np.std(ep_rewards)}")
    print(f"Mean Episode Length: {np.mean(ep_lengths)} +/- {np.std(ep_lengths)}")


def main():
    config = {
        "env": "MVCEnv-v0",
        "seed": 42,
        "n_val_episodes": 20,
        "val_freq": 1024,
        "num_envs": 1,
        "policy_kwargs": {
            "pooling_type": "mean",
            "embed_dim": 128,
            "network_kwargs": {"network": "GAT", "num_layers": 2},
        },
        "PPO": {
            "timesteps": 100000,
            "seed": 42,
            "learning_rate": 1e-5,
            "gamma": 1,
            "n_steps": 1024,
        },
        "eval_seed": 1,
        "n_eval_episodes": 100,
    }

    run_id = int(time.time())

    th.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    def make_env(split, idx):
        def _init():
            return gym.make(config["env"], split=split, seed=config["seed"] + idx)

        return _init

    num_envs = config.get("num_envs", 1)

    print("Constructing train env")
    train_env = VecMonitor(DummyVecEnv([make_env("train", i) for i in range(num_envs)]))
    print("Constructing val env")
    val_env = VecMonitor(DummyVecEnv([make_env("val", i) for i in range(num_envs)]))
    print("Constructing test env")
    test_env = VecMonitor(DummyVecEnv([make_env("test", config["eval_seed"])]))

    # Update policy kwargs with environment dimensions
    config["policy_kwargs"]["node_dim"] = train_env.observation_space[
        "node_features"
    ].shape[1]
    config["policy_kwargs"]["edge_dim"] = train_env.observation_space[
        "edge_features"
    ].shape[2]

    print("Starting PPO training...")
    # Train the policy using PPO
    train_ppo(train_env, val_env, config, run_id)

    print("Evaluating trained policy...")
    evaluate(run_id, test_env, config)


if __name__ == "__main__":
    main()
