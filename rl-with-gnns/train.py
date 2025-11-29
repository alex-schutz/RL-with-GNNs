#!/usr/bin/env python3

import numpy as np
import torch as th
import time

from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sb3_contrib import MaskablePPO

from policy import MaskableGraphActorCriticPolicy
from util import make_train_env, make_eval_env
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from util import get_clean_kwargs, change_obs_action_space


def train_ppo(
    train_env: VecEnv, val_env: VecEnv, test_env: VecEnv, config: dict, run_id: int
):
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
        n_eval_episodes=config["n_eval_episodes"],
        eval_freq=max(config["eval_freq"] // train_env.num_envs, 1),
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
        "seed": 42,
        "policy_kwargs": {
            "network": "GAT",
            "num_layers": 2,
            "aggr": "mean",
        },
        "PPO": {
            "timesteps": 100000,
            "eval_freq": 10000,
            "seed": 42,
        },
    }

    run_id = int(time.time())

    th.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    print("Constructing train env")
    train_env = make_train_env(config)
    print("Constructing val env")
    val_env = make_eval_env(config, "val")
    print("Constructing test env")
    test_env = make_eval_env(config, "test")

    # Get the spec attribute from the first environment in vec_env
    env_spec = train_env.get_attr("graph_spec")[0]
    config["policy_kwargs"]["graph_spec"] = env_spec  # type: ignore

    print("Starting PPO training...")
    # Train the policy using PPO
    train_ppo(train_env, val_env, test_env, config, run_id)


if __name__ == "__main__":
    main()
