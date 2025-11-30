from gymnasium.envs.registration import register

register(
    id="TSPEnv-v0",
    entry_point="rl-with-gnns.env:TSPEnv",
)
