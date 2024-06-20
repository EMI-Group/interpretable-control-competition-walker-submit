from gymnasium.envs.registration import register

register(
    id="2048-v0",
    entry_point="discrete.env_2048.env2048:Env2048"
)
