from gymnasium.envs.registration import register

register(
     id="SimulatorEnv-v1",
     entry_point="gym_env.envs:MultiCarEnv",
     max_episode_steps=600,
)