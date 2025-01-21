from gymnasium.envs.registration import register

register(
     id="SimulatorEnv-v0",
     entry_point="gym_env.envs:SimulatorEnv",
     max_episode_steps=600,
)