import gym_env  # If this fails, install it with `pip install -e .` from \simulator\gym_env
import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("SimulatorEnv-v0")

    episode_rewards = []
    episode_lengths = []

    for episode in range(1000):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            next_obs, reward, terminated, truncated, info = env.step((1, 0))  # Take random action
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                print(f"Episode {episode} finished after {episode_length} steps with reward {episode_reward}")
                env.render()

    print(f"Mean reward: {sum(episode_rewards) / len(episode_rewards)}")
    print(f"Mean length: {sum(episode_lengths) / len(episode_lengths)}")

