import gymnasium as gym

from continuous.continuous_controller import RandomContinuousController

if __name__ == '__main__':
    # the gym environment and the episode length are fixed to these values for the competition
    env = gym.make("Walker2d-v4", render_mode='human')
    episode_length = 1000

    # random controller: it picks random actions at every step
    controller = RandomContinuousController(env.action_space)

    # evaluation loop: first reset, then iteration for episode_length steps
    observation, info = env.reset(seed=0)
    for _ in range(episode_length):
        action = controller.control(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            break

    env.close()
