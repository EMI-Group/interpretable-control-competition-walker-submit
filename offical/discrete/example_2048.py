import gymnasium as gym
from discrete.discrete_controller import RandomDiscreteController

if __name__ == '__main__':
    env = gym.make("2048-v0", render_mode='terminal', seed=1)

    # random controller: it picks random actions at every step
    controller = RandomDiscreteController(seed=1)

    # evaluation loop: first reset, then iteration for episode_length steps
    observation, info = env.reset(seed=0)
    while True:
        action = controller.control(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated or truncated:
            break

    env.close()
