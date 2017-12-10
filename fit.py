import os
import json
import numpy as np
import imageio
import matplotlib.pyplot as plt
from Environment_MarsLanding import MarsLanderEnvironment
from Agent import DQNAgent


def run(episods_nb=10000, render=False, save_episodes=False):
    env = MarsLanderEnvironment()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(action_size, state_size)
    batch_size = 32
    steps = []
    rewards = []
    losses = []
    coords = {}
    try:
        for episode in range(1, episods_nb+1):
            frames = []
            print("\nEpisode: {}".format(episode))
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            done = False
            step = 0
            total_reward = 0
            while not done:
                step += 1
                action = agent.act(state)
                new_state, reward, done, info = env.step(action)
                new_state = np.reshape(new_state, [1, state_size])
                if render:
                    env.render()
                if save_episodes and not episode % 100:
                    frames.append(env.render(mode='rgb_array'))

                agent.remember(state, action, reward, new_state, done)
                state = new_state

                total_reward += reward
                if done or step % 10 == 0:
                    print(["{:+0.2f}".format(x) for x in state.flatten()])
                    print("Step {}: total reward = {:+0.2f}".format(step, total_reward))
                # Save lander coordinates (x, y, velX, velY and angle)
                try:
                    coords[episode].append(info)
                except:
                    coords[episode] = [info]
            if len(frames) != 0:
                imageio.mimsave(os.path.join("imgs", str(episode) + ".gif"), frames)
            steps.append(step)
            rewards.append(total_reward)
            loss = agent.partial_fit(batch_size)
            losses.append(loss)
    finally:
        agent.save_model()
        save_stats(steps, "Step")
        save_stats(rewards, "Reward")
        save_stats(losses, "Loss")
        with open('coords.json', 'w') as f:
            json.dump(coords, f)



def save_stats(y, title=""):
    x = np.arange(1, len(y) + 1)
    plt.figure(figsize=(18, 6))
    plt.plot(x, y)
    plt.xlabel(title, fontsize=18)
    plt.ylabel('Epoch', fontsize=16)
    plt.grid(b=True, which='major', color='k', linestyle='--')
    plt.savefig(os.path.join("imgs", title + ".jpg"))



if __name__ == '__main__':
    run(episods_nb=200, save_episodes=True)
