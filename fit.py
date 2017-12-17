import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import matplotlib.pyplot as plt
from Environment_MarsLanding import MarsLanderEnvironment
from Agent import DQNAgent


def run(episods_nb=1000, save_each_n_episodes=10, render=False, save_info=False):
    env = MarsLanderEnvironment()   # Asign an environment
    state_size = env.observation_space.shape[0]    # Get states amount from the environment
    action_size = env.action_space.n    # And actions amount from the environment
    agent = DQNAgent(action_size, state_size)    # Determine an agent
    batch_size = 64
    close_episodes = 2  # How many neighboring episodes before and after should also be displayed
    # Data to visualize the learning process
    steps = []
    rewards = []
    losses = []
    coords = {}
    try:
        for episode in range(1, episods_nb+1):   # Iterate episodes
            frames = []   # Remember environment states to save a gif for episodes
            print("\nEpisode: {}".format(episode))
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            step = 0
            total_reward = 0
            while True:    # Do some actions until an condition of episode termination is achieved
                step += 1
                action = agent.act(state)
                new_state, reward, done, info = env.step(action)
                new_state = np.reshape(new_state, [1, state_size])
                if render:
                    env.render()
                info["episode_label"] = "Episode #" + str(episode)
                if np.array([not i % save_each_n_episodes for i in range(episode-close_episodes, episode+close_episodes+1)]).any():
                    img = add_text_labels(env.render(mode='rgb_array'), info)
                    frames.append(img)
                # Add experience to memory
                agent.remember(state, action, reward, new_state, done)
                state = new_state
                total_reward += reward
                if done or step % 20 == 0:
                    print(["{:+0.2f}".format(x) for x in state.flatten()])
                    print("Step {}: total reward = {:+0.2f}".format(step, total_reward))
                # Save lander coordinates (x, y, velX, velY and angle)
                if save_info:
                    try: coords[episode].append(info)
                    except: coords[episode] = [info]
                if done:
                    break
            if len(frames) != 0:
                # To save in mp4 format change the extension from gif to mp4 and install ffmpeq using this line once
                # imageio.plugins.ffmpeg.download()
                imageio.mimsave(os.path.join("imgs", str(episode) + ".gif"), frames, fps=20)
            # Save statistics
            steps.append(step)
            rewards.append(total_reward)
            loss = agent.partial_fit(batch_size)
            losses.append(loss)
    finally:
        agent.save_model()
        # Plot the charts
        save_stats(steps, "Step")
        save_stats(rewards, "Reward")
        save_stats(losses, "Loss")
        if save_info:
            with open('coords.json', 'w') as f:
                json.dump(coords, f)



def add_text_labels(img, labels={}):
    # Adds text labels to an episode
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 15)
    draw.text((10, 5), labels.get("wind_label", ""), font=font, fill=(0, 0, 0))
    draw.text((275, 5), labels.get("episode_label", ""), font=font, fill=(0, 0, 0))
    draw.text((500, 5), labels.get("fuel_label", ""), font=font, fill=(0, 0, 0))
    ImageDraw.Draw(img)
    return np.asarray(img)

def save_stats(y, title=""):
    # Build a plot with a defined statistic
    x = np.arange(1, len(y) + 1)
    plt.figure(figsize=(18, 6))
    plt.plot(x, y)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel(title, fontsize=16)
    plt.grid(b=True, which='major', color='k', linestyle='--')
    plt.savefig(os.path.join("imgs", title + ".jpg"))



if __name__ == '__main__':
    run(episods_nb=200, save_each_n_episodes=100)
