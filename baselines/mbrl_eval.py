import sys
import gym
import numpy as np
import os

class MyLogger(object):
    def __init__(self, log_dir, log_name):
        self.log_name = log_name
        self.log_path = os.path.join(log_dir, f"{log_name}.npz")
        self.data = np.array([])

    def write(self, step, value):
        self.data = np.append(self.data, np.array([step, value]))
        np.savez(self.log_path, **{self.log_name: self.data})


def evaluate_ddpg_policy(env_id, agent, max_action, episode=100, random_seed=1000):
    env = gym.make(env_id)
    env.seed(random_seed)
    reward_history = []

    for i in range(episode):
        state = env.reset()
        done = False
        episode_return = 0
        time_step = 0
        while not done:
            # based-on DDPG code
            action, _, _, _ = agent.step(state, apply_noise=False, compute_Q=True)
            next_state, reward, done, info = env.step(max_action * action)

            state = next_state
            episode_return += reward
            sys.stdout.write(f"\rEvaluating episode {i}, at step {time_step}, eval rewards={episode_return}")
            sys.stdout.flush()
            time_step += 1
        reward_history.append(episode_return)

    return np.mean(reward_history)


def evaluate_ppo2_policy(env_id, model, episode=100, random_seed=1000):
    env = gym.make(env_id)
    env.seed(random_seed)
    reward_history = []

    M_states = None  # initial_state --> common/policies.py [Line 38]
    M_dones = np.array([False])

    for i in range(episode):
        state = np.array([env.reset()])
        done = False
        episode_return = 0
        time_step = 0
        while not done:

            # based-on PPO2 Runner code
            actions, _, M_states, _ = model.step(state, S=M_states, M=M_dones)
            state[:], rewards, M_dones, infos = env.step(actions)

            episode_return += np.max(rewards)
            sys.stdout.write(f"\rEvaluating episode {i}, at step {time_step}, eval rewards={episode_return}")
            sys.stdout.flush()
            time_step += 1
        reward_history.append(episode_return)

    return np.mean(reward_history)
