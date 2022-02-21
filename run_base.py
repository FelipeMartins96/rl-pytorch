from tqdm import tqdm
import gym
import rsoccer_gym
import jax
import wandb
import numpy as np
from argparse import ArgumentParser
from agent import DDPG
from buffer import ReplayBuffer
import torch
import time


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f'{device=}')
    total_training_steps = (
        args.training_total_training_steps + args.training_replay_min_size
    )
    replay_capacity = args.training_total_training_steps + args.training_replay_min_size
    min_replay_size = args.training_replay_min_size
    batch_size = args.training_batch_size
    gamma = args.training_gamma
    learning_rate = args.training_learning_rate
    seed = args.seed

    env = gym.make("LunarLanderContinuous-v2")
    env.seed(seed)


    agent = DDPG(env.observation_space, env.action_space, learning_rate, gamma, device)

    buffer = ReplayBuffer(env.observation_space, env.action_space, replay_capacity, device)

    obs = env.reset()
    for step in tqdm(range(total_training_steps), smoothing=0.01):
        with torch.no_grad():
            action = agent.sample_action(torch.Tensor(obs))
            _obs, reward, done, info = env.step(action)
            terminal_state = False if not done or "TimeLimit.truncated" in info else True
            buffer.add(obs, action, 0.0, reward, terminal_state, _obs)

        if step > min_replay_size:
            batch = buffer.get_batch(batch_size)
            agent.update(batch)

        obs = _obs
        if done:
            obs = env.reset()


if __name__ == '__main__':
    parser = ArgumentParser()

    # RANDOM
    parser.add_argument('--seed', type=int, default=0)

    # ENVIRONMENT

    # TRAINING
    parser.add_argument('--training-total-training-steps', type=int, default=75000)
    parser.add_argument('--training-replay-min-size', type=int, default=25000)
    parser.add_argument('--training-batch-size', type=int, default=256)
    parser.add_argument('--training-gamma', type=float, default=0.99)
    parser.add_argument('--training-learning-rate', type=float, default=3e-4)

    args = parser.parse_args()
    a = time.time()
    main(args)
    print(f'time = {time.time()-a}s')

