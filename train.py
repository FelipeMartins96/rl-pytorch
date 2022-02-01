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

def info_to_log(info):
    return {
        'manager/goal': info['manager_weighted_rw'][0],
        'manager/ball_grad': info['manager_weighted_rw'][1],
        'manager/move': info['manager_weighted_rw'][2],
        'manager/collision': info['manager_weighted_rw'][3],
        'manager/energy': info['manager_weighted_rw'][4],
        'worker/dist': info['workers_weighted_rw'][0][0],
        'worker/energy': info['workers_weighted_rw'][0][1],
    }


def run_validation_ep(m_agent, w_agent, env, opponent_policies):
    m_obs = env.reset()
    done = False
    while not done:
        m_action = m_agent.get_action(torch.Tensor(m_obs))
        w_obs = env.set_action_m(m_action)
        w_action = w_agent.get_action(torch.Tensor(w_obs))
        step_action = np.stack([w_action] + [[p()] for p in opponent_policies], axis=0)
        _obs, _, done, _ = env.step(step_action)
        m_obs = _obs.manager


def main(args):
    wandb.init(
        mode=args.wandb_mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        monitor_gym=args.wandb_monitor_gym,
        config=args,
    )

    args.env_name
    total_training_steps = (
        args.training_total_training_steps + args.training_replay_min_size
    )
    replay_capacity = args.training_total_training_steps + args.training_replay_min_size
    min_replay_size = args.training_replay_min_size - 90000
    batch_size = args.training_batch_size
    gamma = args.training_gamma
    learning_rate = args.training_learning_rate
    seed = args.seed

    env = gym.make(
        args.env_name,
        n_robots_blue=args.env_n_robots_blue,
        n_robots_yellow=args.env_n_robots_yellow,
    )
    if args.training_val_frequency:
        val_env = gym.wrappers.RecordVideo(
            gym.make(
                args.env_name,
                n_robots_blue=args.env_n_robots_blue,
                n_robots_yellow=args.env_n_robots_yellow,
            ),
            './monitor/',
            episode_trigger=lambda x: True,
        )
    key = jax.random.PRNGKey(args.seed)
    if args.env_opponent_policy == 'off':
        opponent_policies = [
            lambda: np.array([0.0, 0.0]) for _ in range(args.env_n_robots_yellow)
        ]
    env.set_key(key)
    val_env.set_key(key)

    m_agent = DDPG(*env.get_spaces_m(), learning_rate, gamma)
    w_agent = DDPG(*env.get_spaces_w(), learning_rate, gamma)

    m_buffer = ReplayBuffer(*env.get_spaces_m(), replay_capacity)
    w_buffer = ReplayBuffer(*env.get_spaces_w(), replay_capacity)

    m_obs = env.reset()
    rewards, ep_steps, done, q_losses, pi_losses = 0, 0, False, [], []
    for step in tqdm(range(total_training_steps), smoothing=0.01):
        with torch.no_grad():
        # if args.training_val_frequency and step % args.training_val_frequency == 0:
            # run_validation_ep(m_agent, w_agent, val_env, opponent_policies)
            m_action = m_agent.sample_action(torch.Tensor(m_obs))
            w_obs = env.set_action_m(m_action)
            w_action = w_agent.get_action(torch.Tensor(w_obs))
            step_action = np.stack([w_action] + [[p()] for p in opponent_policies], axis=0)
            _obs, reward, done, info = env.step(step_action)

        terminal_state = False if not done or "TimeLimit.truncated" in info else True

        m_buffer.add(m_obs, m_action, 0.0, reward.manager, terminal_state, _obs.manager)

        if step > min_replay_size:
            m_batch = m_buffer.get_batch(batch_size)
            a,b = m_agent.update(m_batch)
            q_losses.append(a)
            pi_losses.append(b)

        rewards += reward.manager
        ep_steps += 1
        m_obs = _obs.manager
        if done:
            m_obs = env.reset()
            log = info_to_log(info)
            log.update(
                {
                    'ep_reward': rewards,
                    'ep_steps': ep_steps,
                    'q_loss': np.mean(q_losses),
                    'pi_loss': np.mean(pi_losses),
                }
            )
            wandb.log(log, step=step)
            rewards, ep_steps, q_losses, pi_losses = 0, 0, [], []


if __name__ == '__main__':
    parser = ArgumentParser(fromfile_prefix_chars='@')
    # RANDOM
    parser.add_argument('--seed', type=int, default=0)

    # WANDB
    parser.add_argument('--wandb-mode', type=str, default='disabled')
    parser.add_argument('--wandb-project', type=str, default='rsoccer-hrl')
    parser.add_argument('--wandb-entity', type=str, default='felipemartins')
    parser.add_argument('--wandb-name', type=str)
    parser.add_argument('--wandb-monitor-gym', type=bool, default=True)

    # ENVIRONMENT
    parser.add_argument('--env-name', type=str, default='VSSHRL-v0')
    parser.add_argument('--env-n-robots-blue', type=int, default=1)
    parser.add_argument('--env-n-robots-yellow', type=int, default=0)
    parser.add_argument('--env-opponent-policy', type=str, default='off')

    # TRAINING
    parser.add_argument('--training-total-training-steps', type=int, default=3000000)
    parser.add_argument('--training-replay-min-size', type=int, default=100000)
    parser.add_argument('--training-batch-size', type=int, default=256)
    parser.add_argument('--training-gamma', type=float, default=0.95)
    parser.add_argument('--training-learning-rate', type=float, default=1e-4)
    parser.add_argument('--training-val-frequency', type=int, default=100000)
    parser.add_argument('--training-load-worker', type=bool, default=True)

    args = parser.parse_args()
    main(args)
