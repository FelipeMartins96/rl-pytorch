import numpy as np
from networks import DDPGActor, DDPGCritic, TargetActor, TargetCritic
import torch
import torch.nn.functional as F

class DDPG:
    def __init__(self, obs_space, action_space, lr, gamma, device):
        obs_size, act_size = obs_space.shape[0], action_space.shape[0]

        self.actor = DDPGActor(obs_size, act_size).to(device)
        self.critic = DDPGCritic(obs_size, act_size).to(device)
        self.tgt_actor = DDPGActor(obs_size, act_size).to(device)
        self.tgt_critic = DDPGCritic(obs_size, act_size).to(device)
        self.tgt_actor.load_state_dict(self.actor.state_dict())
        self.tgt_critic.load_state_dict(self.critic.state_dict())
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.sigma = 0.2
        self.tau = 0.005

        pass

    def sample_action(self, obs):
        action = self.actor(obs).detach().numpy()
        noise = np.random.normal(size=action.shape) * self.sigma
        return np.clip(action + noise, -1, 1)

    def get_action(self, obs):
        return self.actor(obs).detach().numpy()

    def update(self, batch):
        S_v, A_v, r_v, dones, S_next_v = batch

        # train critic
        self.critic_optim.zero_grad()
        Q_v = self.critic(S_v, A_v)  # expected Q for S,A
        A_next_v = self.tgt_actor(S_next_v)  # Get an Bootstrap Action for S_next
        Q_next_v = self.tgt_critic(S_next_v, A_next_v)  # Bootstrap Q_next
        Q_next_v[dones == 1.] = 0.0  # No bootstrap if transition is terminal
        # Calculate a reference Q value using the bootstrap Q
        Q_ref_v = r_v + Q_next_v * self.gamma
        Q_loss_v = F.mse_loss(Q_v, Q_ref_v.detach())
        Q_loss_v.backward()
        self.critic_optim.step()
        critic_loss = Q_loss_v.cpu().detach().numpy()

        # train actor - Maximize Q value received over every S
        self.actor_optim.zero_grad()
        A_cur_v = self.actor(S_v)
        pi_loss_v = -self.critic(S_v, A_cur_v)
        pi_loss_v = pi_loss_v.mean()
        pi_loss_v.backward()
        self.actor_optim.step()
        actor_loss = pi_loss_v.cpu().detach().numpy()

        # Sync target networks
        for param, target_param in zip(self.actor.parameters(), self.tgt_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.tgt_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss, actor_loss