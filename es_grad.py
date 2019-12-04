from copy import deepcopy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time

import gym
import gym.spaces
import numpy as np
from tqdm import tqdm

from ES import sepCEM, Control
from models import RLNN
from random_process import GaussianNoise, OrnsteinUhlenbeckProcess
from memory import Memory
from util import *


LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPS = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


USE_CUDA = torch.cuda.is_available()
if  USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor


def evaluate(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False):
    """
    Computes the score of an actor on a given number of runs,
    fills the memory if needed
    """

    if not random:
        def policy(state):
            state = FloatTensor(state.reshape(-1))
            action = actor.select_action(state,_eval=True).cpu().data.numpy().flatten()

            if noise is not None:
                action += noise.sample()

            return np.clip(action, -max_action, max_action)

    else:
        def policy(state):
            return env.action_space.sample()

    scores = []
    steps = 0

    for _ in range(n_episodes):

        score = 0
        obs = deepcopy(env.reset())
        done = False

        while not done:

            # get next action and act
            action = policy(obs)
            n_obs, reward, done, _ = env.step(action)
            done_bool = 0 if steps + \
                1 == env._max_episode_steps else float(done)
            score += reward
            steps += 1

            # adding in memory
            if memory is not None:
                memory.add((obs, n_obs, action, reward, done_bool))
            obs = n_obs

            # render if needed
            if render:
                env.render()

            # reset when done
            if done:
                env.reset()

        scores.append(score)

    return np.mean(scores), steps


class MLP(nn.Module):
    def __init__(self,
                layers,
                activation=torch.tanh,
                output_activation=None,
                output_scale=1,
                output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_scale = output_scale
        self.output_squeeze = output_squeeze

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, inputs):
        x = inputs
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x) * self.output_scale
        else:
            x = self.output_activation(self.layers[-1](x)) * self.output_scale
        return x.squeeze() if self.output_squeeze else x



class GaussianPolicy(RLNN):
    def __init__(self,state_dim, action_dim, max_action, hidden_sizes, activation,
                output_activation, action_space):
        super(GaussianPolicy, self).__init__(state_dim, action_dim, max_action)

        # self.which_one = 0

        action_dim = action_space.shape[0]
        self.action_scale = action_space.high[0]
        self.output_activation = output_activation

        self.net = MLP(
            layers=[state_dim] + list(hidden_sizes),
            activation=activation,
            output_activation=activation)#.to(device)

        self.mu = nn.Linear(
            in_features=list(hidden_sizes)[-1], out_features=action_dim)
        """
        Because this algorithm maximizes trade-off of reward and entropy,
        entropy must be unique to state---and therefore log_stds need
        to be a neural network output instead of a shared-across-states
        learnable parameter vector. But for deep Relu and other nets,
        simply sticking an activationless dense layer at the end would
        be quite bad---at the beginning of training, a randomly initialized
        net could produce extremely large values for the log_stds, which
        would result in some actions being either entirely deterministic
        or too random to come back to earth. Either of these introduces
        numerical instability which could break the algorithm. To
        protect against that, we'll constrain the output range of the
        log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is
        slightly different from the trick used by the original authors of
        SAC---they used torch.clamp instead of squashing and rescaling.
        I prefer this approach because it allows gradient propagation
        through log_std where clipping wouldn't, but I don't know if
        it makes much of a difference.
        """
        self.log_std = nn.Sequential(
            nn.Linear(
                in_features=list(hidden_sizes)[-1], out_features=action_dim),
            nn.Tanh())

    def forward(self, x):
        output = self.net(x)
        mu = self.mu(output)
        if self.output_activation:
            mu = self.output_activation(mu)
        log_std = self.log_std(output)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1)

        policy = Normal(mu, torch.exp(log_std))
        pi = policy.rsample()  # Critical: must be rsample() and not sample()
        logp_pi = torch.sum(policy.log_prob(pi), dim=1)

        mu, pi, logp_pi = self._apply_squashing_func(mu, pi, logp_pi)

        # make sure actions are in correct range
        mu_scaled = mu * self.action_scale
        pi_scaled = pi * self.action_scale

        return pi_scaled, mu_scaled, logp_pi

    def _clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = (x > u).float()
        clip_low = (x < l).float()
        return x + ((u - x) * clip_up + (l - x) * clip_low).detach()

    def _apply_squashing_func(self, mu, pi, logp_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)

        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        logp_pi -= torch.sum(
            torch.log(self._clip_but_pass_gradient(1 - pi**2, l=0, u=1) + EPS),
            dim=1)

        return mu, pi, logp_pi

    def select_action(self,o,_eval=False): # deterministic -- eval
        # if args.evaluate_cpu:
        pi, mu, _ = self.forward(torch.Tensor(o.reshape(1, -1)))
        # else:
        # pi, mu, _ = self.forward(torch.Tensor(o.reshape(1, -1)).to(device))
        return mu.cpu().detach().numpy()[0] if _eval else pi.cpu().detach().numpy()[0]


class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = args.layer_norm

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x):

        if not self.layer_norm:
            print("not come here")
            x = torch.tanh(self.l1(x))
            x = torch.tanh(self.l2(x))
            x = self.max_action * torch.tanh(self.l3(x))

        else:
            print("come here")
            x = torch.tanh(self.n1(self.l1(x)))
            x = torch.tanh(self.n2(self.l2(x)))
            x = self.max_action * torch.tanh(self.l3(x))

        return x

    def update(self, memory, batch_size, critic, actor_t):

        # Sample replay buffer
        states, _, _, _, _ = memory.sample(batch_size)

        # Compute actor loss
        if args.use_td3:
            actor_loss = -critic(states, self(states))[0].mean()
        else:
            actor_loss = -critic(states, self(states)).mean()

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), actor_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class Critic(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x, u):

        if not self.layer_norm:
            x = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x = F.leaky_relu(self.l2(x))
            x = self.l3(x)

        else:
            x = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = F.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Q target = reward + discount * Q(next_state, pi(next_state))
        with torch.no_grad():
            target_Q = critic_t(n_states, actor_t(n_states))
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimate
        current_Q = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class CriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(CriticTD3, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n4 = nn.LayerNorm(400)
            self.n5 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

    def forward(self, x, u):

        if not self.layer_norm:
            print("not come here critic")
            x1 = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x1 = F.leaky_relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            print("come here in critic")
            x1 = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x1 = F.leaky_relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = F.leaky_relu(self.l4(torch.cat([x, u], 1)))
            x2 = F.leaky_relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = F.leaky_relu(self.n4(self.l4(torch.cat([x, u], 1))))
            x2 = F.leaky_relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Select action according to policy and add clipped noise
        noise = np.clip(np.random.normal(0, self.policy_noise, size=(
            batch_size, action_dim)), -self.noise_clip, self.noise_clip)
        n_actions = actor_t(n_states) + FloatTensor(noise)
        n_actions = n_actions.clamp(-max_action, max_action)

        # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
        with torch.no_grad():
            target_Q1, target_Q2 = critic_t(n_states, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
            nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str,)
    parser.add_argument('--env', default='HalfCheetah-v2', type=str)
    parser.add_argument('--start_steps', default=10000, type=int)

    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')

    # TD3 parameters
    parser.add_argument('--use_td3', dest='use_td3', action='store_true')
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_freq', default=2, type=int)

    # Gaussian noise parameters
    parser.add_argument('--gauss_sigma', default=0.1, type=float)

    # OU process parameters
    parser.add_argument('--ou_noise', dest='ou_noise', action='store_true')
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)

    # ES parameters
    parser.add_argument('--pop_size', default=10, type=int)
    parser.add_argument('--elitism', dest="elitism",  action='store_true')
    parser.add_argument('--n_grad', default=5, type=int)
    parser.add_argument('--sigma_init', default=1e-3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp_limit', default=1e-5, type=float)
    parser.add_argument('--mult_noise', dest='mult_noise', action='store_true')

    # Training parameters
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--max_steps', default=1000000, type=int)
    parser.add_argument('--mem_size', default=1000000, type=int)
    parser.add_argument('--n_noisy', default=0, type=int)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--n_test', default=1, type=int)

    # misc
    parser.add_argument('--output', default='results/', type=str)
    parser.add_argument('--period', default=5000, type=int)
    parser.add_argument('--n_eval', default=5, type=int)
    parser.add_argument('--save_all_models',
                        dest="save_all_models", action="store_true")
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    with open(args.output + "/parameters.txt", 'w') as file:
        for key, value in vars(args).items():
            file.write("{} = {}\n".format(key, value))

    # environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])

    # if not os.path.exists("./results_curve"):
    #     os.makedirs("./results_curve")

    file_name_score = "score_%s" % str(args.seed)
    file_name_time = "time_%s" % str(args.seed)


    # memory
    memory = Memory(args.mem_size, state_dim, action_dim)

    # critic
    if args.use_td3:
        critic = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    else:
        critic = Critic(state_dim, action_dim, max_action, args)
        critic_t = Critic(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    # actor
    # actor = Actor(state_dim, action_dim, max_action, args)
    actor = GaussianPolicy(state_dim, action_dim, max_action,(400, 300), torch.relu, None, env.action_space)#.to(device)
    actor_t = GaussianPolicy(state_dim, action_dim, max_action,(400, 300), torch.relu, None, env.action_space)
    actor_t.load_state_dict(actor.state_dict())

    # action noise
    if not args.ou_noise:
        a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)
    else:
        a_noise = OrnsteinUhlenbeckProcess(
            action_dim, mu=args.ou_mu, theta=args.ou_theta, sigma=args.ou_sigma)

    if USE_CUDA:
        critic.cuda()
        critic_t.cuda()
        actor.cuda()
        actor_t.cuda()

    # CEM
    es = sepCEM(actor.get_size(), mu_init=actor.get_params(), sigma_init=args.sigma_init, damp=args.damp, damp_limit=args.damp_limit,
                pop_size=args.pop_size, antithetic=not args.pop_size % 2, parents=args.pop_size // 2, elitism=args.elitism)
    # es = Control(actor.get_size(), pop_size=args.pop_size, mu_init=actor.get_params())

    # training
    step_cpt = 0
    total_steps = 0
    actor_steps = 0
    time_start = time.time()
    df = pd.DataFrame(columns=["total_steps", "average_score",
                               "average_score_rl", "average_score_ea", "best_score"])
    evaluations_score = []
    evaluations_time = []
    while total_steps < args.max_steps:

        fitness = []
        fitness_ = []
        es_params = es.ask(args.pop_size)

        # udpate the rl actors and the critic
        if total_steps > args.start_steps:

            for i in range(args.n_grad):

                # set params
                actor.set_params(es_params[i])
                actor_t.set_params(es_params[i])
                actor.optimizer = torch.optim.Adam(
                    actor.parameters(), lr=args.actor_lr)

                # critic update
                for _ in tqdm(range(actor_steps // args.n_grad)):
                    critic.update(memory, args.batch_size, actor, critic_t)

                # actor update
                for _ in tqdm(range(actor_steps)):
                    actor.update(memory, args.batch_size,
                                 critic, actor_t)

                # get the params back in the population
                es_params[i] = actor.get_params()
        actor_steps = 0

        # evaluate noisy actor(s)
        for i in range(args.n_noisy):
            actor.set_params(es_params[i])
            f, steps = evaluate(actor, env, memory=memory, n_episodes=args.n_episodes,
                                render=args.render, noise=a_noise)
            actor_steps += steps
            prCyan('Noisy actor {} fitness:{}'.format(i, f))

        # evaluate all actors
        for params in es_params:

            actor.set_params(params)
            f, steps = evaluate(actor, env, memory=memory, n_episodes=args.n_episodes,
                                render=args.render)
            actor_steps += steps
            fitness.append(f)

            # print scores
            prLightPurple('Actor fitness:{}'.format(f))

        # update es
        es.tell(es_params, fitness)

        # update step counts
        total_steps += actor_steps
        step_cpt += actor_steps

        # save stuff
        if step_cpt >= args.period:
            # evaluate mean actor over several runs. Memory is not filled
            # and steps are not counted
            actor.set_params(es.mu)
            f_mu, _ = evaluate(actor, env, memory=None, n_episodes=args.n_eval,
                               render=args.render)
            prRed('Actor Mu Average Fitness:{}'.format(f_mu))
            evaluations_score.append(f_mu)
            evaluations_time.append(int(time.time()-time_start))

            # np.save(args.output + "/%s" % file_name_score, evaluations_score)
            # np.save(args.output + "/%s" % file_name_time, evaluations_time)

            df.to_pickle(args.output + "/log.pkl")
            res = {"total_steps": total_steps,
                   "average_score": np.mean(fitness),
                   "average_score_half": np.mean(np.partition(fitness, args.pop_size // 2 - 1)[args.pop_size // 2:]),
                   "average_score_rl": np.mean(fitness[:args.n_grad]),
                   "average_score_ea": np.mean(fitness[args.n_grad:]),
                   "best_score": np.max(fitness),
                   "mu_score": f_mu}

            if args.save_all_models:
                os.makedirs(args.output + "/{}_steps".format(total_steps),
                            exist_ok=True)
                critic.save_model(
                    args.output + "/{}_steps".format(total_steps), "critic")
                actor.set_params(es.mu)
                actor.save_model(
                    args.output + "/{}_steps".format(total_steps), "actor_mu")
            else:
                critic.save_model(args.output, "critic")
                actor.set_params(es.mu)
                actor.save_model(args.output, "actor")
            df = df.append(res, ignore_index=True)
            step_cpt = 0
            print(res)

        print("Total steps", total_steps)
        print("Time", int(time.time()-time_start))

    np.save(args.output + "/%s" % file_name_score, evaluations_score)
    np.save(args.output + "/%s" % file_name_time, evaluations_time)

