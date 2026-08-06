# Policy gradient with a learned value function baseline

import flax.nnx as nnx
import optax

import gymnasium as gym
from dataclasses import dataclass
import simple_parsing
from functools import partial
from typing import Literal

from einops import rearrange
from plain_pg import Policy, calc_returns, collect_trajectories


class Value(nnx.Module):
    def __init__(self, state_dim, hidden_dim, rngs):
        self.net = nnx.Sequential(  # output a value instead of an action
            nnx.Linear(state_dim, hidden_dim, rngs=rngs), nnx.relu, nnx.Linear(hidden_dim, 1, rngs=rngs)
        )

    def __call__(self, state):
        return rearrange(self.net(state), "b 1 -> b")


# MC estimate of the plain policy gradient
def policy_loss_fn(policy, states, actions, logprob_weights, norm_weights):
    logprobs = policy.logprob(states, actions)
    if norm_weights:
        logprob_weights = (logprob_weights - logprob_weights.mean()) / (logprob_weights.std() + 1e-8)
    return -(logprobs * logprob_weights).mean()


# MSE loss to predict returns
def value_loss_fn(value, states, returns):
    preds = value(states)
    return ((preds - returns) ** 2).mean()


@nnx.jit(static_argnames=("cfg",))
def train_step(policy, policy_opt, value, value_opt, states, actions, returns, cfg):

    pred_values = value(states)  # no grad (outside of grad fn)
    advantages = returns - pred_values

    # use advantages instead of raw returns
    policy_loss, policy_grads = nnx.value_and_grad(policy_loss_fn)(
        policy, states, actions, advantages, norm_weights=cfg.norm_weights
    )
    policy_opt.update(policy, policy_grads)

    # update value fn after (not before) policy to avoid more bias
    value_loss, value_grads = nnx.value_and_grad(value_loss_fn)(value, states, returns)
    value_opt.update(value, value_grads)

    return policy_loss, value_loss, optax.global_norm(policy_grads)


@dataclass(frozen=True)
class Config:
    pg_type: Literal["plain", "true_rtg", "common_rtg"] = "plain"
    norm_weights: bool = True
    env: str = "CartPole-v1"
    steps_per_batch: int = 5_000
    disc_factor: float = 0.99
    lr: float = 1e-2
    epochs: int = 50
    seed: int = 0


if __name__ == "__main__":
    cfg = simple_parsing.parse(Config)

    env = gym.make(cfg.env)  # try batched/async ones
    env.reset(seed=cfg.seed)
    state_dim = env.observation_space.shape[0]
    rngs = nnx.Rngs(cfg.seed)

    policy = Policy(state_dim=state_dim, hidden_dim=32, action_dim=env.action_space.n, rngs=rngs)
    value = Value(state_dim=state_dim, hidden_dim=32, rngs=rngs)
    policy_opt = nnx.Optimizer(policy, optax.adam(learning_rate=cfg.lr), wrt=nnx.Param)
    value_opt = nnx.Optimizer(value, optax.adam(learning_rate=cfg.lr), wrt=nnx.Param)

    calc_returns_fn = partial(calc_returns, type=cfg.pg_type, disc_factor=cfg.disc_factor)

    for i in range(cfg.epochs):
        states, actions, logprob_weights, eps_sum_rewards = collect_trajectories(
            env, policy, rngs, steps_per_batch=cfg.steps_per_batch, calc_returns_fn=calc_returns_fn
        )
        policy_loss, value_loss, grad_norm = train_step(
            policy=policy,
            policy_opt=policy_opt,
            value=value,
            value_opt=value_opt,
            states=states,
            actions=actions,
            returns=logprob_weights,
            cfg=cfg,
        )
        print(
            f"{i + 1:>3}/{cfg.epochs} policy loss: {policy_loss:.4f}, value loss: {value_loss:.4f}, grad norm: {grad_norm:.4f}, mean reward: {eps_sum_rewards.mean():.4f}"
        )

    env.close()
