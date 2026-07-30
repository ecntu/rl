import numpy as np
import jax.numpy as jnp
import flax.nnx as nnx
import optax

import gymnasium as gym
from dataclasses import dataclass
import simple_parsing


class Policy(nnx.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, rngs):
        self.net = nnx.Sequential(
            nnx.Linear(state_dim, hidden_dim, rngs=rngs), nnx.relu, nnx.Linear(hidden_dim, action_dim, rngs=rngs)
        )

    def __call__(self, state):
        return self.net(state)

    @nnx.jit
    def sample_action(self, state, rngs):
        logits = self(state)
        return rngs.categorical(logits)

    def logprob(self, state, action):
        (bs,) = action.shape
        logits = self(state)
        logprobs = nnx.log_softmax(logits)
        return logprobs[jnp.arange(bs), action]


def collect_trajectories(env, policy, rngs, steps_per_batch, disc_factor):

    # note: we can avoid having an `episode` dim/axis since the loss doesn't really care about it

    states, actions, returns = [], [], []  # returns are used for weighting

    eps_sum_rewards = []  # for logging
    curr_eps_rewards = []
    state, _ = env.reset()

    while len(states) < steps_per_batch:
        action = policy.sample_action(state, rngs)
        action = action.item()

        new_state, reward, terminated, truncated, _ = env.step(action)

        curr_eps_rewards.append(reward)
        states.append(state)
        actions.append(action)
        state = new_state

        if terminated or truncated or len(states) >= steps_per_batch:  # end of episode
            discounted_return = np.array(curr_eps_rewards) @ (disc_factor ** np.arange(len(curr_eps_rewards)))
            returns.extend([discounted_return] * len(curr_eps_rewards))

            if terminated or truncated:  # to avoid deflating with last truncated episode
                eps_sum_rewards.append(sum(curr_eps_rewards))

            curr_eps_rewards = []
            state, _ = env.reset()

    return jnp.array(states), jnp.array(actions), jnp.array(returns), jnp.array(eps_sum_rewards)


# MC estimate of the plain policy gradient
def loss_fn(policy, states, actions, returns):
    logprobs = policy.logprob(states, actions)
    return -(logprobs * returns).mean()


@dataclass(frozen=True)
class Config:
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
    opt = nnx.Optimizer(policy, optax.adam(learning_rate=cfg.lr), wrt=nnx.Param)

    @nnx.jit
    def train_step(policy, opt, states, actions, returns):
        loss, grads = nnx.value_and_grad(loss_fn)(policy, states, actions, returns)
        opt.update(policy, grads)
        return loss, optax.global_norm(grads)

    for i in range(cfg.epochs):
        states, actions, returns, eps_sum_rewards = collect_trajectories(
            env, policy, rngs, steps_per_batch=cfg.steps_per_batch, disc_factor=cfg.disc_factor
        )
        loss, grad_norm = train_step(policy, opt, states, actions, returns)
        print(
            f"{i + 1:>3}/{cfg.epochs} loss: {loss:.4f}, grad norm: {grad_norm:.4f}, mean reward: {eps_sum_rewards.mean():.4f}"
        )

    env.close()
