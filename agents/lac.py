import copy
from typing import Any

import time

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value, DeterministicActor

from utils.resmlp_value import ResMLPValue


class LACAgent(flax.struct.PyTreeNode):


    rng: Any
    network: Any
    config: Any = nonpytree_field()

    z: Any
    a: Any

    # ------------------------------------------------------------ utilities

    def expectation(self, probs):
        if self.config['num_atoms'] == 1:
            return probs
        return jnp.sum(probs * self.z, axis=-1)
    # ------------------------------------------------------------- losses

    def critic_loss(self, batch, grad_params, rng):
        """Scalar (MSE) TD critic loss. Used when num_atoms == 1."""
        if self.config['action_chunking']:
            batch_actions = jnp.reshape(batch['actions'],
                                        (batch['actions'].shape[0], -1))
        else:
            batch_actions = batch['actions'][..., 0, :]

        rng, sample_rng = jax.random.split(rng)
        next_actions = self._target_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select('target_critic')(
            batch['next_observations'][..., -1, :], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = (batch['rewards'][..., -1]
                    + (self.config['discount'] ** self.config['horizon_length'])
                    * batch['masks'][..., -1] * next_q)

        q = self.network.select('critic')(
            batch['observations'], actions=batch_actions, params=grad_params)

        critic_loss = (jnp.square(q - target_q) * batch['valid'][..., -1]).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def critic_loss_dist(self, batch, grad_params, rng):
        """C51 cross-entropy TD critic loss. Used when num_atoms > 1."""
        if self.config['action_chunking']:
            batch_actions = jnp.reshape(batch['actions'],
                                        (batch['actions'].shape[0], -1))
        else:
            batch_actions = batch['actions'][..., 0, :]

        rng, sample_rng = jax.random.split(rng)
        next_actions = self._target_actions(batch['next_observations'][..., -1, :], rng=sample_rng)

        next_qs = self.network.select('target_critic')(
            batch['next_observations'][..., -1, :], actions=next_actions)

        if self.config['q_agg'] == 'min':
            next_expectation_qs = self.expectation(next_qs)
            min_idx = jnp.argmin(next_expectation_qs, axis=0)
            min_idx = min_idx[None, :, None]
            next_q_dist = jnp.take_along_axis(
            next_qs, min_idx, axis=0).squeeze(axis=0)
        else:
            next_q_dist = jnp.mean(next_qs, axis=0)

        gamma = self.config['discount'] ** self.config['horizon_length']
        target_disc = gamma * batch['masks'][..., -1]

        target_dist = self.categorical_projection(
            next_q_dist, batch['rewards'][..., -1], target_disc)

        q = self.network.select('critic')(
            batch['observations'], actions=batch_actions, params=grad_params)

        eps = 1e-8
        per_ensemble_ce = -jnp.sum(target_dist[None, :, :] * jnp.log(q + eps),
                                   axis=-1)
        critic_loss = (per_ensemble_ce * batch['valid'][..., -1]).mean()

        q_exp = self.expectation(q)

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q_exp.mean(),
            'q_max': q_exp.max(),
            'q_min': q_exp.min(),
        }

    def measure_actor_inference_time(self, observations, rng,
                                     num_warmup=3, num_trials=10):
        """Inference = N candidates × K-step Euler rollout + Q/V eval."""
        for _ in range(num_warmup):
            rng, key = jax.random.split(rng)
            a = self.sample_actions(observations, rng=key)
            jax.block_until_ready(a)
        t0 = time.perf_counter()
        for _ in range(num_trials):
            rng, key = jax.random.split(rng)
            a = self.sample_actions(observations, rng=key)
            jax.block_until_ready(a)
        return (time.perf_counter() - t0) / num_trials

    def categorical_projection(self, next_dist, rewards, discounts):
        B = rewards.shape[0]
        num_atoms = self.config['num_atoms']
        v_min = self.config['v_min']
        v_max = self.config['v_max']
        z = jnp.linspace(v_min, v_max, num_atoms)
        delta_z = (v_max - v_min) / (num_atoms - 1)

        Tz = rewards[:, None] + discounts[:, None] * z[None, :]
        Tz = jnp.clip(Tz, v_min, v_max)

        b = (Tz - v_min) / delta_z
        l = jnp.clip(jnp.floor(b).astype(jnp.int32), 0, num_atoms - 1)
        u = jnp.clip(jnp.ceil(b).astype(jnp.int32), 0, num_atoms - 1)

        m = jnp.zeros((B, num_atoms))
        batch_idx = jnp.broadcast_to(jnp.arange(B)[:, None], l.shape)

        upper_weight = b - l
        lower_weight = u - b
        same = (l == u).astype(jnp.float32)

        m = m.at[(batch_idx, l)].add(next_dist * (lower_weight + same))
        m = m.at[(batch_idx, u)].add(next_dist * upper_weight)
        m = m / (m.sum(axis=-1, keepdims=True) + 1e-12)
        return m

    # ------------------------------------------------------------ actor (DPG)
    def actor_loss(self, batch, grad_params, rng):
        """DPG actor loss: -E[Q(s, mu(s))] + lmbda * BC(mu, a_data)."""
        if self.config['action_chunking']:
            batch_actions = jnp.reshape(batch['actions'],
                                        (batch['actions'].shape[0], -1))
        else:
            batch_actions = batch['actions'][..., 0, :]

        mu = self.network.select('actor')(
            batch['observations'], params=grad_params)
        mu = jnp.clip(mu, -1.0, 1.0)

        qs = self.network.select('critic')(batch['observations'], actions=mu)
        qs = self.expectation(qs)
        q = jnp.mean(qs, axis=0)
        q_loss = -q.mean()

        bc_loss = jnp.mean((mu - batch_actions) ** 2)

        actor_loss = self.config['lmbda'] * bc_loss + q_loss

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_loss': bc_loss,
            'q_loss': q_loss,
        }


    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        if self.config['num_atoms'] == 1:
            critic_loss, critic_info = self.critic_loss(
                batch, grad_params, critic_rng)
        else:
            critic_loss, critic_info = self.critic_loss_dist(
                batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info



    # ----------------------------------------------------------- bookkeeping

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')
        agent.target_update(new_network, 'actor')  
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)


    # ------------------------------------------------------------ inference

    @jax.jit
    def sample_actions(self, observations, rng=None):
        actions = self.network.select('actor')(observations)
        return jnp.clip(actions, -1.0, 1.0)

    def _target_actions(self, observations, rng):
        mu = self.network.select('target_actor')(observations)
        return jnp.clip(mu, -1.0, 1.0)

    # -------------------------------------------------------------- factory

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]
        if config['action_chunking']:
            full_actions = jnp.concatenate(
                [ex_actions] * config['horizon_length'], axis=-1)
        else:
            full_actions = ex_actions
        full_action_dim = full_actions.shape[-1]

        z = jnp.linspace(config['v_min'], config['v_max'], config['num_atoms'])
        a = jnp.linspace(config['v_min'], config['v_max'], config['num_atoms'])

        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()
            encoders['target_actor'] = encoder_module()  

        # ---- critic: ResMLP ------------------------------------------
        critic_def = ResMLPValue(
            num_ensembles=config['num_qs'],
            encoder=encoders.get('critic'),
            num_atoms=config['num_atoms'],
            embed_dim=config['critic_embed_dim'],
            num_layers=config['critic_num_layers'],
            sub_layers=config['critic_sub_layers'],
        )




        actor_def = DeterministicActor(
            width=config['actor_width'],
            depth=config['actor_depth'],
            action_dim=full_action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
        )




        network_info = dict(
                actor=(actor_def, (ex_observations,)),
                target_actor=(copy.deepcopy(actor_def),     
                            (ex_observations,)),
                critic=(critic_def, (ex_observations, full_actions)),
                target_critic=(copy.deepcopy(critic_def),
                            (ex_observations, full_actions)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config['weight_decay'] > 0.0:
            network_tx = optax.adamw(learning_rate=config['lr'],
                                     weight_decay=config['weight_decay'])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor'] = (params['modules_actor'])

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network,
                   config=flax.core.FrozenDict(**config), z=z, a=a)


def get_config():
    config = ml_collections.ConfigDict(dict(
        agent_name='lac',
        ob_dims=ml_collections.config_dict.placeholder(list),
        action_dim=ml_collections.config_dict.placeholder(int),

        # Optimisation.
        lr=3e-4,
        batch_size=256,
        weight_decay=0.0,

        # Actor (kept light).
        actor_width=512,
        actor_depth=4,

        actor_layer_norm=False,

        # ResMLP critic capacity knobs.
        # `critic_num_layers` is the *total* number of dense layers; with
        # `critic_sub_layers=4` (paper default), num_blocks = N/4.
        # Suggested sweep: [4, 16, 64, 256, 1024].
        critic_embed_dim=256,
        critic_num_layers=4,        # = 16 residual blocks
        critic_sub_layers=4,          # paper default


        # -----------------------------------------------------------------

        # Distributional / ensembling.
        num_atoms=51,
        v_min=-1000.0,
        v_max=0.0,
        num_qs=1,
        q_agg='mean',

        # Discount / target / horizon.
        discount=0.999,
        tau=0.005,
        horizon_length=ml_collections.config_dict.placeholder(int),
        action_chunking=False,

        # Pure DPG by default; set lmbda > 0 to add a BC regulariser.
        lmbda=20.0,

        # Compat with main.py.

        encoder=ml_collections.config_dict.placeholder(str),
    ))
    return config