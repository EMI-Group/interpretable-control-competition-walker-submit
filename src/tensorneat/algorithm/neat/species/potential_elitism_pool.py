import warnings

import jax, jax.numpy as jnp

from utils import State, fetch_first, I_INF
from .elitism_pool import *


class PotentialElitismPool(ElitismPool):
    def __init__(
        self,
        elite_size,
        potential_size,
        potential_eval_times,
        potential_policy: callable = None,
        *args,
        **kwargs
    ):
        pool_size = elite_size + potential_size
        kwargs["pool_size"] = pool_size
        super().__init__(*args, **kwargs)

        self.elite_size = elite_size
        self.potential_size = potential_size
        if self.potential_size > self.pool_size:
            warnings.warn(
                "potential_size is larger than pool_size, set potential_size to pool_size"
            )
            self.potential_size = self.pool_size
        self.potential_eval_times = potential_eval_times

        if potential_policy is None:
            potential_policy = (
                lambda state, nodes, conns, mean_fitness, eval_times: mean_fitness
            )  # choose mean fitness
        elite_policy = self.elite_policy

        def clip_potential_policy(state, nodes, conns, mean_fitness, eval_times):
            # will have value when [1, self.potential_eval_times], otherwise, -inf
            return jnp.where(
                eval_times <= self.potential_eval_times,  # must be equal
                potential_policy(state, nodes, conns, mean_fitness, eval_times),
                -jnp.inf,
            )

        def clip_elite_policy(state, nodes, conns, mean_fitness, eval_times):
            # will have value when [self.potential_eval_times, inf), otherwise, -inf
            return jnp.where(
                eval_times >= self.potential_eval_times,  # must be equal
                elite_policy(state, nodes, conns, mean_fitness, eval_times),
                -jnp.inf,
            )

        self.elite_policy = clip_elite_policy
        self.potential_policy = clip_potential_policy

    def setup(self, state=State()):
        state = super().setup(state)
        elite_pool = State(
            pool_nodes=jnp.full((self.elite_size, *state.pop_nodes.shape[1:]), jnp.nan),
            pool_conns=jnp.full((self.elite_size, *state.pop_conns.shape[1:]), jnp.nan),
            pool_vals=jnp.full((self.elite_size,), -jnp.inf),
            pool_eval_times=jnp.zeros((self.elite_size,), dtype=jnp.int32),
            pool_mean_fitness=jnp.full((self.elite_size,), -jnp.inf),
            pool_hashs=jnp.zeros((self.elite_size,), dtype=jnp.uint32),
        )
        state = state.update(elite_pool=elite_pool)
        potential_pool = State(
            pool_nodes=jnp.full(
                (self.potential_size, *state.pop_nodes.shape[1:]), jnp.nan
            ),
            pool_conns=jnp.full(
                (self.potential_size, *state.pop_conns.shape[1:]), jnp.nan
            ),
            pool_vals=jnp.full((self.potential_size,), jnp.nan),
            pool_eval_times=jnp.full((self.potential_size,), self.potential_eval_times),
            pool_mean_fitness=jnp.full((self.potential_size,), jnp.nan),
            pool_hashs=jnp.zeros((self.potential_size,), dtype=jnp.uint32),
        )
        state = state.register(potential_pool=potential_pool)
        return state

    def tell(self, state, fitness):
        # update potential pool by pop
        new_potential_pool = pop_update_pool(
            state,
            state.potential_pool,
            state.pop_hashs,
            fitness,
            jnp.ones((self.pop_size,), dtype=jnp.int32),
            self.potential_policy,
        )

        # update elite pool by pop
        new_elite_pool = pop_update_pool(
            state,
            state.elite_pool,
            state.pop_hashs,
            fitness,
            jnp.ones((self.pop_size,), dtype=jnp.int32),
            self.elite_policy,
        )

        # set new elite pool by potential pool
        new_elite_pool = pop_set_new(
            state,
            new_elite_pool,
            new_potential_pool.pool_nodes,
            new_potential_pool.pool_conns,
            new_potential_pool.pool_hashs,
            new_potential_pool.pool_mean_fitness,
            new_potential_pool.pool_eval_times,
            self.elite_policy,
        )

        # set new potential pool by pop
        new_potential_pool = pop_set_new(
            state,
            new_potential_pool,
            state.pop_nodes,
            state.pop_conns,
            state.pop_hashs,
            fitness,
            jnp.ones((self.pop_size,), dtype=jnp.int32),  # pop_eval_times
            self.potential_policy,
        )

        state = state.update(
            elite_pool=new_elite_pool, potential_pool=new_potential_pool
        )

        # do as default
        k1, k2, randkey = jax.random.split(state.randkey, 3)

        state = state.update(generation=state.generation + 1, randkey=randkey)
        state, winner, loser, elite_mask = self.update_species(state, fitness)

        state, pop_nodes, pop_conns = self.create_next_generation(
            state, winner, loser, elite_mask
        )

        # concate the last self.pool_size with the elite pool
        pop_nodes = jnp.concatenate(
            [pop_nodes, state.potential_pool.pool_nodes, state.elite_pool.pool_nodes],
            axis=0,
        )
        pop_conns = jnp.concatenate(
            [pop_conns, state.potential_pool.pool_conns, state.elite_pool.pool_conns],
            axis=0,
        )

        # re-calculate hash
        pop_hashs = jax.vmap(self.genome.hash)(pop_nodes, pop_conns)
        state = state.update(
            pop_nodes=pop_nodes, pop_conns=pop_conns, pop_hashs=pop_hashs
        )

        # speciate!
        state = self.speciate(state)
        return state
