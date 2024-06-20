import warnings

import jax, jax.numpy as jnp

from utils import State, fetch_first, I_INF
from .default import DefaultSpecies


class ElitismPool(DefaultSpecies):
    def __init__(self, pool_size: int, elite_policy: callable = None, *args, **kwargs):
        """
        elite_policy: (state, nodes, conns, mean_fitness, eval_times) -> val
        """

        if "genome_elitism" in kwargs:
            warnings.warn(
                "genome_elitism is not supported in ElitismPool, it will be ignored",
                DeprecationWarning,
            )
        kwargs["genome_elitism"] = 0

        super().__init__(*args, **kwargs)
        self.pool_size = pool_size
        if self.pool_size > self.pop_size:
            raise ValueError("pool_size cannot be greater than pop_size")
        if elite_policy is None:
            elite_policy = (
                lambda state, nodes, conns, mean_fitness, eval_times: mean_fitness
            )  # choose mean fitness
        self.elite_policy = elite_policy

    def setup(self, state=State()):
        state = super().setup(state)
        elite_pool = State(
            pool_nodes=jnp.full((self.pool_size, *state.pop_nodes.shape[1:]), jnp.nan),
            pool_conns=jnp.full((self.pool_size, *state.pop_conns.shape[1:]), jnp.nan),
            pool_vals=jnp.full((self.pool_size,), jnp.nan),
            pool_eval_times=jnp.zeros((self.pool_size,), dtype=jnp.int32),
            pool_mean_fitness=jnp.full((self.pool_size,), jnp.nan),
            pool_hashs=jnp.zeros((self.pool_size,), dtype=jnp.uint32),
        )
        pop_hashs = jax.vmap(self.genome.hash)(state.pop_nodes, state.pop_conns)
        state = state.register(elite_pool=elite_pool, pop_hashs=pop_hashs)
        return state

    def tell(self, state, fitness):
        # update elite pool
        new_pool = pop_update_pool(
            state,
            state.elite_pool,
            state.pop_hashs,
            fitness,
            jnp.ones((self.pop_size,), dtype=jnp.int32),
            self.elite_policy,
        )

        # set new
        new_pool = pop_set_new(
            state,
            new_pool,
            state.pop_nodes,
            state.pop_conns,
            state.pop_hashs,
            fitness,
            jnp.ones((self.pop_size,), dtype=jnp.int32),
            self.elite_policy,
        )
        state = state.update(elite_pool=new_pool)

        # do as default
        k1, k2, randkey = jax.random.split(state.randkey, 3)

        state = state.update(generation=state.generation + 1, randkey=randkey)
        state, winner, loser, elite_mask = self.update_species(state, fitness)

        state, pop_nodes, pop_conns = self.create_next_generation(
            state, winner, loser, elite_mask
        )

        # concate the last self.pool_size with the elite pool
        pop_nodes = jnp.concatenate([pop_nodes, state.elite_pool.pool_nodes], axis=0)
        pop_conns = jnp.concatenate([pop_conns, state.elite_pool.pool_conns], axis=0)

        # re-calculate hash
        pop_hashs = jax.vmap(self.genome.hash)(pop_nodes, pop_conns)
        state = state.update(
            pop_nodes=pop_nodes, pop_conns=pop_conns, pop_hashs=pop_hashs
        )

        # speciate!
        state = self.speciate(state)
        return state

    def cal_spawn_numbers(self, state):
        """
        In next generation, produce self.pop_size - self.pool_size new members
        """

        species_keys = state.species_keys

        is_species_valid = ~jnp.isnan(species_keys)
        valid_species_num = jnp.sum(is_species_valid)
        denominator = (
            (valid_species_num + 1) * valid_species_num / 2
        )  # obtain 3 + 2 + 1 = 6

        rank_score = valid_species_num - self.species_arange  # obtain [3, 2, 1]
        spawn_number_rate = rank_score / denominator  # obtain [0.5, 0.33, 0.17]
        spawn_number_rate = jnp.where(
            is_species_valid, spawn_number_rate, 0
        )  # set invalid species to 0

        # NOTE: the following code is different from the original code
        target_spawn_number = jnp.floor(
            spawn_number_rate * (self.pop_size - self.pool_size)
        )  # calculate member

        # Avoid too much variation of numbers for a species
        previous_size = state.member_count
        spawn_number = (
            previous_size
            + (target_spawn_number - previous_size) * self.spawn_number_change_rate
        )
        spawn_number = spawn_number.astype(jnp.int32)

        # must control the sum of spawn_number to be equal to pop_size
        # NOTE: the following code is different from the original code
        error = (self.pop_size - self.pool_size) - jnp.sum(spawn_number)

        # add error to the first species to control the sum of spawn_number
        spawn_number = spawn_number.at[0].add(error)

        return state, spawn_number

    def create_crossover_pair(self, state, spawn_number, fitness):
        """
        In next generation, produce self.pop_size - self.pool_size new members
        """

        s_idx = self.species_arange
        p_idx = jnp.arange(self.pop_size)

        def aux_func(key, idx):
            members = state.idx2species == state.species_keys[idx]
            members_num = jnp.sum(members)

            members_fitness = jnp.where(members, fitness, -jnp.inf)
            sorted_member_indices = jnp.argsort(members_fitness)[::-1]

            survive_size = jnp.floor(self.survival_threshold * members_num).astype(
                jnp.int32
            )

            select_pro = (p_idx < survive_size) / survive_size
            fa, ma = jax.random.choice(
                key,
                sorted_member_indices,
                shape=(2, self.pop_size - self.pool_size),
                replace=True,
                p=select_pro,
            )

            # NOTE: There is no elite in ElitismPool species
            # elite
            # fa = jnp.where(p_idx < self.genome_elitism, sorted_member_indices, fa)
            # ma = jnp.where(p_idx < self.genome_elitism, sorted_member_indices, ma)
            # elite = jnp.where(p_idx < self.genome_elitism, True, False)
            return fa, ma, jnp.full_like(fa, False, dtype=jnp.bool_)

        randkey_, randkey = jax.random.split(state.randkey)
        fas, mas, elites = jax.vmap(aux_func)(
            jax.random.split(randkey_, self.species_size), s_idx
        )  # fas, mas, elites: (species_size, pop_size - pool_size)

        spawn_number_cum = jnp.cumsum(spawn_number)

        def aux_func(idx):
            loc = jnp.argmax(idx < spawn_number_cum)

            # elite genomes are at the beginning of the species
            idx_in_species = jnp.where(loc > 0, idx - spawn_number_cum[loc - 1], idx)
            return (
                fas[loc, idx_in_species],
                mas[loc, idx_in_species],
                elites[loc, idx_in_species],
            )

        # NOTE: the following code is different from the original code
        part1, part2, elite_mask = jax.vmap(aux_func)(
            jnp.arange(self.pop_size - self.pool_size)
        )

        is_part1_win = fitness[part1] >= fitness[part2]
        winner = jnp.where(is_part1_win, part1, part2)
        loser = jnp.where(is_part1_win, part2, part1)

        return state.update(randkey=randkey), winner, loser, elite_mask

    def create_next_generation(self, state, winner, loser, elite_mask):
        """
        In next generation, produce self.pop_size - self.pool_size new members
        """

        # find next node key
        all_nodes_keys = state.pop_nodes[:, :, 0]
        max_node_key = jnp.max(
            all_nodes_keys, where=~jnp.isnan(all_nodes_keys), initial=0
        )
        next_node_key = max_node_key + 1
        new_node_keys = jnp.arange(self.pop_size - self.pool_size) + next_node_key

        # prepare random keys
        k1, k2, randkey = jax.random.split(state.randkey, 3)
        crossover_randkeys = jax.random.split(k1, self.pop_size - self.pool_size)
        mutate_randkeys = jax.random.split(k2, self.pop_size - self.pool_size)

        wpn, wpc = state.pop_nodes[winner], state.pop_conns[winner]
        lpn, lpc = state.pop_nodes[loser], state.pop_conns[loser]

        # batch crossover
        n_nodes, n_conns = jax.vmap(
            self.genome.execute_crossover, in_axes=(None, 0, 0, 0, 0, 0)
        )(
            state, crossover_randkeys, wpn, wpc, lpn, lpc
        )  # new_nodes, new_conns

        # batch mutation
        m_n_nodes, m_n_conns = jax.vmap(
            self.genome.execute_mutation, in_axes=(None, 0, 0, 0, 0)
        )(
            state, mutate_randkeys, n_nodes, n_conns, new_node_keys
        )  # mutated_new_nodes, mutated_new_conns

        # elitism don't mutate
        pop_nodes = jnp.where(elite_mask[:, None, None], n_nodes, m_n_nodes)
        pop_conns = jnp.where(elite_mask[:, None, None], n_conns, m_n_conns)

        return state.update(randkey=randkey), pop_nodes, pop_conns


def pop_update_pool(
    state, pool, pop_hashs, pop_mean_fitness, pop_eval_times, pool_policy
):
    def process_update(i, p):
        loc = check_in_pool(p, pop_hashs[i])
        return jax.lax.cond(
            loc == I_INF,
            lambda: p,  # not in pool, do nothing now
            lambda: update_pool(
                state, p, loc, pop_mean_fitness[i], pop_eval_times[i], pool_policy
            ),  # update
        )

    return jax.lax.fori_loop(0, pop_hashs.shape[0], process_update, pool)


def pop_set_new(
    state,
    pool,
    pop_nodes,
    pop_conns,
    pop_hashs,
    pop_mean_fitness,
    pop_eval_times,
    pool_policy,
):
    def process_set_new(i, p):
        loc = check_in_pool(p, pop_hashs[i])

        return jax.lax.cond(
            loc == I_INF,
            lambda: add_new(
                state,
                p,
                pop_nodes[i],
                pop_conns[i],
                pop_hashs[i],
                pop_mean_fitness[i],
                pop_eval_times[i],  # eval_times
                pool_policy,
            ),  # add new
            lambda: p,  # don't update this time
        )

    return jax.lax.fori_loop(0, pop_hashs.shape[0], process_set_new, pool)


def add_new(state, pool, nodes, conns, hash, mean_fitness, eval_times, pool_policy):
    new_val = pool_policy(state, nodes, conns, mean_fitness, eval_times)

    def nan_loc():
        return fetch_first(jnp.isnan(pool.pool_mean_fitness))

    def min_loc():
        loc = jnp.argmin(pool.pool_vals)
        return jnp.where(
            pool.pool_vals[loc] >= new_val,
            I_INF,  # not update
            loc,  # update
        )

    select_loc = jax.lax.cond(
        jnp.any(jnp.isnan(pool.pool_vals)),
        nan_loc,  # has nan
        min_loc,  # no nan
    )

    def set_new(loc):
        return pool.update(
            pool_nodes=pool.pool_nodes.at[loc].set(nodes),
            pool_conns=pool.pool_conns.at[loc].set(conns),
            pool_vals=pool.pool_vals.at[loc].set(new_val),
            pool_hashs=pool.pool_hashs.at[loc].set(hash),
            pool_eval_times=pool.pool_eval_times.at[loc].set(eval_times),
            pool_mean_fitness=pool.pool_mean_fitness.at[loc].set(mean_fitness),
        )

    return jax.lax.cond(
        select_loc == I_INF,
        lambda _: pool,  # not update
        set_new,  # update
        operand=select_loc,
    )


def update_pool(state, pool, loc, mean_fitness, eval_times, pool_policy):
    previous_eval_times = pool.pool_eval_times[loc]
    previous_mean = pool.pool_mean_fitness[loc]
    new_eval_times = previous_eval_times + eval_times
    new_mean = (
        previous_mean * previous_eval_times + mean_fitness * eval_times
    ) / new_eval_times
    new_val = pool_policy(
        state, pool.pool_nodes[loc], pool.pool_conns[loc], new_mean, new_eval_times
    )
    return pool.update(
        pool_eval_times=pool.pool_eval_times.at[loc].set(new_eval_times),
        pool_mean_fitness=pool.pool_mean_fitness.at[loc].set(new_mean),
        pool_vals=pool.pool_vals.at[loc].set(new_val),
    )


def check_in_pool(pool, hash):
    return fetch_first(hash == pool.pool_hashs)
