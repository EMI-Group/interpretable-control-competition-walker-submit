import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
sys.path.append("./tensorneat")

import numpy as np
np.random.seed(0)

import jax.numpy as jnp
from multiprocess_pipeline import Pipeline
from algorithm.neat import *
from algorithm.neat.species.elitism_pool import ElitismPool
from algorithm.neat.species.potential_elitism_pool import PotentialElitismPool
from utils import Act, Agg
from multiprocess_gym import MultiProcessEnv
import pandas as pd
import pickle


means = np.array([1.1976822536247531, -0.35102943685124777, -0.2203174498164294, -0.2986962042760092, 0.2610408402156953, -0.21853167127519232, -0.2993755938632375, 0.25157172310134374, -0.8478058736363461, -0.9333181086992711, -4.7226151368863425, -3.04464075826656, -3.0541478390809043, 1.3292168773883335, -3.0206384578424763, -3.061709520277573, 1.2834942168533578])

stds = np.array([0.059991334260223274, 0.3027864946480575, 0.27794117981869126, 0.41703392358360514, 0.3935586236618236, 0.27728799132318876, 0.417414699726967, 0.3981754441811435, 0.5626749085240695, 0.9586579625717025, 4.632738896759641, 5.894634543450535, 5.188063813027244, 7.24995626995127, 5.9084191344240296, 5.206415994377771, 7.256801163226521])

C = 0.1
CONN_WEIGHT = 500
TARGET_FITNESS = 3200

def normalize_obs(obs):
    return (obs - means) / (stds + 1e-6)


def create_cosine_decline(a, v_a, b, v_b):
    # ensure that a > b, for this function is to find decline coscine curve
    # v_x = Acos(Bx+C)+D
    
    if a < b:
        # swap a b
        a, b = b, a
        v_a, v_b = v_b, v_a
    
    A = (v_a - v_b) / 2
    B = jnp.pi / (b - a)
    C = -B * a
    D = (v_a + v_b) / 2

    def cosine_func(x):
        x = jnp.clip(x, b, a)
        return A * jnp.cos(B * x + C) + D

    return cosine_func


def conn_score(state, conn_cnt, mean_val, eval_times):
    return jnp.where(
        mean_val > TARGET_FITNESS,
        CONN_WEIGHT * (112 - conn_cnt),
        0
    )

def norm_conn_elite_policy(state, nodes, conns, mean_val, eval_times):
    conn_cnt = jnp.sum(~jnp.isnan(conns[:, 0]))
    return mean_val + conn_score(state, conn_cnt, mean_val, eval_times)


def norm_conn_potential_policy(state, nodes, conns, mean_val, eval_times):
    conn_cnt = jnp.sum(~jnp.isnan(conns[:, 0]))
    return mean_val / (1 + C / eval_times) + conn_score(state, conn_cnt, mean_val, eval_times)

if __name__ == "__main__":
    genome=DenseInitialize(
        num_inputs=17,
        num_outputs=6,
        max_nodes=17+6,
        max_conns=17*6,
        node_gene=NodeGeneWithoutResponse(
            bias_mutate_rate=0.2,
            bias_init_std=1,
            bias_init_mean=0,
            bias_mutate_power=0.15,
            bias_replace_rate=0.015,
        ),
        conn_gene=DefaultConnGene(
            weight_mutate_rate=0.2,
            weight_replace_rate=0.015,
            weight_mutate_power=0.15,  
            weight_init_std=1,
            weight_init_mean=0,
        ),
        output_transform=Act.standard_tanh,
        input_transform=normalize_obs,
        mutation=AdaptivePrunning(
            delete_prob_func=create_cosine_decline(a=20, v_a=0.1, b=102, v_b=0.8)
        )
    )

    problem_env = MultiProcessEnv(
        env_num_per_worker=2,
        worker_num=250,
        env_name="Walker2d-v4",
        policy_func=lambda params, obs: genome.forward(
            None, params, obs
        ),
        repeat_times=1,
    )

    pipeline = Pipeline(
        seed=0,
        algorithm=NEAT(
            species=PotentialElitismPool(
                elite_size=5,
                potential_size=10,
                elite_policy=norm_conn_elite_policy,
                potential_policy=norm_conn_potential_policy,
                potential_eval_times=5,
                genome=genome,
                pop_size=500,
                species_size=10,
                max_stagnation=7,
                genome_elitism=3,
                compatibility_threshold=1.0,
                survival_threshold=0.1,
            ),
        ),
        problem_env=problem_env,
        generation_limit=400,
        fitness_target=50000,
        is_save=True,
        save_dir="../results"
    )

    state = pipeline.setup()
    state, best, data_record = pipeline.auto_run(state)
    problem_env.close()
