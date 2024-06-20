import os, json
import jax, jax.numpy as jnp
import time
import numpy as np

from algorithm import BaseAlgorithm
from algorithm.neat.species.elitism_pool import ElitismPool
from utils import State, StatefulBaseClass
from multiprocess_gym import MultiProcessEnv

class Pipeline(StatefulBaseClass):
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        seed: int = 42,
        fitness_target: float = 1,
        generation_limit: int = 1000,
        problem_env: MultiProcessEnv = None,
        save_dir=None,
        is_save: bool = False,
    ):

        self.algorithm = algorithm
        self.seed = seed
        self.max_step=1000
        self.fitness_target = fitness_target
        self.generation_limit = generation_limit
        self.pop_size = self.algorithm.pop_size
        self.problem_env = problem_env

        self.best_genome = None
        self.best_fitness = float("-inf")
        self.generation_timestamp = None

        assert self.problem_env.total_envs == self.pop_size, "Env cnts must be equal to pop size"

        self.batch_pop_transform = jax.jit(jax.vmap(self.algorithm.transform, in_axes=(None, 0)))
        self.algorithm_tell = jax.jit(self.algorithm.tell)

        self.is_save = is_save

        if is_save:
            if save_dir is None:
                now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                self.save_dir = f"./{self.__class__.__name__} {now}"
            else:
                self.save_dir = save_dir
            print(f"save to {self.save_dir}")
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir,)


    def setup(self, state=State()):
        state = state.register(randkey=jax.random.PRNGKey(self.seed))

        state = self.algorithm.setup(state)

        if self.is_save:
            # self.save(state=state, path=os.path.join(self.save_dir, "pipeline.pkl"))
            with open(os.path.join(self.save_dir, "config.txt"), "w") as f:
                f.write(json.dumps(self.show_config(), indent=4))
            # create log file
            with open(os.path.join(self.save_dir, "log.txt"), "w") as f:
                f.write("Generation,Max,Min,Mean,Std,Cost Time\n")

        return state


    def step(self, state):
        randkey = jax.random.split(state.randkey)[0]

        pop = self.algorithm.ask(state)
        pop_transformed = self.batch_pop_transform(state, pop)
        seeds = jax.random.randint(randkey, (self.pop_size, ), 0, 2**31 - 1, dtype=jnp.uint32)

        seeds, pop_transformed = jax.device_get([seeds, pop_transformed])
        fitnesses = self.problem_env.evaluate(seeds, pop_transformed)

        fitnesses = jax.device_put(fitnesses)

        # replace nan with -inf
        fitnesses = jnp.where(jnp.isnan(fitnesses), -jnp.inf, fitnesses)

        previous_pop = self.algorithm.ask(state)
        state = self.algorithm_tell(state, fitnesses)

        return state.update(randkey=randkey), previous_pop, fitnesses
    
    def auto_run(self, state):
        print("Start running pipeline")
        self.data_records = []

        for _ in range(self.generation_limit):
            
            self.generation_timestamp = time.time()
            state, previous_pop, fitnesses = self.step(state)
            fitnesses = jax.device_get(fitnesses)

            self.analysis(state, previous_pop, fitnesses)

            if max(fitnesses) >= self.fitness_target:
                print("Fitness limit reached!")
                return state, self.best_genome

        print("Generation limit reached!")
        return state, self.best_genome, self.data_records

    def analysis(self, state, pop, fitnesses):

        valid_fitnesses = fitnesses[~np.isinf(fitnesses)]

        max_f, min_f, mean_f, std_f = (
            max(valid_fitnesses),
            min(valid_fitnesses),
            np.mean(valid_fitnesses),
            np.std(valid_fitnesses),
        )

        new_timestamp = time.time()

        cost_time = new_timestamp - self.generation_timestamp

        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > self.best_fitness:
            self.best_fitness = fitnesses[max_idx]
            self.best_genome = pop[0][max_idx], pop[1][max_idx]

        # save best if save path is not None

        member_count = jax.device_get(self.algorithm.member_count(state))
        species_sizes = [int(i) for i in member_count if i > 0]

        pop = jax.device_get(pop)
        pop_nodes, pop_conns = pop  # (P, N, NL), (P, C, CL)
        nodes_cnt = (~np.isnan(pop_nodes[:, :, 0])).sum(axis=1)  # (P,)
        conns_cnt = (~np.isnan(pop_conns[:, :, 0])).sum(axis=1)  # (P,)

        max_node_cnt, min_node_cnt, mean_node_cnt = (
            max(nodes_cnt),
            min(nodes_cnt),
            np.mean(nodes_cnt),
        )

        max_conn_cnt, min_conn_cnt, mean_conn_cnt = (
            max(conns_cnt),
            min(conns_cnt),
            np.mean(conns_cnt),
        )

        print(
            f"Generation: {self.algorithm.generation(state)}, Cost time: {cost_time * 1000:.2f}ms\n",
            f"\tnode counts: max: {max_node_cnt}, min: {min_node_cnt}, mean: {mean_node_cnt:.2f}\n",
            f"\tconn counts: max: {max_conn_cnt}, min: {min_conn_cnt}, mean: {mean_conn_cnt:.2f}\n",
            f"\tspecies: {len(species_sizes)}, {species_sizes}\n",
            f"\tfitness: valid cnt: {len(valid_fitnesses)}, max: {max_f:.4f}, min: {min_f:.4f}, mean: {mean_f:.4f}, std: {std_f:.4f}\n",
        )

        # append log
        if self.is_save:
            with open(os.path.join(self.save_dir, "log.txt"), "a") as f:
                f.write(
                    f"{self.algorithm.generation(state)},{max_f},{min_f},{mean_f},{std_f},{cost_time}\n"
                )
        
        if isinstance(self.algorithm.species, ElitismPool):
            self.show_pool(state)

    def show_pool(self, state):
        elite_pool = jax.device_get(state.elite_pool)
        print("\tElite Pool:")
        for i in range(elite_pool.pool_mean_fitness.shape[0]):
            print(
                f"\t\t{i}: mean -> {elite_pool.pool_mean_fitness[i]}, times -> {elite_pool.pool_eval_times[i]}, val -> {elite_pool.pool_vals[i]}, hash -> {elite_pool.pool_hashs[i]}"
            )
        if self.save:
            self.algorithm.species.save(
                state.elite_pool, os.path.join(self.save_dir, "elite_pool.pkl")
            )

        if "potential_pool" in state:
            potential_pool = jax.device_get(state.potential_pool)
            print("\tPotential Pool:")
            for i in range(potential_pool.pool_mean_fitness.shape[0]):
                print(
                    f"\t\t{i}: mean -> {potential_pool.pool_mean_fitness[i]}, times -> {potential_pool.pool_eval_times[i]}, val -> {potential_pool.pool_vals[i]}, hash -> {potential_pool.pool_hashs[i]}"
                )
