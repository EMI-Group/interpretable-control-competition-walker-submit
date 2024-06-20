import multiprocessing as mp
import numpy as np
import gymnasium as gym
import psutil


def worker_process(remote, envs, batch_policy_func, cpu_idx, max_step):
    p = psutil.Process()
    p.cpu_affinity([cpu_idx])
    env_cnt = len(envs)

    while True:
        cmd, data = remote.recv()
        if cmd == "evaluate":
            seeds, params = data

            # reset envs
            assert len(seeds) == env_cnt
            env_obs_arr = np.zeros(
                [env_cnt] + list(envs[0].observation_space.shape), dtype=np.float32
            )
            env_terminate_arr = np.zeros(env_cnt, dtype=bool)  # false
            env_truncate_arr = np.zeros(env_cnt, dtype=bool)  # false
            env_cum_reward_arr = np.zeros(env_cnt, dtype=np.float32)

            for i in range(env_cnt):
                obs, _ = envs[i].reset(seed=int(seeds[i]))  # don't need info
                env_obs_arr[i] = obs

            # step until done
            for _ in range(max_step):
                done = env_terminate_arr | env_truncate_arr
                if done.all():  # all done
                    break

                # batch step
                actions = batch_policy_func(params, env_obs_arr)

                # update env by actions
                for i in range(env_cnt):
                    if env_terminate_arr[i] | env_truncate_arr[i]:  # done
                        continue  # do not update
                    else:
                        obs, reward, terminated, truncated, _ = envs[i].step(
                            actions[i]
                        )  # don't need info
                        env_obs_arr[i] = obs
                        env_cum_reward_arr[i] += reward
                        env_terminate_arr[i] = terminated
                        env_truncate_arr[i] = truncated

            remote.send(env_cum_reward_arr)

        elif cmd == "close":
            for env in envs:
                env.close()
            remote.close()
            break

        else:
            raise NotImplementedError


class Worker:
    def __init__(self, envs, batch_policy_func, cpu_idx, max_step=1000) -> None:
        self.env_cnt = len(envs)
        self.parent_conn, child_conn = mp.Pipe()
        process = mp.Process(
            target=worker_process,
            args=(child_conn, envs, batch_policy_func, cpu_idx, max_step),
        )
        process.start()

    def query(self, seeds, batch_params):
        self.parent_conn.send(("evaluate", (seeds, batch_params)))

    def get(self):
        res = self.parent_conn.recv()
        return res

    def close(self):
        self.parent_conn.send(("close", None))


class MultiProcessEnv:
    def __init__(
        self,
        worker_num,
        env_num_per_worker,
        env_name,
        repeat_times=1,
        policy_func=None,
        batch_policy_func=None,
        can_jit=True,
        *env_args,
        **env_kwargs
    ):  
        self.worker_num = worker_num
        self.env_num_per_worker = env_num_per_worker
        self.total_envs = worker_num * env_num_per_worker
        self.repeat_times = repeat_times

        if (batch_policy_func is None) and (policy_func is None):
            raise ValueError("policy is required")
        if batch_policy_func is None:
            if can_jit:
                import jax
                self.batch_policy_func = jax.jit(
                    jax.vmap(policy_func)
                )
            else:

                def batch_policy_func(batch_params, batch_obs):
                    outputs = []
                    for i in range(env_num_per_worker):
                        params = [p[i] for p in batch_params]
                        outputs.append(policy_func(params, batch_obs[i]))
                    return outputs

                self.batch_policy_func = batch_policy_func
                
        else:
            self.batch_policy_func = batch_policy_func

        self.workers = []
        for i in range(worker_num):
            envs = [
                gym.make(env_name, *env_args, **env_kwargs)
                for _ in range(env_num_per_worker)
            ]
            cpu_idx = i % psutil.cpu_count()
            self.workers.append(Worker(envs, self.batch_policy_func, cpu_idx))

        self.action_space = envs[0].action_space
        self.observation_space = envs[0].observation_space

    def evaluate(self, seeds, batch_params):
        total_rewards = []
        for _ in range(self.repeat_times):
            rewards = self.evaluate_once(seeds, batch_params)
            total_rewards.append(rewards)
        return np.mean(total_rewards, axis=0)

    def evaluate_once(self, seeds, batch_params):
        try:
            assert len(seeds) == self.total_envs
            assert len(batch_params[0]) == self.total_envs

            split_seeds = np.array_split(seeds, self.worker_num)
            splitted_arrays = [
                tuple(
                    np.array_split(arr, self.worker_num, axis=0)[i]
                    for arr in batch_params
                )
                for i in range(self.worker_num)
            ]

            rewards = []
            for i in range(self.worker_num):
                self.workers[i].query(split_seeds[i], splitted_arrays[i])

            for i in range(self.worker_num):
                rewards.append(self.workers[i].get())

            rewards = np.concatenate(rewards)
            return rewards
        except Exception as e:
            self.close()
            raise e

    def examine(self, params):
        print(f"examining policy for {self.total_envs} times")
        if isinstance(params, list) or isinstance(params, tuple):
            params = [[p] * self.total_envs for p in params]
        else:
            params = [params] * self.total_envs
        seeds = np.random.randint(0, 1000000, self.total_envs)
        return self.evaluate(seeds, params)
        

    def close(self):
        for worker in self.workers:
            worker.close()
