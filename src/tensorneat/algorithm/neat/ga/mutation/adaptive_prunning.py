import jax, jax.numpy as jnp
from . import DefaultMutation
from utils import (
    fetch_first,
    fetch_random,
    I_INF,
    unflatten_conns,
    check_cycles,
    add_node,
    add_conn,
    delete_node_by_pos,
    delete_conn_by_pos,
    extract_node_attrs,
    extract_conn_attrs,
    set_node_attrs,
    set_conn_attrs,
)


class AdaptivePrunning(DefaultMutation):
    def __init__(
        self,
        delete_prob_func: callable,
    ):
        super().__init__(conn_add=0, conn_delete=0, node_add=0, node_delete=0)
        self.delete_prob_func = delete_prob_func


    def mutate_structure(self, state, randkey, genome, nodes, conns, new_node_key):
        
        def mutate_delete_conn(key_, nodes_, conns_):
            # randomly choose a connection
            i_key, o_key, idx = self.choose_connection_key(key_, conns_)

            return jax.lax.cond(
                idx == I_INF,
                lambda: (nodes_, conns_),  # nothing
                lambda: (nodes_, delete_conn_by_pos(conns_, idx)),  # success
            )

        conn_cnt = (~jnp.isnan(conns[:, 0])).sum()
        remove_conn_prob = self.delete_prob_func(conn_cnt)

        k1, k2 = jax.random.split(randkey)
        r = jax.random.uniform(k1)

        nodes, conns = jax.lax.cond(
            r < remove_conn_prob, 
            mutate_delete_conn, 
            lambda _, nodes_, conns_: (nodes_, conns_),
            k2, 
            nodes, 
            conns
        )

        return nodes, conns
