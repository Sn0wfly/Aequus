import jax
import jax.numpy as jnp
import pyspiel
from open_spiel.python.algorithms import cfr_jax

def cfr_step_gpu(batch_states, regrets_prev, strategy_prev, lr=0.1):
    """Un solo paso de CFR-JAX sobre GPU.
    batch_states: (B, state_size) tensor JAX
    regrets_prev: (B, num_actions) tensor JAX
    strategy_prev: (B, num_actions) tensor JAX
    retorna (new_regrets, new_strategy) ambos en GPU
    """
    return cfr_jax.step(batch_states,
                        regrets_prev,
                        strategy_prev,
                        learning_rate=lr) 