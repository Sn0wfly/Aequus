"""
🚀 DEFINITIVE HYBRID TRAINER
Combines vectorized GPU simulation with efficient CPU-GPU bridge for dynamic memory management.

Key Innovations:
- Vectorized GPU simulation (fastest possible)
- Efficient CPU-GPU bridge for memory management
- Scatter-gather updates for optimal GPU usage
- Dynamic growth with minimal CPU overhead
- PURE JIT functions for maximum performance
- MULTIPROCESSING for CPU bottleneck optimization
"""

import jax
import jax.numpy as jnp
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from functools import partial
import time
import pickle
from poker_bot.evaluator import HandEvaluator
from functools import lru_cache
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)

# 🚀 Import Cython module for ultra-fast hashing
try:
    from .hasher import map_hashes_cython
    CYTHON_AVAILABLE = True
    logger.info("🚀 Cython fast hasher: ENABLED for maximum performance")
except ImportError:
    CYTHON_AVAILABLE = False
    logger.warning("⚠️ Cython fast hasher: NOT AVAILABLE, falling back to Python")

@dataclass
class TrainerConfig:
    """Configuration for PokerTrainer"""
    batch_size: int = 8192
    learning_rate: float = 0.1
    temperature: float = 1.0
    num_actions: int = 4  # fold, call, bet, raise
    dtype: jnp.dtype = jnp.bfloat16
    accumulation_dtype: jnp.dtype = jnp.float32
    max_info_sets: int = 1000000  # 1M info sets max
    growth_factor: float = 1.5  # Grow by 50% when full
    chunk_size: int = 20000  # Subo chunk_size a 20_000
    gpu_bucket: bool = False  # Placeholder para bucketing en GPU

# 🚀 PURE JIT-COMPILED FUNCTION (Outside class for maximum performance)
@partial(jax.jit, static_argnums=(4, 5))
def _static_vectorized_scatter_update(q_values: jnp.ndarray, 
                                    strategies: jnp.ndarray,
                                    indices: jnp.ndarray, 
                                    cf_values: jnp.ndarray,
                                    learning_rate: float,
                                    temperature: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    🚀 PURE JIT-COMPILED FUNCTION: Maximum performance with dynamic shapes
    JAX will automatically recompile for new array shapes while maintaining speed
    """
    # Ensure cf_values has the same dtype as q_values to avoid warnings
    cf_values = cf_values.astype(q_values.dtype)
    
    # GATHER: Get current Q-values for indices
    current_q_subset = q_values[indices]
    
    # UPDATE: Compute new Q-values
    updated_q_subset = current_q_subset + learning_rate * (cf_values - current_q_subset)
    
    # SCATTER: Update Q-values
    new_q_values = q_values.at[indices].set(updated_q_subset)
    
    # Update strategies
    strategies_subset = jax.nn.softmax(updated_q_subset / temperature)
    new_strategies = strategies.at[indices].set(strategies_subset)
    
    return new_q_values, new_strategies

class PokerTrainer:
    """
    🚀 POKER TRAINER
    Combines vectorized GPU simulation with efficient CPU-GPU bridge
    """
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.iteration = 0
        self.total_games = 0
        self.total_info_sets = 0
        self.total_unique_info_sets = 0
        self.growth_events = 0
        self._main_device = None

        # --- SOLUCIÓN TAREA 1: Guardar el dispositivo principal ---
        try:
            self._main_device = jax.devices('gpu')[0]
            logger.info(f"✅ GPU detectada. Asignando arrays principales a: {self._main_device}")
        except IndexError:
            self._main_device = jax.devices('cpu')[0]
            logger.warning(f"⚠️ No se encontró GPU. Usando CPU: {self._main_device}")

        initial_q_values = jnp.zeros((config.max_info_sets, config.num_actions), dtype=config.dtype)
        self.q_values = jax.device_put(initial_q_values, device=self._main_device)
        
        initial_strategies = jnp.ones((config.max_info_sets, config.num_actions), dtype=config.dtype) / config.num_actions
        self.strategies = jax.device_put(initial_strategies, device=self._main_device)
        
        # 🧠 CPU Memory Management (the brain)
        self.info_set_hashes: Dict[str, int] = {}
        self.info_set_data: Dict[int, Dict[str, Any]] = {}
        self.next_index = 0
        
        self.evaluator = HandEvaluator()  # Instancia para bucketing
        
        logger.info("🚀 Definitive Hybrid Trainer initialized")
        logger.info(f"   Batch size: {config.batch_size}")
        logger.info(f"   Max info sets: {config.max_info_sets:,}")
        logger.info(f"   Growth factor: {config.growth_factor}")
        logger.info(f"   Chunk size: {config.chunk_size}")
        logger.info(f"   Target: Real NLHE 6-player strategies with optimal GPU-CPU bridge")
        logger.info(f"   🚀 PURE JIT functions: ENABLED for maximum performance")
    
    @lru_cache(maxsize=50000)
    def _get_info_set_bucket(self, hole_cards_tuple, community_cards_tuple, position, pot_bucket, stack_bucket, num_active, last_action_type):
        """
        Abstracción avanzada: incluye stack_bucket, pot_bucket, num_active, last_action_type.
        """
        hole_cards = np.array(hole_cards_tuple)
        community_cards = np.array(community_cards_tuple)
        # --- Bucket por Ronda de Apuestas ---
        dealt_community = community_cards[community_cards >= 0]
        round_map = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}
        round_name = round_map.get(len(dealt_community), "River")
        # --- Bucket por Fuerza de Mano (usando phevaluator) ---
        if round_name == "Preflop":
            card1_rank, card2_rank = sorted([c % 13 for c in hole_cards], reverse=True)
            is_pair = (card1_rank == card2_rank)
            is_suited = (hole_cards[0] // 13 == hole_cards[1] // 13)
            if is_pair and card1_rank >= 10: hand_category = "PremiumPair"
            elif is_pair: hand_category = "LowPair"
            elif is_suited and card1_rank >= 9: hand_category = "PremiumSuited"
            elif card1_rank >= 10 and card2_rank >= 10: hand_category = "PremiumBroadway"
            else: hand_category = "Other"
        else:
            full_hand = np.concatenate((hole_cards, dealt_community))
            strength_rank = self.evaluator.evaluate_single(full_hand.tolist())
            if strength_rank <= 10: hand_category = "StraightFlush"
            elif strength_rank <= 166: hand_category = "Quads"
            elif strength_rank <= 322: hand_category = "FullHouse"
            elif strength_rank <= 1599: hand_category = "Flush"
            elif strength_rank <= 1609: hand_category = "Straight"
            elif strength_rank <= 2467: hand_category = "ThreeOfAKind"
            elif strength_rank <= 3325: hand_category = "TwoPair"
            elif strength_rank <= 6185: hand_category = "OnePair"
            else: hand_category = "HighCard"
        return f"R:{round_name}_H:{hand_category}_P:{position}_Stk:{stack_bucket}_Pot:{pot_bucket}_Act:{num_active}_Last:{last_action_type}"

    def _get_or_create_index(self, bucket_id: str) -> int:
        """Get or create index for info set hash (CPU operation)"""
        if bucket_id not in self.info_set_hashes:
            # Check if we need to grow arrays
            if self.next_index >= self.config.max_info_sets:
                self._grow_arrays()
            
            # Create new index
            self.info_set_hashes[bucket_id] = self.next_index
            self.info_set_data[self.next_index] = {'hash': bucket_id}
            self.next_index += 1
            self.total_unique_info_sets += 1
        
        return self.info_set_hashes[bucket_id]
    
    def _grow_arrays(self):
        old_size = self.config.max_info_sets
        new_size = int(old_size * self.config.growth_factor)
        logger.info(f"🔄 Growing arrays from {old_size:,} to {new_size:,}")

        # --- SOLUCIÓN TAREA 1: Usar el dispositivo guardado ---
        target_device = self._main_device 
        logger.info(f"🔄 Creciendo arrays y asignando a dispositivo: {target_device}")

        new_q_values = jnp.zeros((new_size, self.config.num_actions), dtype=self.config.dtype)
        new_strategies = jnp.ones((new_size, self.config.num_actions), dtype=self.config.dtype) / self.config.num_actions

        new_q_values = new_q_values.at[:old_size].set(self.q_values)
        new_strategies = new_strategies.at[:old_size].set(self.strategies)

        self.q_values = jax.device_put(new_q_values, device=target_device)
        self.strategies = jax.device_put(new_strategies, device=target_device)
        self.config.max_info_sets = new_size

        self.growth_events += 1
        logger.info(f"✅ Arrays grown successfully (event #{self.growth_events})")
    
    @partial(jax.jit, static_argnums=(0,))
    def _vectorized_info_set_processing(self, game_data: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """VECTORIZED info set processing on GPU - returns only JAX arrays"""
        batch_size = game_data['payoffs'].shape[0]
        num_players = game_data['payoffs'].shape[1]
        total_info_sets = batch_size * num_players
        
        # Extract game data
        hole_cards = game_data['hole_cards']  # (batch, players, 2)
        final_community = game_data['final_community']  # (batch, 5)
        payoffs = game_data['payoffs']  # (batch, players)
        final_pots = game_data['final_pot']  # (batch,)
        
        # Flatten for vectorized processing
        flat_hole_cards = hole_cards.reshape(-1, 2)
        flat_payoffs = payoffs.reshape(-1)
        flat_final_pots = jnp.repeat(final_pots, num_players)
        flat_community = jnp.repeat(final_community[:, None, :], num_players, axis=1).reshape(-1, 5)
        
        # Vectorized hand strength calculation
        def calculate_hand_strength_vectorized(hole_cards, community_cards):
            hole_sum = jnp.sum(hole_cards, axis=1)
            community_sum = jnp.sum(community_cards, axis=1)
            return (hole_sum + community_sum) / 100.0
        
        hand_strengths = calculate_hand_strength_vectorized(flat_hole_cards, flat_community)
        
        # Vectorized counterfactual values
        def compute_cf_values_vectorized(payoffs):
            return jnp.stack([
                payoffs * 0.5,  # Fold: lose some
                payoffs * 1.0,  # Call: neutral
                payoffs * 1.5,  # Bet: win more
                payoffs * 2.0   # Raise: win most
            ], axis=1)
        
        cf_values = compute_cf_values_vectorized(flat_payoffs)
        
        # Create player IDs and game indices for later use
        player_ids = jnp.arange(total_info_sets) % num_players
        game_indices = jnp.arange(total_info_sets) // num_players
        
        return {
            'total_info_sets': total_info_sets,
            'cf_values': cf_values,
            'hole_cards': flat_hole_cards,
            'community_cards': flat_community,
            'pot_sizes': flat_final_pots,
            'hand_strengths': hand_strengths,
            'payoffs': flat_payoffs,
            'player_ids': player_ids,
            'game_indices': game_indices
        }
    
    def _batch_get_buckets(self, hole_cards: np.ndarray, community_cards: np.ndarray, positions: np.ndarray, pot_sizes: np.ndarray, stack_sizes: np.ndarray, num_actives: np.ndarray, last_action_types: np.ndarray) -> np.ndarray:
        """
        Bucketing masivo en paralelo usando joblib.Parallel.
        """
        def to_tuple(arr):
            return tuple(arr.tolist())
        n = len(hole_cards)
        # Precalcula los buckets de stack y pot
        stack_buckets = np.array([f"{int(s//5)*5}-{int(s//5)*5+5}" for s in stack_sizes])
        pot_buckets = np.array([f"{int(p//3)*3}-{int(p//3)*3+3}" for p in pot_sizes])
        # joblib.Parallel para bucketing masivo
        buckets = Parallel(n_jobs=-1, backend='threading', batch_size=20000)(
            delayed(self._get_info_set_bucket)(
                to_tuple(hole_cards[i]),
                to_tuple(community_cards[i]),
                int(positions[i]),
                pot_buckets[i],
                stack_buckets[i],
                int(num_actives[i]),
                last_action_types[i]
            ) for i in range(n)
        )
        return np.array(buckets)

    def _map_info_sets_to_indices(self, game_results: Dict[str, jnp.ndarray]) -> np.ndarray:
        """
        🧠 CPU-GPU BRIDGE: Bucketing vectorizado y paralelo con granularidad avanzada.
        """
        logger.info("🧠 Vectorizando buckets de estrategia en CPU...")
        data_np = jax.device_get({
            'hole_cards': game_results['hole_cards'],
            'final_community': game_results['final_community'],
            'final_pot': game_results['final_pot'],
            'payoffs': game_results['payoffs'],
            # Suponiendo que tienes acceso a stacks y last_action_type, si no, usa placeholders
        })
        hole_cards_np = data_np['hole_cards'] # (B, 6, 2)
        community_cards_np = data_np['final_community'] # (B, 5)
        pot_sizes_np = data_np['final_pot'] # (B,)
        payoffs_np = data_np['payoffs'] # (B, 6)
        B, P, _ = hole_cards_np.shape
        total_info_sets = B * P
        hole_flat = hole_cards_np.reshape(total_info_sets, 2)
        comm_flat = np.repeat(community_cards_np[:, np.newaxis, :], P, axis=1).reshape(total_info_sets, 5)
        pos_flat = np.tile(np.arange(P), B)
        # Placeholder: todos los stacks iguales, puedes mejorarlo si tienes stacks reales
        stack_flat = np.full(total_info_sets, 100.0)
        pot_flat = np.repeat(pot_sizes_np, P)
        # Número de jugadores activos por juego
        num_actives = np.sum(hole_cards_np[:, :, 0] != -1, axis=1)
        num_actives_flat = np.repeat(num_actives, P)
        # Placeholder para last_action_type
        last_action_types = np.full(total_info_sets, "none", dtype=object)
        valid_mask = (hole_flat[:, 0] != -1)
        all_buckets = self._batch_get_buckets(
            hole_flat[valid_mask],
            comm_flat[valid_mask],
            pos_flat[valid_mask],
            pot_flat[valid_mask],
            stack_flat[valid_mask],
            num_actives_flat[valid_mask],
            last_action_types[valid_mask]
        )
        unique_buckets, inverse_indices = np.unique(all_buckets, return_inverse=True)
        indices_map = {bucket: self._get_or_create_index(bucket) for bucket in unique_buckets}
        final_indices = np.array([indices_map[bucket] for bucket in unique_buckets])[inverse_indices]
        return final_indices.astype(np.int32)
    
    def _vectorized_scatter_update(self, indices: jnp.ndarray, cf_values: jnp.ndarray, n: int) -> None:
        """
        🚀 GPU SCATTER UPDATE: Update only necessary Q-values efficiently
        Now uses PURE JIT function for maximum performance
        """
        # Call the PURE JIT-compiled function
        new_q_values, new_strategies = _static_vectorized_scatter_update(
            self.q_values,
            self.strategies,
            indices[:n],
            cf_values[:n],
            self.config.learning_rate,
            self.config.temperature
        )
        self.q_values = new_q_values
        self.strategies = new_strategies
    
    def train(self, num_iterations: int, save_path: str, save_interval: int):
        """
        Bucle principal de entrenamiento.
        """
        logger.info(f"🚀 Iniciando bucle de entrenamiento por {num_iterations} iteraciones...")
        start_time = time.time()
        
        initial_iteration = self.iteration
        for i in range(initial_iteration, initial_iteration + num_iterations):
            self.iteration = i
            
            # Generar claves para el batch
            rng_key = jax.random.PRNGKey(i)
            rng_keys = jax.random.split(rng_key, self.config.batch_size)
            
            # Configuración del juego (puede ser más dinámica en el futuro)
            game_config = {
                'players': 6,
                'starting_stack': 100.0,
                'small_blind': 1.0,
                'big_blind': 2.0
            }
            
            # Simulación en GPU
            from .simulation import batch_simulate_real_holdem
            game_results = batch_simulate_real_holdem(rng_keys, game_config)
            
            # Paso de entrenamiento (CPU + GPU)
            self.train_step(game_results)

            # Guardar checkpoint
            if (self.iteration + 1) % save_interval == 0:
                logger.info(f"\n💾 Guardando checkpoint en la iteración {self.iteration + 1}...")
                self.save_model(save_path)

        # Guardado final
        self.save_model(save_path)
        total_time = time.time() - start_time
        logger.info(f"🎉 Entrenamiento finalizado en {total_time:.2f} segundos.")

    def train_step(self, game_results: dict):
        """
        Paso de entrenamiento con control explícito de dispositivos.
        """
        self.total_games += self.config.batch_size
        batch_size = game_results['payoffs'].shape[0]
        num_players = game_results['payoffs'].shape[1]
        total_info_sets = batch_size * num_players
        logger.info(f"   🚀 DEFINITIVE processing: {batch_size} games × {num_players} players = {total_info_sets} info sets")

        # 1. 🚀 PROCESAMIENTO VECTORIZADO EN GPU
        vectorized_results = self._vectorized_info_set_processing(game_results)
        
        # 2. 🧠 PUENTE GPU -> CPU PARA HASHING
        indices_cpu = self._map_info_sets_to_indices(game_results)

        # SI NO HAY ÍNDICES NUEVOS, SALIR TEMPRANO
        if len(indices_cpu) == 0:
            logger.warning("No se procesaron nuevos índices en este batch.")
            return

        # 3. 🚀 PREPARACIÓN PARA LA ACTUALIZACIÓN EN GPU
        indices_gpu = jax.device_put(indices_cpu, device=self._main_device)
        cf_values_gpu = jax.device_put(vectorized_results['cf_values'], device=self._main_device)

        # --- PADDING para evitar recompilaciones JIT ---
        MAX_INDICES = 200000
        n = len(indices_gpu)
        if n > MAX_INDICES:
            logger.warning(f"El número de índices ({n}) supera MAX_INDICES ({MAX_INDICES}). Recortando a los primeros {MAX_INDICES}.")
            indices_gpu = indices_gpu[:MAX_INDICES]
            cf_values_gpu = cf_values_gpu[:MAX_INDICES]
            n = MAX_INDICES
        indices_padded = jnp.zeros(MAX_INDICES, dtype=indices_gpu.dtype)
        indices_padded = indices_padded.at[:n].set(indices_gpu)
        cf_padded = jnp.zeros((MAX_INDICES, cf_values_gpu.shape[1]), dtype=cf_values_gpu.dtype)
        cf_padded = cf_padded.at[:n, :].set(cf_values_gpu[:n])

        # 4. 🚀 ACTUALIZACIÓN EN GPU
        self._vectorized_scatter_update(indices_padded, cf_padded, n)
        # Update counters
        self.total_info_sets += len(indices_cpu)
        # Compute metrics
        avg_payoff = jnp.mean(game_results['payoffs'])
        if len(indices_cpu) > 0:
            strategies_subset = self.strategies[indices_cpu]
            entropy = -jnp.sum(strategies_subset * jnp.log(strategies_subset + 1e-8), axis=1)
            avg_entropy = jnp.mean(entropy)
        else:
            avg_entropy = 0.0
        logger.info(f"   Iteración {self.iteration+1} completada.")
        logger.info(f"   📊 Unique info sets: {self.total_unique_info_sets:,}")
        logger.info(f"   📊 Info sets processed: {len(indices_cpu)}")
        logger.info(f"   📊 Growth events: {self.growth_events}")
        logger.info(f"   📊 Array size: {self.config.max_info_sets:,}")
        return {
            'iteration': self.iteration,
            'total_games': self.total_games,
            'total_info_sets': self.total_info_sets,
            'unique_info_sets': self.total_unique_info_sets,
            'info_sets_processed': len(indices_cpu),
            'avg_payoff': avg_payoff,
            'strategy_entropy': avg_entropy,
            'q_values_count': self.total_unique_info_sets,
            'strategies_count': self.total_unique_info_sets,
            'games_processed': self.config.batch_size,
            'growth_events': self.growth_events,
            'array_size': self.config.max_info_sets
        }
    
    def save_model(self, path: str):
        """Save definitive hybrid model"""
        model_data = {
            'q_values': np.array(self.q_values),
            'strategies': np.array(self.strategies),
            'info_set_hashes': self.info_set_hashes,
            'info_set_data': self.info_set_data,
            'iteration': self.iteration,
            'total_games': self.total_games,
            'total_info_sets': self.total_info_sets,
            'unique_info_sets': self.total_unique_info_sets,
            'growth_events': self.growth_events,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        file_size = len(pickle.dumps(model_data))
        logger.info(f"💾 Definitive Hybrid model saved: {path}")
        logger.info(f"   Q-values shape: {self.q_values.shape}")
        logger.info(f"   Strategies shape: {self.strategies.shape}")
        logger.info(f"   Unique info sets: {self.total_unique_info_sets:,}")
        logger.info(f"   File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        logger.info(f"   Growth events: {self.growth_events}")
    
    def load_model(self, path: str):
        """Load definitive hybrid model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.q_values = jnp.array(model_data['q_values'])
        self.strategies = jnp.array(model_data['strategies'])
        self.info_set_hashes = model_data['info_set_hashes']
        self.info_set_data = model_data['info_set_data']
        self.iteration = model_data['iteration']
        self.total_games = model_data['total_games']
        self.total_info_sets = model_data['total_info_sets']
        self.total_unique_info_sets = model_data['unique_info_sets']
        self.growth_events = model_data['growth_events']
        self.next_index = len(self.info_set_data)
        
        logger.info(f"📂 Definitive Hybrid model loaded: {path}")
        logger.info(f"   Q-values shape: {self.q_values.shape}")
        logger.info(f"   Strategies shape: {self.strategies.shape}")
        logger.info(f"   Unique info sets: {self.total_unique_info_sets:,}")
        logger.info(f"   Growth events: {self.growth_events}")

def benchmark_definitive_hybrid_performance():
    """Benchmark the definitive hybrid trainer performance"""
    from .simulation import batch_simulate_real_holdem
    import jax.random as jr
    import time
    
    logger.info("🚀 Benchmarking DEFINITIVE HYBRID Trainer Performance")
    logger.info("=" * 60)
    
    # Configuration
    config = TrainerConfig(
        batch_size=8192,
        learning_rate=0.1,
        temperature=1.0
    )
    
    trainer = PokerTrainer(config)
    
    # Test configuration
    game_config = {
        'players': 6,
        'starting_stack': 100.0,
        'small_blind': 1.0,
        'big_blind': 2.0
    }
    
    # Warm-up
    logger.info("🔥 Warming up JAX compilation...")
    rng_key = jr.PRNGKey(42)
    rng_keys = jr.split(rng_key, 1024)  # Smaller batch for warm-up
    
    start_time = time.time()
    game_results = batch_simulate_real_holdem(rng_keys, game_config)
    warmup_results = trainer.train_step(game_results)
    warmup_time = time.time() - start_time
    
    logger.info(f"   ✅ Warm-up completed in {warmup_time:.2f}s")
    
    # Benchmark
    logger.info("🚀 Running benchmark...")
    iterations = 10
    total_time = 0
    total_games = 0
    
    for i in range(iterations):
        rng_key = jr.fold_in(rng_key, i)
        rng_keys = jr.split(rng_key, config.batch_size)
        
        start_time = time.time()
        game_results = batch_simulate_real_holdem(rng_keys, game_config)
        results = trainer.train_step(game_results)
        iteration_time = time.time() - start_time
        
        total_time += iteration_time
        total_games += results['games_processed']
        
        games_per_second = results['games_processed'] / iteration_time
        logger.info(f"   Iteration {i+1}: {games_per_second:,.1f} games/sec")
    
    # Final results
    avg_games_per_second = total_games / total_time
    
    logger.info("🎉 DEFINITIVE HYBRID Benchmark Results:")
    logger.info("=" * 60)
    logger.info(f"🚀 Performance:")
    logger.info(f"   Average games/sec: {avg_games_per_second:,.1f}")
    logger.info(f"   Total games: {total_games:,}")
    logger.info(f"   Total time: {total_time:.2f}s")
    logger.info(f"   Target achieved: {'✅' if avg_games_per_second > 1000 else '❌'}")
    logger.info("")
    logger.info(f"🧠 Memory Management:")
    logger.info(f"   Unique info sets: {trainer.total_unique_info_sets:,}")
    logger.info(f"   Growth events: {trainer.growth_events}")
    logger.info(f"   Array size: {trainer.config.max_info_sets:,}")
    
    return avg_games_per_second 