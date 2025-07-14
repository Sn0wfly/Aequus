# analyze_convergence.py (Versión Final, compatible con arrays)

import os
import sys
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importamos la clase de configuración para que pickle pueda cargarla
from poker_bot.core.trainer import TrainerConfig

def load_model_data(filepath):
    """Carga los diccionarios de hashes y los arrays de estrategias."""
    print(f"Cargando: {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        hashes = data.get('info_set_hashes', {})
        strategies = data.get('strategies', np.array([]))
        
        if not isinstance(hashes, dict) or not isinstance(strategies, np.ndarray):
            print(" -> ERROR: El formato del modelo no es el esperado (se necesita 'info_set_hashes' y 'strategies').")
            return None, None
            
        print(f" -> Éxito. Encontrados {len(hashes):,} hashes y array de estrategias de forma {strategies.shape}.")
        return hashes, strategies
    except Exception as e:
        print(f" -> ERROR al cargar o leer el archivo: {e}")
        return None, None

def calculate_strategy_diff(hashes1, strategies1, hashes2, strategies2):
    """
    Calcula la diferencia promedio de estrategias entre dos snapshots,
    usando los hashes para encontrar los índices correctos en los arrays.
    """
    total_diff_sq = 0.0
    common_keys_count = 0
    
    # Itera sobre los hashes del snapshot más antiguo
    for info_hash, index1 in hashes1.items():
        # Busca si el mismo hash existe en el segundo snapshot
        if info_hash in hashes2:
            index2 = hashes2[info_hash]
            
            # Comprueba si los índices son válidos para ambos arrays de estrategias
            if index1 < len(strategies1) and index2 < len(strategies2):
                strat1 = strategies1[index1]
                strat2 = strategies2[index2]
                
                diff_sq = np.sum((strat1 - strat2)**2)
                total_diff_sq += diff_sq
                common_keys_count += 1
    
    return total_diff_sq / common_keys_count if common_keys_count > 0 else 0

def main():
    model_dir = "models"
    
    if not os.path.isdir(model_dir):
        print(f"Error: No se encuentra la carpeta '{model_dir}'.")
        return

    pattern = re.compile(r'.*_(\d+)\.pkl$')
    snapshots = sorted(
        [(int(match.group(1)), f) for f in os.listdir(model_dir) if (match := pattern.match(f))],
        key=lambda x: x[0]
    )

    if len(snapshots) < 2:
        print(f"Se encontraron {len(snapshots)} snapshots en la carpeta '{model_dir}'. Necesitas al menos dos para comparar.")
        return

    print(f"Se encontraron {len(snapshots)} snapshots para analizar.")
    
    diffs = []
    iterations = []

    for i in range(len(snapshots) - 1):
        iter1, path1_name = snapshots[i]
        iter2, path2_name = snapshots[i+1]
        
        path1 = os.path.join(model_dir, path1_name)
        path2 = os.path.join(model_dir, path2_name)
        
        print(f"\nComparando snapshot de iteración {iter1} con {iter2}...")
        
        hashes1, strategies1 = load_model_data(path1)
        hashes2, strategies2 = load_model_data(path2)
        
        if hashes1 is None or hashes2 is None:
            print(" -> Saltando comparación debido a un error de carga.")
            continue

        diff = calculate_strategy_diff(hashes1, strategies1, hashes2, strategies2)
        diffs.append(diff)
        iterations.append(iter2)
        
        print(f"  Diferencia promedio (MSE): {diff:.8f}")

    if not diffs:
        print("\nNo se pudieron calcular diferencias. Revisa los archivos.")
        return

    # Graficar los resultados
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(iterations, diffs, marker='o', linestyle='-', color='b')
    ax.set_title('Convergencia de la Estrategia de Aequus', fontsize=16)
    ax.set_xlabel('Iteración de Entrenamiento', fontsize=12)
    ax.set_ylabel('Diferencia Promedio de Estrategia (MSE)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_yscale('log')
    
    print("\nMostrando gráfico de convergencia... Cierra la ventana del gráfico para terminar.")
    plt.show()

if __name__ == "__main__":
    main()