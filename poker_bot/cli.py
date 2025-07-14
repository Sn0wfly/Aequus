# poker_bot/cli.py

"""
Interfaz de Línea de Comandos para JaxPoker.
"""

import logging
import os
import click
import jax

# Importa las clases y configuraciones principales desde la nueva estructura
from .core.trainer import PokerTrainer, TrainerConfig
from .bot import PokerBot

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """JaxPoker: Un entrenador de IA de Póker de alto rendimiento."""
    pass

@cli.command()
@click.option('--iterations', default=10000, help='Número de iteraciones de entrenamiento.')
@click.option('--batch-size', default=8192, help='Tamaño del batch para la simulación en GPU.')
@click.option('--save-interval', default=1000, help='Guardar checkpoint cada N iteraciones.')
@click.option('--model-path', default='models/gto_model.pkl', help='Ruta para guardar el modelo entrenado.')
def train(iterations: int, batch-size: int, save_interval: int, model_path: str):
    """Entrena el modelo de IA de Póker usando el entrenador GTO."""
    
    logger.info("🚀 Iniciando el entrenamiento del modelo GTO de JaxPoker...")
    logger.info(f"Dispositivos JAX detectados: {jax.devices()}")
    
    # Crear el directorio de modelos si no existe
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Configurar el entrenador
    config = TrainerConfig(
        batch_size=batch_size
        # Puedes añadir más parámetros de configuración aquí si los mueves desde el trainer
    )
    trainer = PokerTrainer(config)

    # Lógica para cargar un checkpoint si existe
    if os.path.exists(model_path):
        logger.info(f"📂 Cargando checkpoint existente desde: {model_path}")
        try:
            trainer.load_model(model_path)
            logger.info(f"✅ Reanudando entrenamiento desde la iteración {trainer.iteration}.")
        except Exception as e:
            logger.error(f"❌ No se pudo cargar el modelo. Empezando de cero. Error: {e}")

    # Iniciar el bucle de entrenamiento
    trainer.train(
        num_iterations=iterations,
        save_path=model_path,
        save_interval=save_interval
    )

    logger.info("🎉 ¡Entrenamiento completado!")

@cli.command()
@click.option('--model', required=True, help='Ruta al modelo GTO entrenado (.pkl).')
def play(model: str):
    """Inicia una sesión de juego interactiva contra el bot entrenado."""
    if not os.path.exists(model):
        logger.error(f"Modelo no encontrado en la ruta: {model}")
        return

    logger.info(f"🤖 Cargando bot con el modelo: {model}")
    
    # La clase `PokerBot` necesitará ser actualizada para cargar el nuevo formato del modelo
    # y para tener una lógica de juego interactivo.
    # bot = PokerBot(model_path=model)
    # bot.start_interactive_session()

    click.echo("Funcionalidad de juego interactivo aún no implementada.")

if __name__ == '__main__':
    cli()