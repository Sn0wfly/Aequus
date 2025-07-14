# setup.py (Versión a prueba de balas)
from setuptools import setup, find_packages

setup(
    name="aequus-poker-bot",
    version="1.3.0", # Incrementamos la versión
    packages=find_packages(),
    install_requires=[
        'jax[cuda12_pip]',
        'numpy',
        'click',
        'PyYAML',
        'tqdm',
        'psutil',
        'phevaluator',
    ],
    entry_points={
        'console_scripts': [
            'aequus=poker_bot.cli:cli',
        ],
    },
    python_requires=">=3.8",
)