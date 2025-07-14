# setup.py (Versión a prueba de balas)
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy

# La definición de la extensión ahora es más explícita
# Se le dice que el "nombre del módulo" que se importará es poker_bot.core.hasher
# y que la fuente está en poker_bot/core/hasher.pyx
extensions = [
    Extension(
        "poker_bot.core.hasher",
        ["poker_bot/core/hasher.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="aequus-poker-bot",
    version="1.0.2", # Incrementamos de nuevo
    packages=find_packages(), # Encuentra automáticamente el paquete poker_bot

    # No necesitamos 'package_data' si 'ext_modules' está bien configurado
    
    # Las dependencias de compilación se quedan en pyproject.toml
    # (Asegúrate de que el pyproject.toml que te di antes sigue ahí)

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
    zip_safe=False,
)