from setuptools import setup, find_packages, Extension

def get_extensions():
    """Define las extensiones de Cython."""
    import numpy
    from Cython.Build import cythonize
    extensions = [
        Extension(
            "poker_bot.core.hasher",
            ["poker_bot/core/hasher.pyx"],
            include_dirs=[numpy.get_include()]
        )
    ]
    return cythonize(extensions, language_level="3")

setup(
    name="aequus-poker-bot",
    version="1.0.1", # Incrementamos la versión para forzar la actualización
    packages=find_packages(),
    
    # Añadimos esto para asegurar que el .pyx se incluya
    package_data={
        'poker_bot.core': ['*.pyx'],
    },
    
    setup_requires=[
        'setuptools>=64',
        'cython>=3.0.0',
        'numpy>=1.21.0'
    ],
    
    ext_modules=get_extensions(),

    install_requires=[
        'jax[cuda12_pip]',
        'numpy',
        'click',
        'PyYAML',
        'tqdm',
        'psutil',
        'phevaluator',
        'Cython',
    ],
    
    entry_points={
        'console_scripts': [
            'aequus=poker_bot.cli:cli',
        ],
    },
    python_requires=">=3.8",
    zip_safe=False,
)