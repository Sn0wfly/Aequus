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
    version="1.0.0",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=[
        'jax[cuda12_pip]',
        'numpy',
        'click',
        'PyYAML',
        'tqdm',
        'psutil',
        'phevaluator',
        'Cython', # Aún es bueno tenerlo aquí para algunos casos
    ],
    entry_points={
        'console_scripts': [
            'aequus=poker_bot.cli:cli',
        ],
    },
    python_requires=">=3.8",
    zip_safe=False,
)