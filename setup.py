# setup.py (Versión Robusta y Moderna)

from setuptools import setup, find_packages, Extension

# --- Configuración ---
# No necesitas importar Cython aquí arriba. `setup_requires` se encargará de ello.
# Dejamos numpy para el `include_dirs`, que se ejecuta después de la instalación de dependencias.

def get_extensions():
    """
    Función para definir las extensiones solo cuando son necesarias,
    evitando errores de importación de Cython.
    """
    import numpy
    from Cython.Build import cythonize

    extensions = [
        Extension(
            "poker_bot.core.hasher",
            ["poker_bot/core/hasher.pyx"],
            include_dirs=[numpy.get_include()]
        )
    ]
    # `cythonize` convierte nuestros archivos .pyx en archivos .c
    return cythonize(extensions, language_level="3")


setup(
    name="aequus-poker-bot", # Nuevo nombre del paquete
    version="1.0.0",
    packages=find_packages(),
    
    # ¡LA CLAVE ESTÁ AQUÍ!
    # `setup_requires` le dice a pip: "Instala estos paquetes ANTES de ejecutar el resto del setup".
    setup_requires=[
        'setuptools>=64', # Versión moderna de setuptools
        'cython>=3.0.0',
        'numpy>=1.21.0'
    ],
    
    # `ext_modules` ahora se define de forma "lazy" o perezosa.
    # Se llamará a `get_extensions()` después de que `setup_requires` haya instalado Cython.
    ext_modules=get_extensions(),

    # `install_requires` son las dependencias que el paquete necesita para CORRER.
    install_requires=[
        'jax[cuda12_pip]',
        'numpy',
        'click',
        'PyYAML',
        'tqdm',
        'psutil',
        'phevaluator',
        # Cython ya no es estrictamente necesario aquí, porque solo se usa para compilar,
        # pero es bueno dejarlo por claridad.
        'Cython',
    ],
    
    entry_points={
        'console_scripts': [
            # Renombramos el comando a `aequus` para que coincida con el proyecto
            'aequus=poker_bot.cli:cli',
        ],
    },
    python_requires=">=3.8",
    zip_safe=False, # Necesario para las extensiones de Cython
)