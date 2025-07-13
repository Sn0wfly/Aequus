#!/bin/bash

echo "🔧 INSTALLING CORRECT JAX CUDA VERSION..."

# Remove problematic XLA flags
unset XLA_FLAGS

# Check CUDA installation
echo "🔍 CUDA setup:"
nvidia-smi
nvcc --version

# Completely remove JAX
echo "🔄 Removing existing JAX installation..."
pip uninstall -y jax jaxlib

# Install CUDA-enabled jaxlib first
echo "📦 Installing CUDA-enabled jaxlib..."
pip install --upgrade --no-deps jaxlib==0.6.2+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install JAX
echo "📦 Installing JAX..."
pip install --upgrade jax==0.6.2

# Set environment variables
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export CUDA_VISIBLE_DEVICES=0

# Test JAX setup
echo "🔄 Testing JAX CUDA setup..."
python -c "
import os
import jax
import jax.numpy as jnp

print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())

try:
    # Test GPU operations
    gpu_devices = jax.devices('gpu')
    if len(gpu_devices) > 0:
        print('🎮 GPU devices found:', gpu_devices)
        with jax.default_device(gpu_devices[0]):
            x = jnp.array([1.0, 2.0, 3.0])
            y = jnp.array([4.0, 5.0, 6.0])
            z = x + y
            print('✅ GPU operations work:', z)
        print('🎉 GPU is working! Ready for poker training!')
    else:
        print('⚠️  No GPU devices found')
        
except Exception as e:
    print('❌ GPU test failed:', e)
    print('💻 Using CPU fallback')
    
    # Test CPU operations
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    z = x + y
    print('✅ CPU operations work:', z)
"

echo "✅ Setup complete!" 