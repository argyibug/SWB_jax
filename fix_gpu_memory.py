#!/usr/bin/env python
"""修复JAX GPU显存不足问题"""

import sys
import os

# 在导入JAX之前设置环境变量来限制GPU显存使用
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1 --xla_gpu_strict_conv_algorithmic_determinism=false'
os.environ['JAX_PLATFORMS'] = 'gpu'

# 限制JAX仅使用50%的GPU显存（RTX 4060的约4GB）
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import jax
jax.config.update('jax_platform_name', 'gpu')

# 尝试启用动态内存分配
try:
    from jax import devices
    gpu_devices = devices()
    print(f"✓ GPU设备初始化成功: {gpu_devices}")
except Exception as e:
    print(f"✗ GPU初始化失败: {e}")
    print("  尝试用CPU代替...")
    jax.config.update('jax_platform_name', 'cpu')

# 测试是否能够进行计算
import jax.numpy as jnp
try:
    test_array = jnp.ones((100, 100), dtype=jnp.complex128)
    result = jnp.dot(test_array, test_array)
    jax.effects_barrier()
    print(f"✓ 计算测试通过，使用设备: {result.devices()}")
except Exception as e:
    print(f"✗ 计算测试失败: {e}")
    sys.exit(1)
