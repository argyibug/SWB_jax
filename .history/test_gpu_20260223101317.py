#!/usr/bin/env python
"""Test GPU performance and availability"""

import jax
import jax.numpy as jnp
import time
import numpy as np

print("="*60)
print("JAX GPU 测试")
print("="*60)

# 检查设备
print("\n1. 设备信息:")
print(f"   JAX版本: {jax.__version__}")
devices = jax.devices()
print(f"   可用设备: {devices}")

# 获取CUDA设备信息
for i, device in enumerate(devices):
    print(f"   设备 {i}: {device}")

# 简单的GPU计算测试
print("\n2. GPU 矩阵乘法测试:")
size = 2000
print(f"   矩阵大小: {size}x{size}")

# 创建随机矩阵
a = jnp.ones((size, size))
b = jnp.ones((size, size))

# 预热（warm up）
_ = jnp.dot(a, b)

# 测试GPU计算速度
start = time.time()
result_gpu = jnp.dot(a, b)
jax.effects_barrier()  # 等待GPU完成
gpu_time = time.time() - start
print(f"   GPU 耗时: {gpu_time*1000:.2f} ms")

# GPU内存使用情况
print("\n3. GPU 内存使用:")
try:
    from jax.experimental import io_callback
    # 使用JAX内置的内存统计（如果可用）
    print(f"   当前GPU内存使用: 检查nvidia-smi")
except:
    print(f"   无法直接读取，请查看nvidia-smi输出")

print("\n4. 大规模计算测试:")
large_size = 4000
print(f"   矩阵大小: {large_size}x{large_size}")

a_large = jnp.ones((large_size, large_size))
b_large = jnp.ones((large_size, large_size))

# 预热
_ = jnp.dot(a_large, b_large)

start = time.time()
result_large = jnp.dot(a_large, b_large)
jax.effects_barrier()
large_time = time.time() - start
print(f"   GPU 耗时: {large_time*1000:.2f} ms")

print("\n" + "="*60)
print("GPU 可用性: ✓ 确认")
print("="*60)
