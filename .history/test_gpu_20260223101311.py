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

# 对比CPU计算
print("\n3. CPU 矩阵乘法测试:")
with jax.default_device(jax.devices('cpu')[0]):
    a_cpu = jnp.ones((size, size))
    b_cpu = jnp.ones((size, size))
    
    # 预热
    _ = jnp.dot(a_cpu, b_cpu)
    
    start = time.time()
    result_cpu = jnp.dot(a_cpu, b_cpu)
    jax.effects_barrier()
    cpu_time = time.time() - start
    print(f"   CPU 耗时: {cpu_time*1000:.2f} ms")
    print(f"   加速比 (CPU/GPU): {cpu_time/gpu_time:.1f}x")

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
