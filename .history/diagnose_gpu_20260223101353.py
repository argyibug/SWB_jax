#!/usr/bin/env python
"""诊断GPU性能问题"""

import jax
import jax.numpy as jnp
import time
import numpy as np
import sys

print("="*70)
print("JAX GPU 诊断")
print("="*70)

# 1. 设备检查
print("\n【设备检查】")
devices = jax.devices()
print(f"可用设备: {devices}")

# 2. JAX配置检查
print("\n【JAX配置】")
print(f"启用x64: {jax.config.jax_enable_x64}")
print(f"平台: {jax.config.jax_platform_name if hasattr(jax.config, 'jax_platform_name') else 'N/A'}")

# 3. GPU/CUDA识别问题
print("\n【设备识别分析】")
try:
    # JAX 0.8.1方式
    for i, device in enumerate(devices):
        device_str = str(device)
        print(f"  设备{i}: {device_str}")
        # CUDA设备会包含 cuda:
        if 'cuda' in device_str.lower():
            print(f"    ✓ CUDA GPU已识别")
        # JAX 0.8.1中没有.platform属性
        # 这可能是GPU检测失败的原因
except Exception as e:
    print(f"  ✗ 设备检查出错: {e}")

# 4. JIT编译测试
print("\n【JIT编译测试】")

@jax.jit
def simple_matmul(A, B):
    return jnp.dot(A, B)

size = 2000
A = jnp.ones((size, size), dtype=jnp.complex128)
B = jnp.ones((size, size), dtype=jnp.complex128)

# 第一次调用（包含JIT编译）
print(f"  输入大小: {size}x{size} (complex128)")
start = time.time()
result = simple_matmul(A, B)
jax.effects_barrier()
first_call = time.time() - start
print(f"  首次调用耗时: {first_call*1000:.2f} ms (包含JIT编译)")

# 第二次调用（只执行计算）
start = time.time()
result = simple_matmul(A, B)
jax.effects_barrier()
second_call = time.time() - start
print(f"  第二次调用耗时: {second_call*1000:.2f} ms (纯计算)")

if first_call > second_call * 3:
    print(f"  ⚠️ JIT编译耗时很长 ({first_call/second_call:.1f}x)")
else:
    print(f"  ✓ 编译时间合理")

# 5. 内存转移测试（可能比GPU计算慢）
print("\n【数据转移vs计算性能】")

# 测试numpy -> JAX的转换
start = time.time()
for _ in range(100):
    arr = np.ones((2000, 2000), dtype=np.complex128)
    jax_arr = jnp.asarray(arr)
transfer_time_1 = (time.time() - start) / 100
print(f"  NumPy → JAX转移 (2000x2000): {transfer_time_1*1000:.3f} ms")

# GPU上的计算时间
A_small = jnp.ones((2000, 2000), dtype=jnp.complex128)
B_small = jnp.ones((2000, 2000), dtype=jnp.complex128)
@jax.jit
def matmul(x, y):
    return jnp.dot(x, y)

start = time.time()
for _ in range(10):
    matmul(A_small, B_small)
    jax.effects_barrier()
compute_time = (time.time() - start) / 10
print(f"  GPU矩阵乘法 (2000x2000): {compute_time*1000:.3f} ms")

if transfer_time_1 > compute_time:
    print(f"  ⚠️ 数据转移比GPU计算慢! ({transfer_time_1/compute_time:.1f}x)")
    print(f"    → 可能原因: 过多的CPU↔GPU数据转移")

# 6. 动态形状问题
print("\n【动态形状问题检查】")
print("  提示: 动态形状会禁用JIT编译")
print("  如果输入形状在运行时变化，应使用static_argnums参数")

# 7. 建议
print("\n【性能优化建议】")
print("""
✓ 问题诊断完成。可能的Linux vs Windows性能差异原因:

1. 【GPU检测问题】
   - run_swb_calculation_jax.py中使用`d.platform == 'gpu'`检查
   - JAX 0.8.1中设备对象可能没有.platform属性
   - 导致GPU检测失败，实际运行在CPU

2. 【数据转移开销】
   - 如果频繁在numpy/CPU ↔ JAX/GPU转移数据
   - Linux驱动程序可能比Windows略慢

3. 【初始化开销】
   - CUDA初始化在首次使用时较慢
   - 在Linux上可能更明显

4. 【线程/调度差异】
   - Linux和Windows线程调度不同
   - CUDA上下文切换方式不同

【立即检查的修复】
  → 检查GPU是否真正在使用（nvidia-smi查看GPU内存占用）
  → 修复GPU检测代码（使用'cuda'字符串匹配）
  → 验证JIT编译是否生效
""")

print("\n" + "="*70)
