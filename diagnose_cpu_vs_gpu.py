#!/usr/bin/env python
"""快速诊断：为什么CPU比GPU快"""

import jax
import jax.numpy as jnp
import time
import numpy as np

print("="*80)
print("【诊断】CPU vs GPU 性能对比")
print("="*80)

# 1. 检查计算设备配置
print("\n【1】设备检查")
print(f"JAX 版本: {jax.__version__}")
print(f"可用设备: {jax.devices()}")

# 尝试强制 GPU
jax.config.update("jax_enable_x64", True)
devices = jax.devices()
gpu_devices = [d for d in devices if 'cuda' in str(d).lower()]
print(f"GPU 设备: {gpu_devices}")

# 2. 单个 vmap 操作的性能对比
print("\n【2】单次 vmap 操作性能")

def compute_single(k):
    """模拟光谱计算中的单个 k 点操作"""
    return jnp.sum(jnp.sin(k) * jnp.cos(k))

kpath = jnp.linspace(0, 2*3.14159, 500)  # 500 个 k 点

# 测试 vmap 性能
@jax.jit
def compute_batch_jit(k_array):
    return jax.vmap(compute_single)(k_array)

print(f"输入大小: {kpath.shape}")

# 预热
_ = compute_batch_jit(kpath)
jax.effects_barrier()

# 测量 GPU 上的性能
start = time.time()
for _ in range(5):
    result_gpu = compute_batch_jit(kpath)
    jax.effects_barrier()
gpu_time = (time.time() - start) / 5

print(f"GPU vmap 耗时: {gpu_time*1000:.2f}ms")

# 3. 检查是否存在数据转移开销
print("\n【3】数据转移检查")

# 测试频繁的 numpy → jax 转换
numpy_arr = np.ones(500)
start = time.time()
for _ in range(100):
    jax_arr = jnp.asarray(numpy_arr)
numpy_to_jax_time = (time.time() - start) / 100

print(f"NumPy → JAX 转移（500元素）: {numpy_to_jax_time*1000:.3f}ms")
print(f"100 次转移总耗时: {numpy_to_jax_time*100*1000:.1f}ms")

# 4. 检查 Bogoliubov_transform_jax_batch 的性能
print("\n【4】关键函数性能分析")
try:
    from bogoliubov_transform_jax import Bogoliubov_transform_jax_batch
    
    # 小规模测试
    k1_test = jnp.ones(10)
    k2_test = jnp.ones(10)
    
    start = time.time()
    U, eng = Bogoliubov_transform_jax_batch(
        0, k1_test, k2_test, 2*np.pi/3, 4*np.pi/3,
        0.49j, 0.49j, 0.49j, 0.226, -0.226, 0.226, 0.94, 0.1,
        1.0, 1.0, 1.0
    )
    jax.effects_barrier()
    first_time = time.time() - start
    
    start = time.time()
    U, eng = Bogoliubov_transform_jax_batch(
        0, k1_test, k2_test, 2*np.pi/3, 4*np.pi/3,
        0.49j, 0.49j, 0.49j, 0.226, -0.226, 0.226, 0.94, 0.1,
        1.0, 1.0, 1.0
    )
    jax.effects_barrier()
    second_time = time.time() - start
    
    print(f"Bogoliubov_transform_jax_batch:")
    print(f"  首次调用（含编译）: {first_time*1000:.2f}ms")
    print(f"  后续调用（纯计算）: {second_time*1000:.2f}ms")
    
    if first_time / second_time > 10:
        print(f"  ⚠️ 编译开销很大 ({first_time/second_time:.0f}x)")
        print(f"  → 大量小规模调用会被编译时间主导")
        
except Exception as e:
    print(f"无法加载该函数: {e}")

# 5. 检查是否有隐藏的数据转移或同步操作
print("\n【5】可能的性能瓶颈检查")

print("""
如果 CPU 比 GPU 快，可能的原因：

1. ⚠️ 计算规模太小
   - L1=10 时只有 100 个 k 点
   - 100 个 k 点 × 9 个通道 × 100 个频率 = 90,000 个小计算
   - 每个计算都需要 JIT 编译 → 总编译时间巨大！
   
2. ⚠️ 频繁的 JIT 编译
   - 如果代码中有动态形状或条件分支 → 多次编译
   - 每次编译可能 100+ ms
   - 9 个通道 × 1000 个编译 = 900 秒！
   
3. ⚠️ 编译时间 > 计算时间
   - 小规模问题上，开销经常主导
   - vmap 虽然向量化了，但仍需编译
   
4. ⚠️ GPU 启动时间
   - GPU 初始化可能占用 1-2 秒
   - 小问题可能无法摊销此开销
   
5. ✓ CPU 是多核合适问题大小
   - L1=10 的问题对 CPU 来说刚好
   - NumPy/SciPy 已做过多年优化
   - 16 核 CPU 可能胜过小 GPU
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

建议的调查步骤：

1. 运行更大的 L1 值:
   python run_swb_calculation_jax.py --L1=20 --device='gpu'
   python run_swb_calculation_jax.py --L1=20 --device='cpu'
   
   预期: L1=20 时 GPU 应该更快（工作量增加，开销相对减少）

2. 添加时间测量:
   在 spectral_calculation_jax.py 中添加打印语句：
   
   import time
   start = time.time()
   spec = calculate_spectral_jax_vectorized(...)
   elapsed = time.time() - start
   print(f"光谱计算耗时: {elapsed:.2f}s")

3. 检查是否真的在使用 GPU:
   在运行计算的同时执行:
   python monitor_gpu.py --interval=1
   
   如果 GPU 利用率一直为 0, 说明代码根本没用 GPU

4. 启用 JAX 日志:
   JAX_PLATFORMS=cpu python run_swb_calculation_jax.py --L1=10 --device='gpu'
   
   查看是否有编译警告或错误
""")

print("="*80)
