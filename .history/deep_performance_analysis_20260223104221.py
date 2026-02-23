#!/usr/bin/env python
"""
深度性能分析 - 找出慢的原因
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import psutil
import os

# 启用x64
jax.config.update("jax_enable_x64", True)

print("="*80)
print("深度性能分析 - GPU vs CPU")
print("="*80)

# 1. 检查GPU使用情况
print("\n【1】GPU状态检查")
devices = jax.devices()
gpu_devices = [d for d in devices if 'cuda' in str(d).lower()]
print(f"可用CUDA设备: {gpu_devices}")

try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                            '--format=csv,noheader'],
                           capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"nvidia-smi输出: {result.stdout.strip()}")
except:
    print("无法获取nvidia-smi信息")

# 2. 性能对比 - 不同规模的矩阵乘法
print("\n【2】矩阵乘法性能对比（GPU vs CPU）")
print("-"*80)

sizes = [500, 1000, 2000, 3000, 4000]
results = []

for size in sizes:
    print(f"\n测试规模: {size}x{size}")
    
    # GPU计算
    a_gpu = jnp.ones((size, size), dtype=jnp.complex128)
    b_gpu = jnp.ones((size, size), dtype=jnp.complex128)
    
    @jax.jit
    def matmul_gpu(x, y):
        return jnp.dot(x, y)
    
    # 预热
    _ = matmul_gpu(a_gpu, b_gpu)
    jax.effects_barrier()
    
    # 测量GPU
    start = time.time()
    for _ in range(3):
        _ = matmul_gpu(a_gpu, b_gpu)
        jax.effects_barrier()
    gpu_time = (time.time() - start) / 3
    
    # CPU计算
    a_cpu = np.ones((size, size), dtype=np.complex128)
    b_cpu = np.ones((size, size), dtype=np.complex128)
    
    start = time.time()
    for _ in range(3):
        _ = np.dot(a_cpu, b_cpu)
    cpu_time = (time.time() - start) / 3
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
    
    print(f"  GPU: {gpu_time*1000:.2f}ms")
    print(f"  CPU: {cpu_time*1000:.2f}ms")
    print(f"  加速比: {speedup:.1f}x")
    
    results.append({
        'size': size,
        'gpu_time': gpu_time,
        'cpu_time': cpu_time,
        'speedup': speedup
    })
    
    if speedup < 1.0:
        print(f"  ⚠️ GPU比CPU还慢！可能存在问题")

# 3. 数据转移开销分析
print("\n【3】数据转移开销分析")
print("-"*80)

transfer_tests = [100, 500, 1000, 5000]

for size in transfer_tests:
    # NumPy → JAX转移
    arr = np.ones((size, size), dtype=np.complex128)
    
    start = time.time()
    for _ in range(10):
        jax_arr = jnp.asarray(arr)
    transfer_time = (time.time() - start) / 10
    
    # 转移速度（GB/s）
    data_size_gb = (size * size * 16) / (1024**3)  # complex128 = 16bytes
    transfer_speed = data_size_gb / transfer_time if transfer_time > 0 else 0
    
    print(f"矩阵大小: {size}x{size} ({data_size_gb:.3f}GB)")
    print(f"  转移时间: {transfer_time*1000:.2f}ms")
    print(f"  转移速度: {transfer_speed:.1f}GB/s")

# 4. JIT编译开销
print("\n【4】JIT编译开销分析")
print("-"*80)

a = jnp.ones((2000, 2000), dtype=jnp.complex128)
b = jnp.ones((2000, 2000), dtype=jnp.complex128)

@jax.jit
def test_jit(x, y):
    return jnp.dot(x, y)

# 第一次调用（包含编译）
start = time.time()
_ = test_jit(a, b)
jax.effects_barrier()
first_call = time.time() - start

# 第二次调用（只计算）
start = time.time()
_ = test_jit(a, b)
jax.effects_barrier()
second_call = time.time() - start

print(f"首次调用（含JIT编译）: {first_call*1000:.2f}ms")
print(f"后续调用（纯计算）: {second_call*1000:.2f}ms")
print(f"编译开销: {(first_call - second_call)*1000:.2f}ms ({first_call/second_call:.1f}x)")

if first_call / second_call > 50:
    print("⚠️ JIT编译开销非常大，小规模计算会变慢")

# 5. 内存压力测试
print("\n【5】GPU内存压力和代码执行效率")
print("-"*80)

# 模拟实际计算的内存访问模式
print("测试实际计算任务（类似SWB计算）...")

nk = 100  # k点数量
nw = 100  # 频率数量

# 分配GPU内存
k_points = jnp.ones((nk, 2), dtype=jnp.complex128)
freq_array = jnp.ones(nw, dtype=jnp.complex128)

@jax.jit
def sim_swb_calc(k, freq):
    """模拟SWB计算中的主要运算"""
    # 计算哈密顿量（向量化）
    H = jnp.zeros((k.shape[0], 4, 4), dtype=jnp.complex128)
    
    # 模拟对频率的循环
    results = []
    for w in freq:
        # 构建能量分母
        E = w - 0.5j
        # 计算格林函数
        G = 1.0 / (E + 1.0j)
        results.append(G)
    
    return jnp.array(results)

start = time.time()
result = sim_swb_calc(k_points, freq_array)
jax.effects_barrier()
elapsed = time.time() - start

print(f"模拟SWB计算（{nk}个k点, {nw}个频率）: {elapsed*1000:.2f}ms")

# 6. 当前系统状态
print("\n【6】系统状态")
print("-"*80)
print(f"CPU使用率: {psutil.cpu_percent(interval=1):.1f}%")
print(f"内存使用: {psutil.virtual_memory().percent:.1f}%")

try:
    # 获取当前Python进程的内存
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"当前进程内存: {mem_info.rss / (1024**2):.1f}MB")
except:
    pass

# 7. 诊断结论
print("\n【7】诊断结论")
print("="*80)

gpu_ok = True
if results:
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    if avg_speedup < 1.5:
        print("⚠️ GPU相对于CPU的加速不明显（< 1.5x）")
        print("   可能原因:")
        print("   1. 计算规模太小，无法填满GPU")
        print("   2. 内存转移开销太大")
        print("   3. 代码中有大量数据转移")
        print("   4. GPU驱动程序问题")
        gpu_ok = False
    else:
        print(f"✓ GPU加速效果好 (平均 {avg_speedup:.1f}x)")

# 8. 建议
print("\n【8】优化建议】")
print("="*80)

print("""
如果GPU仍然比CPU慢或加速不明显:

【立即检查】
1. 运行时监控GPU使用:
   在另一个终端执行:
   python monitor_gpu.py --interval=0.5
   
   观察:
   - GPU利用率是否> 10%
   - 显存占用是否增长
   
2. 检查计算规模:
   - 如果L1=5 (太小), 编译开销会主导
   - 尝试增大到L1=20或更大
   
3. 检查代码中的数据转移:
   搜索代码中的 jnp.asarray(), np.array(), .to_numpy() 等
   这些都会触发CPU↔GPU数据转移

【性能优化】
1. 批量计算（避免循环中的小规模计算）
2. 减少数据转移次数（数据在GPU上"留着"）
3. 使用vmap替代循环进行向量化
4. 检查是否所有关键函数都被@jax.jit装饰

【如果问题仍未解决】
可能是:
- CUDA版本不匹配
- 驱动程序问题  
- 硬件配置问题

检查命令:
  python -c "import jax; print(jax.config.jax_platform_name)"  # 应该是gpu
  python -c "import jaxlib; print(jaxlib.cuda.__version__)"     # CUDA版本
""")

print("="*80)
