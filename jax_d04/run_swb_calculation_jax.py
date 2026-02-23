"""
主运行脚本 - SWB计算流程

Author: ZhouChk
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from typing import Tuple, Optional

# 启用64位精度
jax.config.update("jax_enable_x64", True)

def set_device_config(device_type: str = "auto"):
    """
    设置 JAX 计算设备
    
    Parameters:
    -----------
    device_type : str
        设备类型 - "auto", "gpu", "cpu"
    """
    if device_type == "cpu":
        jax.config.update('jax_platform_name', 'cpu')
        print(f"🔧 强制使用 CPU 进行计算")
    elif device_type == "gpu":
        # 检查 GPU 是否可用（需要在设置平台之前检查）
        try:
            # 先尝试获取 GPU 设备
            gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
            if gpu_devices:
                jax.config.update('jax_platform_name', 'gpu')
                print(f"🚀 强制使用 GPU 进行计算")
            else:
                print("⚠️ GPU 不可用，自动回退到 CPU")
                jax.config.update('jax_platform_name', 'cpu')
        except:
            print("⚠️ GPU 检测失败，自动回退到 CPU")
            jax.config.update('jax_platform_name', 'cpu')
    else:  # auto
        # 让 JAX 自动选择最佳设备
        try:
            gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
            if gpu_devices:
                print(f"🚀 自动选择: 使用 GPU 进行计算")
            else:
                print(f"🔧 自动选择: 使用 CPU 进行计算")
        except:
            print(f"🔧 自动选择: 使用 CPU 进行计算")
    
    # 显示当前设备信息
    print(f"当前默认后端: {jax.default_backend()}")
    print(f"可用设备: {jax.devices()}")
    print()

# 导入各个模块 - 使用JAX版本
from gamma_functions import set_global_params
from saddle_point_optimization_jax import optimize_saddle_point
from bogoliubov_transform_jax import Bogoliubov_transform_2_jax
from spectral_calculation_jax import calculate_spectral_jax_vectorized
from spectral_calculation import get_triangular_lattice_path, convert_to_cartesian_coordinates
from visualization import plot_dispersion, plot_spectral_intensity, plot_multiple_spectral_channels
from IO import write_results_to_file, read_results_from_file

def run_complete_calculation(L1: int = 10, verbose: bool = True, device: str = "auto"):
    """
    运行完整的SWB计算
    
    Parameters:
    -----------
    L1 : int
        晶格大小 (使用较小值以便测试)
    verbose : bool
        是否打印详细信息
    device : str
        计算设备 - "auto", "gpu", "cpu"
    """
    
    # 设置计算设备
    set_device_config(device)
    
    if verbose:
        print("开始完整的SWB计算")
        print("=" * 50)
    
    # 1. 设置系统参数
    if verbose:
        print("步骤1: 设置系统参数")
    
    # 模型参数
    J1xy = J2xy = J3xy = 1.0
    J1z = J2z = J3z = 1.0  # 各向同性
    S = 0.5
    
    # 计算组合参数
    J1plus = (J1z + J1xy) / 2
    J2plus = (J2z + J2xy) / 2
    J3plus = (J3z + J3xy) / 2
    
    # 磁序参数
    Q1 = 2*np.pi/3
    Q2 = 4*np.pi/3
    
    # 晶格参数
    L2 = L1
    k1_1d = 2*np.pi/L1 * np.arange(L1)
    k2_1d = 2*np.pi/L2 * np.arange(L2)
    k1_2d, k2_2d = np.meshgrid(k1_1d, k2_1d, indexing='ij')
    k1 = k1_2d.flatten()
    k2 = k2_2d.flatten()
    Nsites = len(k1)
    h = 1.0 / Nsites
    
    # 设置全局参数
    set_global_params(J1plus=J1plus, J2plus=J2plus, J3plus=J3plus, Q1=Q1, Q2=Q2)
    
    if verbose:
        print(f"  晶格大小: {L1}x{L2}, 总格点数: {Nsites}")
        print(f"  交换耦合: J1={J1xy}, J2={J2xy}, J3={J3xy}")
        print(f"  自旋: S={S}")
    
    # 2. 鞍点优化
    if verbose:
        print("\\n步骤2: 鞍点优化")
    
    x0 = np.array([0.54, 0.3])  # 初始猜测
    
    start_time = time.time()
    try:
        result = optimize_saddle_point(
            k1, k2, h, Q1, Q2, x0, 
            S=S, J1plus=J1plus, J2plus=J2plus, J3plus=J3plus
        )
        A1, A2, A3, B1, B2, B3, lambda_param = result
        optimization_time = time.time() - start_time
        
        if verbose:
            print(f"  优化完成 (用时: {optimization_time:.2f}秒)")
            print(f"  A1 = {np.imag(A1):.6f}i")
            print(f"  B1 = {B1:.6f}")
            print(f"  lambda = {lambda_param:.6f}")
            
    except Exception as e:
        print(f"  鞍点优化失败: {e}")
        print("  使用预设参数继续计算")
        
        A1 = A2 = A3 = 0.49126303j
        B1 = 0.22640955
        B2 = -B1
        B3 = B1
        lambda_param = 0.94176189
    
    # 保存结果到文件
    import os
    os.makedirs('results', exist_ok=True)
    write_results_to_file(f'results/swb_L{L1}.dat', A1, A2, A3, B1, B2, B3, lambda_param)
    
    return A1, A2, A3, B1, B2, B3, lambda_param

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SWB系统计算')
    parser.add_argument('--L1', type=int, default=10, help='晶格大小 (默认: 10)')
    parser.add_argument('--quiet', action='store_true', help='安静模式')
    parser.add_argument('--load', type=str, help='从文件加载结果 (例如: results/swb.dat)')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'gpu', 'cpu'],
                       help='计算设备: auto(自动), gpu(强制GPU), cpu(强制CPU) (默认: auto)')
    
    args = parser.parse_args()
    
    # 如果指定了加载文件，读取并显示结果
    if args.load:
        try:
            A1, A2, A3, B1, B2, B3, lambda_param = read_results_from_file(args.load)
            print("\n加载的优化结果:")
            print(f"  A1 = {np.real(A1):.6f}+{np.imag(A1):.6f}i")
            print(f"  A2 = {np.real(A2):.6f}+{np.imag(A2):.6f}i")
            print(f"  A3 = {np.real(A3):.6f}+{np.imag(A3):.6f}i")
            print(f"  B1 = {np.real(B1):.6f}+{np.imag(B1):.6f}i")
            print(f"  B2 = {np.real(B2):.6f}+{np.imag(B2):.6f}i")
            print(f"  B3 = {np.real(B3):.6f}+{np.imag(B3):.6f}i")
            print(f"  lambda = {lambda_param:.6f}")
            return
        except FileNotFoundError:
            print(f"错误: 文件 '{args.load}' 不存在")
            return
        except Exception as e:
            print(f"读取文件失败: {e}")
            return
    
    args = parser.parse_args()
    
    print("三角格子反铁磁体自旋子系统计算")
    print("=" * 50)
    
    result = run_complete_calculation(L1=args.L1, verbose=not args.quiet, device=args.device)
    
    print("\\n" + "=" * 50)
    print("计算完成！")

if __name__ == "__main__":
    main()