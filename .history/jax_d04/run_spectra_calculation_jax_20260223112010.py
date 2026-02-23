"""
主运行脚本 - Spectra计算流程 (JAX GPU加速版本)

Author: ZhouChk
JAX优化: GPU加速批量计算
"""

import os
# 在导入JAX之前设置GPU内存选项（关键！）
# 启用动态显存分配，避免一开始预分配所有显存导致cublas初始化失败
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time
from typing import Tuple, Optional

# 导入各个模块
from gamma_functions import set_global_params
from saddle_point_optimization import optimize_saddle_point
from bogoliubov_transform_jax import Bogoliubov_transform_2_jax
from spectral_calculation_jax import calculate_spectral_jax_vectorized
from spectral_calculation import get_triangular_lattice_path, convert_to_cartesian_coordinates
from visualization import plot_dispersion, plot_spectral_intensity, plot_multiple_spectral_channels
from IO import write_results_to_file, read_results_from_file

# 启用64位精度
jax.config.update("jax_enable_x64", True)

# GPU内存优化：限制JAX使用GPU显存比例，避免cublas初始化失败
# RTX 4060有8GB显存，JAX默认预分配大部分，导致其他操作失败
# 设置为False启用动态内存分配（按需要申请而不是预分配）
try:
    jax.config.update('jax_default_prolog_preamble_flags', '')
    jax.config.gpu_memory_fraction = 0.7  # 限制到70%，留出空间给cuBLAS等
except:
    pass  # 如果设置失败就跳过

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
            # JAX 0.8.1: 使用字符串匹配检测CUDA设备
            gpu_devices = [d for d in jax.devices() if 'cuda' in str(d).lower()]
            if gpu_devices:
                jax.config.update('jax_platform_name', 'gpu')
                print(f"🚀 强制使用 GPU 进行计算 - {gpu_devices}")
            else:
                print("⚠️ GPU 不可用，自动回退到 CPU")
                jax.config.update('jax_platform_name', 'cpu')
        except Exception as e:
            print(f"⚠️ GPU 检测失败: {e}，自动回退到 CPU")
            jax.config.update('jax_platform_name', 'cpu')
    else:  # auto
        # 让 JAX 自动选择最佳设备
        try:
            # JAX 0.8.1: 使用字符串匹配检测CUDA设备
            gpu_devices = [d for d in jax.devices() if 'cuda' in str(d).lower()]
            if gpu_devices:
                print(f"🚀 自动选择: 使用 GPU 进行计算 - {gpu_devices}")
            else:
                print(f"🔧 自动选择: 使用 CPU 进行计算")
        except Exception as e:
            print(f"🔧 自动选择: 使用 CPU 进行计算 (错误: {e})")
    
    # 显示当前设备信息
    print(f"当前默认后端: {jax.default_backend()}")
    print(f"可用设备: {jax.devices()}")
    print()

def run_spectra_calculation_jax(L1: int,
                            A1: complex,
                            A2: complex,
                            A3: complex,
                            B1: complex,
                            B2: complex,
                            B3: complex,
                            lambda_param: float,
                            verbose: bool = True,
                            device: str = "auto",
                            beta: float = 100,
                            eta: float = 0.01,
                            d_omega: float = 0.01,
                            max_omega: float = 2.0):
    
    # 设置计算设备
    set_device_config(device)
    
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
    

    # 计算色散关系
    if verbose:
        print("\\n步骤3: 计算色散关系 (JAX GPU加速)")
    
    # 获取高对称点路径
    try:
        k_path, k_distances, k_tick_positions = get_triangular_lattice_path(L1)
        k_cartesian = convert_to_cartesian_coordinates(k_path)
        
        # 对于L1=10: n_points≈50, L1=20: n_points≈100, L1=50: n_points≈250
        n_points = min(len(k_cartesian), max(50, 5 * L1))
        indices = np.linspace(0, len(k_cartesian)-1, n_points, dtype=int)
        
        # 批量处理所有k点 (JAX加速)
        k_path_selected = k_cartesian[indices]
        kx_batch = jnp.array(k_path_selected[:, 0])
        ky_batch = jnp.array(k_path_selected[:, 1])
        
        if verbose:
            print(f"  批量计算 {n_points} 个k点...")
        
        # 使用JAX批量Bogoliubov变换计算本征值 (返回2个值: Ubov, ek)
        _, ek_batch = Bogoliubov_transform_2_jax(
            0, kx_batch, ky_batch, Q1, Q2,
            A1, A2, A3, B1, B2, B3,
            lambda_param, h, J1plus, J2plus, J3plus
        )
        
        # 转换为NumPy用于绘图
        eigenvalues = np.array(ek_batch[:, :2])  # 取前两个能带
        
        # 绘制色散关系
        import os
        os.makedirs('results', exist_ok=True)
        fig1 = plot_dispersion(
            k_distances[indices], eigenvalues, 
            k_tick_positions, ['X', 'M', 'Γ', "K'", 'M', 'Y', 'K'],
            title="Spinon Dispersion",
            save_path=f"results/dispersion_L{L1}.png"
        )
        
        if verbose:
            print("  色散关系计算完成并保存")
            
    except Exception as e:
        print(f"  色散关系计算失败: {e}")
    
    # 计算光谱函数 (简化版本)
    if verbose:
        print("\\n步骤4: 计算光谱函数")
    
    try:
        # 光谱参数 - omega密度可以保持固定，k点密度随色散关系一致
        omega_max = max_omega
        d_omega = d_omega
        n_omega = int(omega_max / d_omega)
        
        omega_idx = np.linspace(0.01, omega_max, n_omega)
        
        # 使用与色散关系完全相同的k点（密度随L1增加）
        k_path_spectral = k_cartesian[indices]  # 使用相同的indices
        k_distances_spectral = k_distances[indices]  # 使用相同的k距离
        
        if verbose:
            print(f"  使用 {len(k_path_spectral)} 个k点进行光谱计算")
        
        # 定义通道
        channels = [
            # 全部通道
            np.array([[1, 0, 0],
                     [0, 0, 0], 
                     [0, 0, 0]]),
            # Sz通道
            np.array([[0, 0, 0],
                     [0, 0, 0],
                     [0, 0, 1]]),
            # Sx + Sy通道
            np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]),
        ]
        
        channel_names = ['Sx_Channel', 'Sz_Channel', 'Sy_Channel']
        
        spectral_data_list = []
        
        total_spec_time = 0
        for i, (channel, name) in enumerate(zip(channels, channel_names)):
            if verbose:
                print(f"  计算 {name} (JAX GPU加速)...")
            
            # 添加计时
            spec_start = time.time()
            
            # calculate_spectral_jax_vectorized返回 (kpath_len, omega_len)
            spec = calculate_spectral_jax_vectorized(
                k_path_spectral, omega_idx, eta, channel,
                A1, A2, A3, B1, B2, B3, lambda_param, h,
                k1, k2, Q1, Q2,
                J1plus, J2plus, J3plus,
                batch_size=50,  # 优化: 均衡计算速度和显存使用
                beta=beta
            )
            
            spec_time = time.time() - spec_start
            total_spec_time += spec_time
            
            if verbose:
                print(f"    {name} 耗时: {spec_time:.2f}秒")
            
            spectral_data_list.append(spec.T)
        
        # 使用与色散关系相同的高对称点标签
        # 绘制光谱函数
        
        # 设置强度范围（可选）
        # 格式: [(vmin, vmax), ...] 对应每个通道
        intensity_ranges = [
            (0, 10),  # All Channels: 所有通道强度较大
            (0, 10),  # Sz Channel: 纵向通道
            (0, 10)   # Sx+Sy Channel: 横向通道
        ]
        
        fig2 = plot_multiple_spectral_channels(
            k_distances_spectral, omega_idx,
            spectral_data_list, channel_names,
            k_tick_positions, ['X', 'M', 'Γ', "K'", 'M', 'Y', 'K'],
            main_title=f"Spectral Intensity (L={L1})",
            omega_range=(0, omega_max),
            intensity_ranges=intensity_ranges,  # 添加强度范围设置
            save_path=f"results/spectral_L{L1}.png"
        )
        
            # 保存光谱数据到文件
            # spectral_data_list[0] 对应 'Sx Channels'，形状为 (omega_len, k_len)
        for i, name in enumerate(channel_names):
            channel_aa_data = spectral_data_list[i]  # (omega, k)
            spectral_output = {
                'k_path': k_path_spectral,  # (n_points, 2)
                'k_distances': k_distances_spectral,  # (n_points,)
                'omega_idx': omega_idx,  # (n_omega,)
                'spectral_intensity': channel_aa_data,  # (n_omega, n_points)
                'k_tick_positions': k_tick_positions,
                'k_tick_labels': ['X', 'M', 'Γ', "K'", 'M', 'Y', 'K']
            }
            
            # 使用numpy保存
            spectral_file = f'results/spectral_{name}_L{L1}.dat'
            np.savez(spectral_file, **spectral_output)
            
            if verbose:
                print(f"  {name} 光谱数据已保存到 {spectral_file}")
        
        if verbose:
            print(f"  光谱计算总耗时: {total_spec_time:.2f}秒")
            
    except Exception as e:
        print(f"  光谱函数计算失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 保存高对称点索引到文件
    try:
        k_tick_labels = ['X', 'M', 'Γ', "K'", 'M', 'Y', 'K']
        k_indices = []
        
        # 根据 k_distances 和 k_tick_positions 找出对应的索引
        for pos in k_tick_positions:
            # 找最接近的k点索引
            idx = np.argmin(np.abs(k_distances_spectral - pos))
            k_indices.append(idx)
        
        # 保存到文件
        with open(f'results/k_indx_L{L1}.dat', 'w') as f:
            f.write("# High Symmetry Points K-indices\n")
            f.write("# Format: Label Index\n")
            for label, idx in zip(k_tick_labels, k_indices):
                f.write(f"{label} {idx}\n")
        
        if verbose:
            print(f"高对称点索引已保存到 results/k_indx_L{L1}.dat")
    except Exception as e:
        print(f"保存高对称点索引失败: {e}")
    
    if verbose:
        print("\n计算完成！)")
        print(f"结果已保存为 results/dispersion_L{L1}.png 和 results/spectral_L{L1}.png")
    
    return {
        'saddle_point': (A1, A2, A3, B1, B2, B3, lambda_param),
        'system_params': {
            'L1': L1, 'L2': L2, 'Nsites': Nsites,
            'J1xy': J1xy, 'J2xy': J2xy, 'J3xy': J3xy, 'S': S
        }
    }

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SWB系统计算')
    parser.add_argument('--L1', type=int, default=10, help='晶格大小 (默认: 10)')
    parser.add_argument('--beta', type=float, default=100, help='反演温度 (默认: 100)')
    parser.add_argument('--eta', type=float, default=0.01, help='频率展宽 (默认: 0.01)')
    parser.add_argument('--domega', type=float, default=0.01, help='频率步长 (默认: 0.01)')
    parser.add_argument('--max_omega', type=float, default=2.0, help='最大频率 (默认: 2.0)')
    parser.add_argument('--quiet', action='store_true', help='安静模式')
    parser.add_argument('--load', type=str, default='', help='从文件加载结果 (默认: results/swb_L{L1}.dat)')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'gpu', 'cpu'],
                       help='计算设备: auto(自动), gpu(强制GPU), cpu(强制CPU) (默认: auto)')
    
    args = parser.parse_args()
    
    # 如果未指定加载路径，根据L1值生成默认路径
    if not args.load:
        args.load = f'results/swb_L{args.L1}.dat'
    
    print("三角格子反铁磁体光谱计算")
    print("=" * 50)
    
    # 加载鞍点优化结果
    try:
        A1, A2, A3, B1, B2, B3, lambda_param = read_results_from_file(args.load)
        print("\n加载的鞍点优化结果:")
        print(f"  A1 = {np.real(A1):.6f}+{np.imag(A1):.6f}i")
        print(f"  A2 = {np.real(A2):.6f}+{np.imag(A2):.6f}i")
        print(f"  A3 = {np.real(A3):.6f}+{np.imag(A3):.6f}i")
        print(f"  B1 = {np.real(B1):.6f}+{np.imag(B1):.6f}i")
        print(f"  B2 = {np.real(B2):.6f}+{np.imag(B2):.6f}i")
        print(f"  B3 = {np.real(B3):.6f}+{np.imag(B3):.6f}i")
        print(f"  lambda = {lambda_param:.6f}")
        print()
    except FileNotFoundError:
        print(f"\n错误: 文件 '{args.load}' 不存在")
        print("请先运行 run_swb_calculation.py 生成优化结果")
        return
    except Exception as e:
        print(f"\n读取文件失败: {e}")
        return
    
    # 运行光谱计算
    result = run_spectra_calculation_jax(
        L1=args.L1, 
        A1=A1, A2=A2, A3=A3, 
        B1=B1, B2=B2, B3=B3, 
        lambda_param=lambda_param, 
        verbose=not args.quiet,
        device=args.device,
        beta=args.beta,
        eta=args.eta,
        d_omega=args.domega,
        max_omega=args.max_omega
    )
    
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()