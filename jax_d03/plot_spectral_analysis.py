"""
光谱分析绘图脚本 - 读取并可视化光谱数据

Author: ZhouChk

功能:
1. 读取保存的channel aa光谱数据
2. 绘制频谱强度图
3. 绘制特定动量点的谱函数对频率的函数
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from IO import load_spectral_data
from visualization import *

import argparse
import os


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='光谱数据分析和可视化')
    parser.add_argument('--L1', type=int, default=10, help='晶格大小 (默认: 10)')
    parser.add_argument('--load', type=str, default='', help='指定数据文件路径')
    parser.add_argument('--k-index', type=int, default=None, help='动量点索引')
    parser.add_argument('--plot-all', action='store_true', help='绘制所有高对称点的谱函数')
    parser.add_argument('--channel', type=str, default='Sz_Channel', help='通道名称 (默认: Sz_Channel)')
    
    args = parser.parse_args()
    
    # 构建数据文件路径
    if not args.load:
        data_file = f'results/spectral_{args.channel}_L{args.L1}.dat.npz'
    else:
        data_file = args.load
    
    print(f"加载光谱数据: {data_file}")
    
    try:
        data = load_spectral_data(data_file)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print(f"请先运行 run_spectra_calculation_jax.py --L1 {args.L1}")
        return
    
    print(f"✓ 成功加载数据")
    print(f"  k点数量: {len(data['k_distances'])}")
    print(f"  频率点数量: {len(data['omega_idx'])}")
    print()
    
    # 绘制频谱强度图
    print("绘制频谱强度图...")
    plot_spectral_intensity(
        data['k_distances'],
        data['omega_idx'],
        data['spectral_intensity'],
        data['k_tick_positions'],
        data['k_tick_labels'],
        title=f"{args.channel} Spectral Intensity (L={args.L1})",
        intensity_range=(0, 10),  # 设置强度范围
        save_path=f'results/spectral_intensity_L{args.L1}.png'
    )
    print()
    
    # 绘制特定动量点或多个点的谱函数
    if args.plot_all:
        print("绘制所有高对称点的谱函数...")
        # 绘制所有高对称点
        for i, (pos, label) in enumerate(zip(data['k_tick_positions'], data['k_tick_labels'])):
            k_idx = np.argmin(np.abs(data['k_distances'] - pos))
            plot_momentum_point_spectrum(
                data, args.L1, k_index=k_idx,
                save_path=f'results/spectral_point_{label}_L{args.L1}.png',name=args.channel
            )
    else:
        print("绘制单个动量点的谱函数...")
        plot_momentum_point_spectrum(
            data, args.L1, k_index=args.k_index,
            save_path=f'results/spectral_single_point_L{args.L1}.png',name=args.channel
        )
    
    print("\n✓ 分析完成！")


if __name__ == "__main__":
    main()
