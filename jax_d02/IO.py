"""
IO行脚本

Author: ZhouChk
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Tuple, Optional

def write_results_to_file(filename: str, A1: complex, A2: complex, A3: complex, 
                         B1: complex, B2: complex, B3: complex, lambda_param: float):
    """
    将鞍点优化结果写入文件
    
    Parameters:
    -----------
    filename : str
        输出文件名
    A1, A2, A3 : complex
        鞍点参数A
    B1, B2, B3 : float
        鞍点参数B
    lambda_param : float
        拉格朗日乘数
    """
    with open(filename, 'w') as f:
        f.write("# SWB Saddle Point Optimization Results\n")
        f.write("# Format: Parameter = Real + Imag*j\n")
        f.write(f"A1_real = {np.real(A1):.15e}\n")
        f.write(f"A1_imag = {np.imag(A1):.15e}\n")
        f.write(f"A2_real = {np.real(A2):.15e}\n")
        f.write(f"A2_imag = {np.imag(A2):.15e}\n")
        f.write(f"A3_real = {np.real(A3):.15e}\n")
        f.write(f"A3_imag = {np.imag(A3):.15e}\n")
        f.write(f"B1_real = {np.real(B1):.15e}\n")
        f.write(f"B1_imag = {np.imag(B1):.15e}\n")
        f.write(f"B2_real = {np.real(B2):.15e}\n")
        f.write(f"B2_imag = {np.imag(B2):.15e}\n")
        f.write(f"B3_real = {np.real(B3):.15e}\n")
        f.write(f"B3_imag = {np.imag(B3):.15e}\n")
        f.write(f"lambda = {lambda_param:.15e}\n")
    print(f"结果已保存到: {filename}")


def read_results_from_file(filename: str) -> Tuple[complex, complex, complex, complex, complex, complex, float]:
    """
    从文件读取鞍点优化结果
    
    Parameters:
    -----------
    filename : str
        输入文件名
        
    Returns:
    --------
    tuple
        (A1, A2, A3, B1, B2, B3, lambda_param)
    """
    data = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            key, value = line.split('=')
            data[key.strip()] = float(value.strip())
    
    A1 = data['A1_real'] + 1j * data['A1_imag']
    A2 = data['A2_real'] + 1j * data['A2_imag']
    A3 = data['A3_real'] + 1j * data['A3_imag']
    B1 = data['B1_real'] + 1j * data['B1_imag']
    B2 = data['B2_real'] + 1j * data['B2_imag']
    B3 = data['B3_real'] + 1j * data['B3_imag']
    lambda_param = data['lambda']
    
    print(f"从文件读取结果: {filename}")
    print(f"  A1 = {A1}")
    print(f"  B1 = {B1}")
    print(f"  lambda = {lambda_param}")
    
    return A1, A2, A3, B1, B2, B3, lambda_param

def load_spectral_data(filepath):
    """
    加载光谱数据文件
    
    Parameters:
    -----------
    filepath : str
        .npz 文件路径
        
    Returns:
    --------
    dict : 包含以下键的字典
        - k_path: (n_points, 2) k空间路径
        - k_distances: (n_points,) 沿路径的距离
        - omega_idx: (n_omega,) 频率点
        - spectral_intensity: (n_omega, n_points) 光谱强度
        - k_tick_positions: 高对称点位置
        - k_tick_labels: 高对称点标签
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    data = np.load(filepath, allow_pickle=True)
    
    # 返回字典形式
    result = {}
    for key in data.files:
        result[key] = data[key]
    
    return result