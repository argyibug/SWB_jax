"""
谱函数计算模块

Author: ZhouChk
"""

import numpy as np
from typing import Tuple, Union
from bogoliubov_transform import Bogoliubov_transform_2
from Hamiltonian import Ham

def calculate_spectral(kpath_idx: np.ndarray, omega_idx: np.ndarray, eta: float, 
                      channel: np.ndarray, A1: complex, A2: complex, A3: complex, 
                      B1: float, B2: float, B3: float, lambda_param: float, h: float,
                      k1: np.ndarray, k2: np.ndarray, Q1: float, Q2: float, Nsites: int, beta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算光谱函数
    
    Parameters:
    -----------
    kpath_idx : ndarray
        k路径索引，形状 (kpath_len, 2)
    omega_idx : ndarray
        频率数组
    eta : float
        展宽参数
    channel : ndarray
        通道矩阵 (3x3)，指定计算哪些自旋分量
    A1, A2, A3 : complex
        鞍点参数A
    B1, B2, B3 : float
        鞍点参数B
    lambda_param : float
        拉格朗日乘数
    h : float
        对称破缺场
    k1, k2 : ndarray
        动量网格 (全布里渊区)
    Q1, Q2 : float
        磁序波矢
    Nsites : int
        格点数
        
    Returns:
    --------
    tuple
        (spec, chi_r, chi_i, spt) - 光谱函数，磁化率实部/虚部，单粒子谱
    """
    kpath_len = len(kpath_idx)
    omega_len = len(omega_idx)
    
    # 初始化结果数组
    spec = np.zeros((kpath_len, omega_len))
    spt = np.zeros((kpath_len, omega_len))
    chi_r = np.zeros((kpath_len, omega_len))
    chi_i = np.zeros((kpath_len, omega_len))
    
    # 定义自旋算符 (Pauli矩阵的扩展)
    u = np.zeros((3, 4, 4), dtype=complex)
    
    # S^z算符
    uz = 0.5 * np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]], dtype=complex)
    
    # S^x算符
    ux = 0.5 * np.array([[0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0]], dtype=complex)
    
    # S^y算符
    uy = 0.5 * np.array([[0, 0, 1j, 0],
                        [0, 0, 0, 1j],
                        [-1j, 0, 0, 0],
                        [0, -1j, 0, 0]], dtype=complex)
    
    u[0, :, :] = ux
    u[1, :, :] = uy
    u[2, :, :] = uz
    
    # 度规矩阵
    g = np.eye(4)
    g[1, 1] = -1
    g[3, 3] = -1

    order_vec = np.array([[[Q1, Q2],[0, 0],[0, 0]],
                          [[0, 0],[Q1, Q2],[0, 0]],
                          [[0, 0],[0, 0],[0, 0]]], dtype=np.complex128)
    dw=omega_idx[1]-omega_idx[0]
    
    U_k, eng_k, _, _, _, _ = Bogoliubov_transform_2(
        0, k1, k2 , Q1, Q2, 
        A1, A2, A3, B1, B2, B3, lambda_param, h)
    
    Nsites = len(k1)
    print(f"Bogoliubov transform check: Number of sites = {Nsites}")
        
    # 对每个自旋分量对进行循环
    for mu in range(3):
        for nu in range(3):
            if channel[mu, nu] == 1:
                print(f"Computing channel ({mu}, {nu})")                
                # 对每个k点进行循环
                for k_idx in range(kpath_len):
                    kx = kpath_idx[k_idx, 0]
                    ky = kpath_idx[k_idx, 1] 
                    static_temp=0
                    # 对每个频-率进行循环
                    for w_idx in range(omega_len):
                        omega = omega_idx[w_idx]
                        kq1 = k1+kx + order_vec[mu, nu, 0]
                        kq2 = k2+ky + order_vec[mu, nu, 1]
                        
                        # 计算Bogoliubov变换
                        U_kq, eng_kq, _, _, _, _ = Bogoliubov_transform_2(
                            0, kq1, kq2, Q1, Q2, 
                            A1, A2, A3, B1, B2, B3, lambda_param, h)
                        
                        temp_0=0  
                        temp_2=0                      
                        for idx in range(Nsites):
                            uu_kq = U_kq[idx, :, :]
                            uu_k = U_k[idx, :, :]
                            uu_kq_d = uu_kq.conj().T
                            uu_k_d = uu_k.conj().T

                            se_kq = eng_kq[idx, :] @ g
                            se_k = eng_k[idx, :] @ g
                            
                            # 构建格林函数
                            for m in range(15):
                                x_kq = m % 4
                                x_k = (m // 4) % 4

                                T_kq = np.outer(uu_kq[:, x_kq], uu_kq_d[x_kq, :])*g[x_kq, x_kq]
                                T_k = np.outer(uu_k[:, x_k], uu_k_d[x_k, :])*g[x_k, x_k]

                                f=+nb(beta*se_kq[x_kq])-nb(beta*se_k[x_k])
                                f1=omega + 1j*eta + se_kq[x_kq]-se_k[x_k]
                                f=f/f1
                                
                                # temp_0=temp_0 + f
                                temp_0=temp_0 + f*np.trace(T_kq @ u[mu, :, :] @ T_k @ u[nu, :, :])
                        temp_0=temp_0/(Nsites)
                        temp_2=0              

                        # print(f"k-point ({kx}, {ky}), omega={omega}: temp_0={temp_0}")          
                        
                        # 累积结果
                        chi_r[k_idx, w_idx] += np.real(temp_0 + temp_2)
                        chi_i[k_idx, w_idx] += np.imag(temp_0 + temp_2)
                        spec[k_idx, w_idx] += -np.imag(temp_0 + temp_2) / np.pi
                        static_temp += -np.imag(temp_0 + temp_2) #/ np.pi#*dw
                        spt[k_idx, w_idx] += np.imag(temp_0)
                    
                    print(f"{k_idx/kpath_len},kx={kx}, ky={ky}")
    
    return spec, chi_r, chi_i, spt

def nb(x: float) -> float:
    return 1/(np.exp(x)-1)

def calculate_spectral_simple(kx: float, ky: float, omega_array: np.ndarray, eta: float,
                            A1: complex, A2: complex, A3: complex, 
                            B1: float, B2: float, B3: float, 
                            lambda_param: float, h: float, 
                            Q1: float, Q2: float, beta: float) -> np.ndarray:
    """
    计算单个k点的光谱函数 (简化版本)
    
    Parameters:
    -----------
    kx, ky : float
        动量分量
    omega_array : ndarray
        频率数组
    eta : float
        展宽参数
    A1, A2, A3 : complex
        鞍点参数A
    B1, B2, B3 : float
        鞍点参数B
    lambda_param : float
        拉格朗日乘数
    h : float
        对称破缺场
    Q1, Q2 : float
        磁序波矢
        
    Returns:
    --------
    ndarray
        光谱函数
    """
    spectrum = np.zeros(len(omega_array))
    
    for i, omega in enumerate(omega_array):
        # 计算动态磁化率
        chi = calculate_dynamic_susceptibility(kx, ky, omega, eta, A1, A2, A3, 
                                             B1, B2, B3, lambda_param, h, Q1, Q2, beta)
        
        # 谱函数 = -Im[χ]/π
        spectrum[i] = -np.imag(chi) / np.pi
    
    return spectrum

def calculate_dynamic_susceptibility(kx: float, ky: float, omega: float, eta: float,
                                   A1: complex, A2: complex, A3: complex, 
                                   B1: float, B2: float, B3: float, 
                                   lambda_param: float, h: float, 
                                   Q1: float, Q2: float) -> complex:
    """
    计算单个(k,ω)点的动态磁化率
    
    Parameters:
    -----------
    kx, ky : float
        动量分量
    omega : float
        频率
    eta : float
        展宽参数
    A1, A2, A3 : complex
        鞍点参数A
    B1, B2, B3 : float
        鞍点参数B
    lambda_param : float
        拉格朗日乘数
    h : float
        对称破缺场
    Q1, Q2 : float
        磁序波矢
        
    Returns:
    --------
    complex
        动态磁化率
    """
    # 这里实现简化的动态磁化率计算
    # 使用解析延拓: iω → ω + iη
    iw = omega + 1j * eta
    
    # 计算单粒子格林函数
    U_k, eng_k, _, _, _, _ = Bogoliubov_transform_2(
        0, np.array([kx]), np.array([ky]), Q1, Q2, 
        A1, A2, A3, B1, B2, B3, lambda_param, h)
    
    # 构建格林函数
    g_matrix = np.eye(4)
    g_matrix[1, 1] = -1
    g_matrix[3, 3] = -1
    
    # 简化的磁化率计算
    chi = 0.0
    
    for i in range(4):
        for j in range(4):
            denominator = iw - eng_k[0, i]
            if abs(denominator) > 1e-12:
                chi += U_k[0, i, j] * np.conj(U_k[0, i, j]) / denominator
    
    return chi

def build_k_path(high_symmetry_points: list, labels: list, n_points: int) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    构建高对称点路径
    
    Parameters:
    -----------
    high_symmetry_points : list
        高对称点列表，每个元素是[kx, ky]
    labels : list
        高对称点标签
    n_points : int
        每段路径的点数
        
    Returns:
    --------
    tuple
        (k_path, k_distances, k_tick_positions) - k路径，距离，标记位置
    """
    n_segments = len(high_symmetry_points) - 1
    k_path = []
    k_distances = []
    k_tick_positions = []
    
    distance = 0.0
    
    for i in range(n_segments):
        start_point = np.array(high_symmetry_points[i])
        end_point = np.array(high_symmetry_points[i + 1])
        
        if i == 0:
            k_tick_positions.append(distance)
        
        # 生成路径点
        for j in range(n_points):
            if i == n_segments - 1 and j == n_points - 1:
                # 最后一个点
                point = end_point
            else:
                t = j / n_points
                point = (1 - t) * start_point + t * end_point
            
            k_path.append(point)
            k_distances.append(distance)
            
            if j < n_points - 1:
                # 计算到下一个点的距离
                next_t = (j + 1) / n_points
                next_point = (1 - next_t) * start_point + next_t * end_point
                distance += np.linalg.norm(next_point - point)

            # print(f"Segment {i}, Point {j}: k = {point}, Distance = {distance}")
        
        # 添加段结束点的标记位置
        k_tick_positions.append(distance)
    
    return np.array(k_path), np.array(k_distances), k_tick_positions

def get_triangular_lattice_path(L1: int = 10) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    获取三角格子的标准高对称点路径
    
    Returns:
    --------
    tuple
        (k_path, k_distances, k_tick_positions, labels) - 标准三角格子路径
    """
    # 三角格子的高对称点
    high_symmetry_points = [
        [0.5, -0.25],    # X
        [0.5, 0],        # M  
        [0, 0],          # Γ
        [2/3, 1/3],      # K'
        [1/2, 1/2],      # M
        [0, 0.25],       # Y
        [1/3, 2/3]       # K
    ]
    
    labels = ['X', 'M', 'Γ', "K'", 'M', 'Y', 'K']
    
    return build_k_path(high_symmetry_points, labels, L1*L1)

def convert_to_cartesian_coordinates(k_path: np.ndarray) -> np.ndarray:
    """
    将相对坐标转换为笛卡尔坐标
    
    Parameters:
    -----------
    k_path : ndarray
        相对坐标中的k路径
        
    Returns:
    --------
    ndarray
        笛卡尔坐标中的k路径
    """
    # 三角格子的倒格矢
    a1 = np.array([1, 0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    b1 = np.array([2*np.pi, -2*np.pi/np.sqrt(3)])
    b2 = np.array([0, 4*np.pi/np.sqrt(3)])
    
    # 转换矩阵
    b_matrix = np.array([b1, b2])
    a_matrix = np.array([a1, a2])
    t_matrix = b_matrix @ a_matrix.T
    
    k_cartesian = np.zeros_like(k_path)
    for i, k_point in enumerate(k_path):
        k_cartesian[i] = k_point @ t_matrix
        # print(f"Converting k-point {k_point} to Cartesian: {k_cartesian[i]}")
    
    return k_cartesian