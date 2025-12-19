"""
Bogoliubov变换模块

Author: ZhouChk
"""

import numpy as np
import scipy.linalg as la
from typing import Union, Tuple
from gamma_functions import (gamma_A, gamma_B, set_global_params)
from Hamiltonian import Ham

def Bogoliubov_transform_2(omega: float, k1: Union[float, np.ndarray], k2: Union[float, np.ndarray], 
                         Q1: float, Q2: float, 
                         A1: complex, A2: complex, A3: complex, 
                         B1: complex, B2: complex, B3: complex, 
                         lambda_param: float, h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    改进的Bogoliubov变换 - 计算Bogoliubov矩阵和能谱    
    
    Parameters:
    -----------
    omega : float
        频率参数
    k1, k2 : float or ndarray
        动量分量
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
    tuple
        (Ubov, ek, Enk, eg, g, H) - Bogoliubov矩阵，能谱，等
    """
    if lambda_param > 10**2:
        print(f"Warning: Lambda should not be that large: {lambda_param}")
    
    k1 = np.atleast_1d(k1)
    k2 = np.atleast_1d(k2)
    nk = len(k1)
    
    # 初始化数组
    Ubov = np.zeros((nk, 4, 4), dtype=complex)
    Enk = np.zeros((nk, 4, 4), dtype=complex)
    H = np.zeros((nk, 4, 4), dtype=complex)
    ek = np.zeros((nk, 4), dtype=float)
    eg = np.zeros((nk, 4), dtype=float)
    
    # 构建哈密顿量
    H = Ham(omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, lambda_param, h)
    
    # 度规矩阵
    g = np.eye(4)
    g[1, 1] = -1
    g[3, 3] = -1
    
    for i in range(nk):
        # 提取第i个k点的哈密顿量
        h1 = H[i, :, :]
        
        # 检查特征值是否为正
        eigenvals = np.linalg.eigvals(h1)
        eigenvals_sorted = np.sort(eigenvals)
        if eigenvals_sorted[0] < 0:
            print(f"Negative eigenvalue at k-point {i} ({k1[i]:.4f}, {k2[i]:.4f}): {eigenvals_sorted[0]:.6f}")
            print("Hamiltonian matrix:\n", h1)
            print("eigenvalues:\n", eigenvals_sorted)
            print("parameters: A1, A2, A3, B1, B2, B3, lambda_param, h =",
                  A1, A2, A3, B1, B2, B3, lambda_param, h)
            raise ValueError(f"Negative eigenvalue detected at k-point {i}")

        try:
            # Cholesky分解确保正定性
            r = la.cholesky(h1, lower=False)
            
            # 构造变换后的矩阵
            ht = r @ g @ r.T
            
            # 计算特征值和特征向量
            enk_vals, ut = la.eig(ht)
            sort_enk = np.sort(enk_vals)
            
            # 重新排序特征向量
            idxtab = [3, 1, 2, 0]  # Python从0开始索引
            un = np.zeros((4, 4), dtype=complex)
            ekk = np.zeros((4, 4), dtype=complex)
            indx = np.zeros(4, dtype=bool)
            
            for k in range(4):
                idx = idxtab[k]
                val = sort_enk[idx]
                ekk[k, k] = val
                
                # 找到对应的特征向量
                test = False
                for j in range(4):
                    eigenval_test = ut[:, j].conj().T @ ht @ ut[:, j]
                    if abs(eigenval_test - val) <= 1e-5 and not indx[j]:
                        un[:, k] = ut[:, j]
                        indx[j] = True
                        test = True
                        break
                
                if not test:
                    print(f"Warning: Failed to find eigenvector for k-point {i}")
            
            # 计算最终的Bogoliubov矩阵
            ekk_sq = la.sqrtm(g @ ekk)
            r_inv = np.linalg.inv(r)
            uv = r_inv @ un @ ekk_sq
            
            # 检查对易性
            unitarity_error = np.max(np.abs(uv.conj().T @ g @ uv - g))
            if unitarity_error > 1e-6:
                print(f"Warning: Bogoliubov transformation is not unitary at k-point {i}, error: {unitarity_error}")
            
            # 存储结果
            Ubov[i, :, :] = uv
            
            # 存储结果(real部分)
            for j in range(4):
                ek[i, j] = np.real(g[j, j] * ekk[j, j])
                if ek[i, j] < -0.01:
                    print(f"Negative eigenvalue at k-point {i} ({k1[i]:.4f}, {k2[i]:.4f}), mode {j}: {ek[i, j]:.6f}")
                Enk[i, j, j] = 1 / ek[i, j] if abs(ek[i, j]) > 1e-12 else 0
                eg[i, j] = np.real(eigenvals_sorted[j])
                
        except la.LinAlgError:
            print(f"Cholesky decomposition failed at k-point {i}")
            # 使用备选方法
            eigenvals, eigenvecs = la.eig(h1)
            eigenvals_sorted = np.sort(eigenvals)
            for j in range(4):
                ek[i, j] = eigenvals_sorted[j]
                eg[i, j] = eigenvals_sorted[j]
    
    return Ubov, ek, Enk, eg, g, H

def Bogoliubov_constraint(omega: float, k1: Union[float, np.ndarray], k2: Union[float, np.ndarray], 
                         Q1: float, Q2: float, 
                         A1: complex, A2: complex, A3: complex, 
                         B1: complex, B2: complex, B3: complex, 
                         lambda_param: float, h: float) -> float:
    """
    计算Bogoliubov约束条件 (所有特征值应为正)
    
    Parameters:
    -----------
    omega : float
        频率参数
    k1, k2 : float or ndarray
        动量分量
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
        每个k点的最小特征值
    """
    k1 = np.atleast_1d(k1)
    k2 = np.atleast_1d(k2)
    nk = len(k1)
    
    # 构建哈密顿量
    H = Ham(omega, k1, k2, Q1, Q2, A1, A2, A3, B1, B2, B3, lambda_param, h)
    
    eig_min = np.zeros(nk)
    
    for i in range(nk):
        h1 = H[i, :, :]
        # if i==0:
        #     print("Hamiltonian at first k-point:\n", h1)
        eigenvals = np.linalg.eigvals(h1)
        eig_min[i] = np.min(np.real(eigenvals))
        # if i==0:
        #     print(f"k-point {i} ({k1[i]:.4f}, {k2[i]:.4f}): min eigenvalue = {eigenvals}")
        #     print("omega:", omega, "A1:", A1, "B1:", B1, "lambda_param:", lambda_param, "h:", h)
        #     print("Gamma_A:", gamma_A(k1[i] + Q1/2, k2[i] + Q2/2, A1, A2, A3))
        #     print("Gamma_B:", gamma_B(k1[i] + Q1/2, k2[i] + Q2/2, B1, B2, B3))
        # if i==0 or i==133:
        #     print(f"k-point {i} ({k1[i]:.4f}, {k2[i]:.4f}): min eigenvalue = {eig_min[i]:.6f}")

    # print("************************************************")
    # print("Minimum eigenvalue across all k-points:", np.min(eig_min))
    # print("************************************************")
    return np.min(eig_min)

def saddle_point_sum(Ut: np.ndarray, k1: np.ndarray, k2: np.ndarray, 
                    Q1: float, Q2: float) -> Tuple[float, complex, complex, np.ndarray]:
    """
    计算鞍点方程中的求和项
    
    Parameters:
    -----------
    Ut : ndarray
        Bogoliubov变换矩阵 (Nsites, 4, 4)
    k1, k2 : ndarray
        动量网格
    Q1, Q2 : float
        磁序波矢
        
    Returns:
    --------
    tuple
        (lambda, AA, BB, Usum) - 各种求和结果
    """
    Nsites = len(k1)
    Usum = np.zeros((3, 4, 4), dtype=complex)
    
    for i in range(Nsites):
        ut = Ut[i, :, :]  # 第i个k点的4x4矩阵
        
        # 构建三角函数矩阵
        cc = np.array([[np.cos(k1[i] + Q1/2), 0, 0, 0],
                       [0, np.cos(k1[i] + Q1/2), 0, 0],
                       [0, 0, np.cos(-k1[i] + Q1/2), 0],
                       [0, 0, 0, np.cos(-k1[i] + Q1/2)]])
        
        ss = np.array([[0, np.sin(k1[i] + Q1/2), 0, 0],
                       [np.sin(k1[i] + Q1/2), 0, 0, 0],
                       [0, 0, 0, np.sin(-k1[i] + Q1/2)],
                       [0, 0, np.sin(-k1[i] + Q1/2), 0]])
        
        # 计算组合矩阵
        combined = np.zeros((3, 4, 4), dtype=complex)
        combined[0, :, :] = ut @ np.conj(ut).T @ np.eye(4) - np.eye(4)
        combined[1, :, :] = ut @ np.conj(ut).T @ cc
        combined[2, :, :] = ut @ np.conj(ut).T @ ss
        
        Usum += combined
    
    # 归一化
    Usum /= Nsites
    
    # 计算各个量
    lam = np.trace(Usum[0, :, :]) / 4
    AA = 1j * np.trace(Usum[2, :, :]) / 8
    BB = np.trace(Usum[1, :, :]) / 8
    # print(f"saddle point sum results: lambda={lam}, A={AA}, B={BB}")
    
    return float(np.real(lam)), AA, BB, Usum