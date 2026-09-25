"""
Bogoliubov变换模块 - JAX版本（支持GPU加速）

Author: ZhouChk
JAX优化: 批量矩阵运算，JIT编译，GPU/TPU加速
"""

import jax
import jax.numpy as jnp
from typing import Tuple
from functools import partial
from Hamiltonian_jax import Ham_jax

# 启用64位精度以匹配NumPy
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnums=(1,))
def bogoliubov_single_k(H_k: jnp.ndarray, spin_n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    对单个k点进行Bogoliubov变换 - JAX版本
    
    完全遵循NumPy版本的实现逻辑
    
    Parameters:
    -----------
    H_k : jnp.ndarray
        单个k点的哈密顿量矩阵 (4*spin_n, 4*spin_n)
        
    Returns:
    --------
    Ubov : jnp.ndarray
        Bogoliubov变换矩阵 (4*spin_n, 4*spin_n)
    ek : jnp.ndarray
        能谱 (4*spin_n,)
    """
    # 度规矩阵
    spin_n = 1
    mat_dim = 4 * spin_n

    g = jnp.eye(mat_dim, dtype=jnp.float64)

    def make_g(n, carry):
        # jax.debug.print("设置度规矩阵 g 的元素 g[{n},{n}] = -1", n=n, ordered=True)
        return carry.at[n, n].set(-1)
    g = jax.lax.fori_loop(2*spin_n, 4*spin_n, make_g, g)
    
    # Cholesky分解 (upper triangular, 与NumPy版本一致)
    entest=jnp.linalg.eigvals(H_k)
    # jax.debug.print("输入矩阵H_k的特征值: {entest}", entest=entest)
    
    r_lower = jnp.linalg.cholesky(H_k)  # JAX默认返回下三角
    r = r_lower.T.conj()  # 转置得到上三角
    
    # 构造变换后的矩阵，并做一次数值厄米化
    ht = r @ g @ r.T.conj()
    # ht = 0.5 * (ht + ht.T.conj())

    # 对厄米矩阵使用eigh，返回正交归一特征向量
    enk_vals, ut = jnp.linalg.eigh(ht)
    sort_idx = jnp.argsort(enk_vals.real)[::-1]
    sort_enk = enk_vals.real[sort_idx]
    
    un = ut[:, sort_idx]

    def compute_eigenval(v):
        return v.conj().T @ ht @ v

    # jax.debug.print("排序后的特征值: {sort_enk}", sort_enk=sort_enk,ordered=True)
    ekk_diag = jax.vmap(compute_eigenval, in_axes=1)(un)
    # jax.debug.print("逐列验证特征值 v^H H v: {ekk_diag}", ekk_diag=ekk_diag, ordered=True)
    # jax.debug.print("验证误差 (v^H H v - sort_enk): {delta}", delta=(ekk_diag.real - sort_enk), ordered=True)

    # 计算最终的Bogoliubov矩阵
    ekk = jnp.diag(ekk_diag)
    ekk_sq = jnp.diag(jnp.sqrt(jnp.diag(g @ ekk)))  # sqrt(g @ ekk)
    r_inv = jnp.linalg.inv(r)
    Ubov = r_inv @ un @ ekk_sq
    
    # 提取最终能谱
    ek = jnp.array([g[j, j] * ekk_diag[j] for j in range(mat_dim)]).real
    
    return Ubov, ek

@partial(jax.jit, static_argnums=(15, 16))
def Bogoliubov_transform_jax_batch(omega: float, k1: jnp.ndarray, k2: jnp.ndarray,
                                    A1: complex, A2: complex, A3: complex, 
                                    B1: float, B2: float, B3: float, 
                                    lambda_param: float, h: float,
                                    J1plus: float, J2plus: float, J3plus: float,
                                    bond_tab: jnp.ndarray, spin_n: int, bond_n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    批量Bogoliubov变换 - JAX版本（完全向量化，GPU加速）
    
    Parameters:
    -----------
    omega : float
        频率参数
    k1, k2 : jnp.ndarray
        动量分量 (nk,)
    A1, A2, A3 : complex
        鞍点参数A
    B1, B2, B3 : float
        鞍点参数B
    lambda_param : float
        拉格朗日乘数
    h : float
        对称破缺场
    J1plus, J2plus, J3plus : float
        交换耦合参数
        
    Returns:
    --------
    Ubov : jnp.ndarray
        Bogoliubov变换矩阵 (nk, 4, 4)
    ek : jnp.ndarray
        能谱 (nk, 4)
    """
    # 构建哈密顿量（批量）
    H = Ham_jax(omega, k1, k2, A1, A2, A3, B1, B2, B3, 
                lambda_param, h, J1plus, J2plus, J3plus, bond_tab, spin_n, bond_n)
    
    # spin_n 参与矩阵维度构造，必须作为静态参数显式传入单点变换。
    Ubov, ek = jax.vmap(lambda H_k: bogoliubov_single_k(H_k, spin_n))(H)
    
    return Ubov, ek

def Bogoliubov_transform_2_jax(omega: float, k1: jnp.ndarray, k2: jnp.ndarray,
                               A1: complex, A2: complex, A3: complex, 
                               B1: float, B2: float, B3: float, 
                               lambda_param: float, h: float,
                               J1plus: float, J2plus: float, J3plus: float,
                               bond_tab: jnp.ndarray, spin_n: int, bond_n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Bogoliubov变换 - JAX版本（包装函数，兼容原接口）
    
    Returns:
    --------
    tuple
        (Ubov, ek) - Bogoliubov矩阵，能谱
    """
    # 确保输入是JAX数组
    k1 = jnp.atleast_1d(k1)
    k2 = jnp.atleast_1d(k2)
    
    # 执行批量计算
    spin_n = 1
    Ubov, ek = Bogoliubov_transform_jax_batch(
        omega, k1, k2, A1, A2, A3, B1, B2, B3, 
        lambda_param, h, J1plus, J2plus, J3plus, bond_tab, spin_n, bond_n
    )
    
    return Ubov, ek


@partial(jax.jit, static_argnums=(15, 16))
def Bogoliubov_constraint_jax_batch(omega: float, k1: jnp.ndarray, k2: jnp.ndarray, 
                                    A1: complex, A2: complex, A3: complex, 
                                    B1: float, B2: float, B3: float, 
                                    lambda_param: float, h: float,
                                    J1plus: float, J2plus: float, J3plus: float, 
                                    bond_tab: jnp.ndarray, spin_n: int, bond_n: int) -> jnp.ndarray:
    
    # 构建哈密顿量（批量）
    H = Ham_jax(omega, k1, k2, A1, A2, A3, B1, B2, B3, 
                lambda_param, h, J1plus, J2plus, J3plus, bond_tab, spin_n, bond_n)
    # jax.debug.print("lambda_param: {}", lambda_param, ordered=True)
    # jax.debug.print("H shape: {}", H.shape, ordered=True)
    # jax.debug.print("H[0]:\n{}", H[0], ordered=True)

    # 使用vmap进行向量化处理所有k点
    min_eng = jax.vmap(get_min_energy_jax)(H)
    
    return min_eng

def Bogoliubov_constraint_jax(omega: float, k1: jnp.ndarray, k2: jnp.ndarray,
                               A1: complex, A2: complex, A3: complex, 
                               B1: float, B2: float, B3: float, 
                               lambda_param: float, h: float,
                               J1plus: float, J2plus: float, J3plus: float,
                               bond_tab: jnp.ndarray, spin_n: int, bond_n: int) -> float:
    """
    Bogoliubov变换 - JAX版本（包装函数，兼容原接口）
    
    Returns:
    --------
    float
        所有 k 点最小本征值中的最小值
    """
    # 确保输入是JAX数组
    k1 = jnp.atleast_1d(k1)
    k2 = jnp.atleast_1d(k2)
    
    # 执行批量计算
    min_eng = Bogoliubov_constraint_jax_batch(
        omega, k1, k2, A1, A2, A3, B1, B2, B3, 
        lambda_param, h, J1plus, J2plus, J3plus, bond_tab, spin_n, bond_n
    )

    min_energy = jnp.min(min_eng)
    
    return min_energy

@partial(jax.jit, static_argnums=())
def get_min_energy_jax(H_k: jnp.ndarray) -> jnp.ndarray:
    
    # 计算特征值和特征向量 (使用eig以匹配NumPy版本)
    enk_vals, _ = jnp.linalg.eig(H_k)
    min_energies = jnp.min(enk_vals.real)
    # jax.debug.print("min_energies: {}", min_energies, ordered=True)

    return min_energies


@partial(jax.jit, static_argnums=(7, 8))
def saddle_point_sum_jax(Ubov: jnp.ndarray, k1: jnp.ndarray, k2: jnp.ndarray,
                         J1plus: float, J2plus: float, J3plus: float,
                         bond_tab: jnp.ndarray, spin_n: int, bond_n: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    计算鞍点求和 - JAX版本（完全向量化，与NumPy版本一致）
    
    Parameters:
    -----------
    Ubov : jnp.ndarray
        Bogoliubov变换矩阵 (nk, 4, 4)
    k1, k2 : jnp.ndarray
        动量分量 (nk,)
    Q1, Q2 : float
        磁序波矢
        
    Returns:
    --------
    tuple
        (lambda, AA, BB, Usum) - 各种求和结果
    """
    spin_n = 1

    mat_dim = 4 * spin_n # 哈密顿量矩阵的维度
    nk = Ubov.shape[0]
    bond_n_eff = min(bond_n, bond_tab.shape[0])
    Jplus = jnp.array([J1plus, J2plus, J3plus], dtype=jnp.float64)
    #Jplus = jnp.array([J1plus, 0, 0], dtype=jnp.float64)

    from Hamiltonian_jax import compute_diag_term, bond_body

    B_I = jnp.array([1, 0, 0], dtype=jnp.float64)
    A_I = jnp.array([1, 0, 0], dtype=jnp.float64)
    I_0 = jnp.array([0, 0, 0], dtype=jnp.float64)

    # 1. 预计算 lam_template (k无关，只算一次)
    lam_template = jnp.zeros((1, mat_dim, mat_dim), dtype=jnp.complex128)
    lam_template = jax.lax.fori_loop(0, spin_n,
        lambda i, lam_cur: compute_diag_term(i, lam_cur, 1, 1, 0, spin_n, mat_dim),
        lam_template)
    lam_template = lam_template[0]  # (mat_dim, mat_dim)

    # 2. 预计算所有 k 点的 cc_cur 和 ss_cur
    def compute_k_hamiltonians(k1_i, k2_i):
        cc_init = jnp.zeros((1, mat_dim, mat_dim), dtype=jnp.complex128)
        ss_init = jnp.zeros((1, mat_dim, mat_dim), dtype=jnp.complex128)
        cc_cur = jax.lax.fori_loop(0, bond_n_eff,
            lambda b, c: bond_body(b, c, B_I, I_0, bond_tab, spin_n, mat_dim, k1_i, k2_i), cc_init)
        ss_cur = jax.lax.fori_loop(0, bond_n_eff,
            lambda b, s: bond_body(b, s, I_0, A_I, bond_tab, spin_n, mat_dim, k1_i, k2_i), ss_init)
        return cc_cur[0], ss_cur[0]  # (mat_dim, mat_dim) each

    cc_cur_all, ss_cur_all = jax.vmap(compute_k_hamiltonians)(k1, k2)
    # cc_cur_all: (nk, mat_dim, mat_dim)
    # ss_cur_all: (nk, mat_dim, mat_dim)

    # 3. 每个 k 点只做矩阵乘法
    def compute_single_k_contribution(ut, cc_cur_k, ss_cur_k):
        ut_uth = ut @ jnp.conj(ut).T  # (mat_dim, mat_dim)
        combined_0 = ut_uth @ lam_template - lam_template
        combined_1 = ut_uth @ cc_cur_k
        combined_2 = ut_uth @ ss_cur_k
        # jax.debug.print("lam_template: {}", lam_template, ordered=True)
        # jax.debug.print("cc_cur_k: {}", cc_cur_k, ordered=True)
        # jax.debug.print("ss_cur_k: {}", ss_cur_k, ordered=True)
        return jnp.stack([combined_0, combined_1, combined_2], axis=0)  # (3, mat_dim, mat_dim)

    all_contributions = jax.vmap(compute_single_k_contribution)(Ubov, cc_cur_all, ss_cur_all)
    # all_contributions: (nk, 3, mat_dim, mat_dim)

    # 对所有k点求和并归一化
    Usum = jnp.sum(all_contributions, axis=0) / nk

    lam = jnp.trace(Usum[0, :, :]) * 0.25
    AA = jnp.trace(Usum[2, :, :]) * 0.125
    BB = jnp.trace(Usum[1, :, :]) * 0.125

    return jnp.real(lam), AA, BB, Usum
