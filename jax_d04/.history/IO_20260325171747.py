
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Tuple, Optional

def read_unit_vectors(filepath: str = 'unit_vector.in') -> Tuple[int, np.ndarray]:
    
    """
    读取单位矢量文件
    
    Parameters:
    -----------
    filepath : str
        单位矢量文件路径（默认为 'unit_vector.in'）
        
    Returns:
    --------
    tuple
        (dimension, unit_vectors)
        - dimension: int，矢量维度
        - unit_vectors: np.ndarray，形状为 (n_vectors, dimension) 的基础矢量数组
        
    Examples:
    ---------
    >>> dim, vectors = read_unit_vectors('unit_vector.in')
    >>> print(f"维度: {dim}")
    >>> print(f"基础矢量:\n{vectors}")
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")
    
    vectors_data = []
    dimension = None
    
    with open(filepath, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            
            # 第一行（非注释）为维度
            if dimension is None:
                dimension = int(line)
                continue
            
            # 后续行为基础矢量
            # 处理包含 sqrt、分数等数学表达式的字符串
            components = line.split()
            vector = []
            for comp in components:
                # 使用 numpy 和 eval 来处理数学表达式（sqrt、分数等）
                comp = comp.replace('sqrt', 'np.sqrt')
                comp = comp.replace('^', '**')  # 处理 ^ 指数符号
                try:
                    value = eval(comp, {'np': np, '__builtins__': {}})
                    vector.append(float(value))
                except Exception as e:
                    raise ValueError(f"无法解析第 {line_idx+1} 行的表达式 '{comp}': {str(e)}")
            
            if len(vector) != dimension:
                raise ValueError(f"第 {line_idx+1} 行的矢量维度 {len(vector)} 不匹配预声明维度 {dimension}")
            vectors_data.append(vector)
    
    if dimension is None:
        raise ValueError("文件中没有找到维度信息")
    
    if not vectors_data:
        raise ValueError("文件中没有找到任何基础矢量")
    
    unit_vectors = np.array(vectors_data, dtype=float)
    
    print(f"已读取 {filepath}:")
    print(f"  维度: {dimension}")
    print(f"  基础矢量数量: {len(vectors_data)}")
    
    return dimension, unit_vectors

def read_spin_in_cell(dim: int, filepath: str = 'cellspin.in') -> Tuple[np.ndarray, int, np.ndarray]:
    """
    读取自旋晶胞信息。

    文件格式（忽略注释行和空行后）：
    1) 前 dim 行: 自旋 unit cell 的平移矢量
    2) 第 dim+1 行: 自旋数目
    3) 第 dim+2 行开始: 每行一个自旋坐标

    Parameters:
    -----------
    dim : int
        空间维度
    filepath : str
        输入文件路径（默认 'cellspin.in'）

    Returns:
    --------
    tuple
        (spin_cell_vectors, spin_positions)
        - spin_cell_vectors: np.ndarray, 形状 (dim, dim)
        - spin_positions: np.ndarray, 形状 (n_spin, dim)
    """
    if dim <= 0:
        raise ValueError(f"dim 必须为正整数，当前值为 {dim}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    def _parse_vector(line: str, line_number: int) -> list:
        comps = line.split()
        values = []
        for comp in comps:
            expr = comp.replace('sqrt', 'np.sqrt').replace('^', '**')
            try:
                values.append(float(eval(expr, {'np': np, '__builtins__': {}})))
            except Exception as e:
                raise ValueError(f"无法解析第 {line_number} 行表达式 '{comp}': {str(e)}")
        return values

    content_lines = []
    with open(filepath, 'r') as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith('#'):
                continue
            content_lines.append(stripped)

    min_required_lines = dim + 1
    if len(content_lines) < min_required_lines:
        raise ValueError(
            f"文件有效行数不足，至少需要 {min_required_lines} 行（{dim} 行平移矢量 + 1 行自旋数目）"
        )

    spin_cell_vectors_data = []
    for i in range(dim):
        vector = _parse_vector(content_lines[i], i + 1)
        if len(vector) != dim:
            raise ValueError(f"第 {i+1} 行平移矢量维度为 {len(vector)}，应为 {dim}")
        spin_cell_vectors_data.append(vector)

    try:
        n_spin = int(content_lines[dim])
    except Exception as e:
        raise ValueError(f"第 {dim+1} 行自旋数目无法解析为整数: {str(e)}")

    if n_spin <= 0:
        raise ValueError(f"自旋数目必须为正整数，当前值为 {n_spin}")

    coordinate_lines = content_lines[dim + 1:]
    if len(coordinate_lines) < n_spin:
        raise ValueError(f"自旋坐标行数不足，需要 {n_spin} 行，实际 {len(coordinate_lines)} 行")

    spin_positions_data = []
    for i in range(n_spin):
        vector = _parse_vector(coordinate_lines[i], dim + 2 + i)
        if len(vector) != dim:
            raise ValueError(f"第 {dim+2+i} 行自旋坐标维度为 {len(vector)}，应为 {dim}")
        spin_positions_data.append(vector)

    spin_cell_vectors = np.array(spin_cell_vectors_data, dtype=float)
    spin_positions_fractional = np.array(spin_positions_data, dtype=float)
    
    # 将分数坐标转换为实空间坐标: r = x_1*a_1 + x_2*a_2 + ...
    # spin_positions = spin_positions_fractional @ spin_cell_vectors
    spin_positions = spin_positions_fractional @ spin_cell_vectors

    print(f"已读取 {filepath}:")
    print(f"  维度: {dim}")
    print(f"  自旋晶胞平移矢量数: {len(spin_cell_vectors_data)}")
    print(f"  自旋数目: {n_spin}")

    return spin_cell_vectors, n_spin, spin_positions

def gen_bond_table(spin_positions: np.ndarray, unit_vectors: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    生成自旋键表。

    Parameters:
    -----------
    dim : int
        空间维度
    filepath : str
        输入文件路径（默认 'cellspin.in')

    Returns:
    --------
    tuple
        (spin_cell_vectors, spin_positions)
        - spin_cell_vectors: np.ndarray, 形状 (dim, dim)
        - spin_positions: np.ndarray, 形状 (n_spin, dim)
    """
    extand_spin_table = []
    n_spin = len(spin_positions)
    tran_table = [[0,0], [0,1], [1,0], [1,-1], [0,-1], [-1,0], [-1,1]]
    tran_table = np.array(tran_table) @ unit_vectors
    print(f"生成平移表，共 {len(tran_table)} 个平移向量:")
    for i, tran in enumerate(tran_table):
        print(f"  平移 {i}: {tran}")

    for i in range(len(tran_table)):
        for j in range(len(spin_positions)):
            extand_spin_table.append(spin_positions[j]+tran_table[i])
            #print(f"添加扩展自旋位置: {spin_positions[j]} + {tran_table[i]} = {spin_positions[j]+tran_table[i]}")

    print(f"生成扩展自旋位置总数: {len(extand_spin_table)}")

    print(f"单位矢量矩阵:\n{unit_vectors}")
    if unit_vectors.ndim == 2 and unit_vectors.shape[0] == unit_vectors.shape[1]:
        inv_unit_vectors = np.linalg.inv(unit_vectors)
        print(f"单位矢量逆矩阵:\n{inv_unit_vectors}")
        print(f"验证单位矢量与逆矩阵的乘积:\n{unit_vectors @ inv_unit_vectors}")
    else:
        print("单位矢量不是方阵，无法计算逆矩阵")
    print(f"自旋位置矩阵:\n{[1,0]@unit_vectors}")

    bond_table=[]
    n_bond=0
    for i in range(len(spin_positions)):
        for j in range(len(extand_spin_table)):
            if i == -j:
                continue
            else:
                dist = np.linalg.norm(spin_positions[i] - extand_spin_table[j])
                if (dist-1)**2 < 1e-5:  # 可以根据实际情况调整距离阈值
                    print(f"找到键: 自旋 {i} 与扩展位置 {j} 距离 {dist:.2e}")
                    site_i = i % len(spin_positions)
                    site_j = j % len(spin_positions)
                    if (site_i == 0 and site_j == 1) or (site_i == 1 and site_j == 0):
                        type = 0
                        rij=spin_positions[i] - extand_spin_table[j]
                        bond_table.append((i, j, site_i, site_j, rij, rij@inv_unit_vectors, type))
                    elif (site_i == 1 and site_j == 2) or (site_i == 2 and site_j == 1):
                        type = 1
                        rij=spin_positions[i] - extand_spin_table[j]
                        bond_table.append((i, j, site_i, site_j, rij, rij@inv_unit_vectors, type))
                    elif (site_i == 0 and site_j == 2) or (site_i == 2 and site_j == 0):
                        type = 2
                        rij=spin_positions[i] - extand_spin_table[j]
                        bond_table.append((i, j, site_i, site_j, rij, rij@inv_unit_vectors, type))
                    n_bond += 1
                    print(f"  键向量: {spin_positions[i] - extand_spin_table[j]}")

    print(f"总共找到 {n_bond} 个键")

    with open('bond.log', 'w') as f:
        f.write(f"{n_spin}\n")
        f.write(f"{n_bond}\n")
        for bond in bond_table:
            f.write(f"{bond}\n")
    print("bond_table 已写入 bond.log")

    return bond_table, n_spin, n_bond

def read_bond_table(filepath: str = 'bond.log') -> Tuple[np.ndarray, int, int]:
    """
    从文件读取自旋键表。

    Parameters:
    -----------
    filepath : str
        键表文件路径（默认 'bond.log'）

    Returns:
    --------
    tuple
        (bond_table, n_spin, n_bond)
        - bond_table: np.ndarray，包含键信息
        - n_spin: int，自旋数目
        - n_bond: int，键的数量
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if not lines:
        raise ValueError("文件内容为空")

    try:
        n_spin = int(lines[0].strip())
        n_bond = int(lines[1].strip())
    except Exception as e:
        raise ValueError(f"无法解析前两行的数量: {str(e)}")

    bond_table_data = []
    for line_idx, line in enumerate(lines[2:], start=3):
        line = line.strip()
        if not line:
            continue
        try:
            # 仅开放解析 bond.log 所需的最小命名空间（如 array([...])）
            safe_globals = {'np': np, 'array': np.array, '__builtins__': {}}
            bond_info = eval(line, safe_globals, {})  # 注意：eval 有安全风险，确保输入文件可信
            if len(bond_info) not in (6, 7):
                raise ValueError(
                    f"第 {line_idx} 行的键信息长度既不是 6 也不是 7: {bond_info}"
                )
            bond_table_data.append(bond_info)
        except Exception as e:
            raise ValueError(f"无法解析第 {line_idx} 行的键信息 '{line}': {str(e)}")

    if len(bond_table_data) != n_bond:
        raise ValueError(f"实际读取的键数量 {len(bond_table_data)} 与声明的数量 {n_bond} 不匹配")

    bond_table = np.array(bond_table_data, dtype=object)  # 使用 object 类型以保持原始数据结构

    print(f"已从 {filepath} 读取 {n_bond} 个键")

    return bond_table, n_spin, n_bond

def create_bond_table_file(unit_vector_filepath: str = 'unit_vector.in', cellspin_filepath: str = 'cellspin.in'):
    """
    创建自旋键表文件。
    """
    [dim, unit_vectors] = read_unit_vectors(filepath=unit_vector_filepath)
    print(f"维度: {dim}")
    print("基础矢量:")
    print(unit_vectors)

    [spin_dim, spin_num, cell_spins] = read_spin_in_cell(dim=dim, filepath=cellspin_filepath)
    print(f"自旋晶胞维度: {spin_dim}")
    print(f"自旋数目: {spin_num}")
    print("自旋晶胞平移矢量:")
    print(cell_spins)

    [bond_table, n_spin, n_bond] = gen_bond_table(spin_positions=cell_spins, unit_vectors=unit_vectors)
    return bond_table, n_spin, n_bond

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