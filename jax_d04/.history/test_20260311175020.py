import IO
from Hamiltonian_jax import Ham_jax

# 模型参数
J1xy = J2xy = J3xy = 1.0
    J1z = J2z = J3z = 1.0  # 各向同性
    S = 0.5
    
    # 计算组合参数
    J1plus = (J1z + J1xy) / 2
    J2plus = (J2z + J2xy) / 2
    J3plus = (J3z + J3xy) / 2
    
    # 晶格参数
    L2 = L1
    k1_1d = 2*np.pi/L1 * np.arange(L1)
    k2_1d = 2*np.pi/L2 * np.arange(L2)
    k1_2d, k2_2d = np.meshgrid(k1_1d, k2_1d, indexing='ij')
    k1 = k1_2d.flatten()
    k2 = k2_2d.flatten()
    Nsites = len(k1)
    h = 1.0 / Nsites
    
    # 读取晶格信息
    [bond_tab, n_bond] = create_bond_table_file(unit_vector_filepath='unit_vector.in', cellspin_filepath='cellspin.in')
