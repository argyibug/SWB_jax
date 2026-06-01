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
L1 = 10
L2 = L1
k1 = 0.1
k2 = 0.1
Nsites = 10
h = 1.0 / Nsites

A1 = A2 = A3 = 0.49126303j
B1 = 0.22640955
B2 = -B1
B3 = B1
lambda_param = 0.94176189
    
# 读取晶格信息
[bond_tab, n_spin, n_bond] = IO.create_bond_table_file(unit_vector_filepath='unit_vector.in', cellspin_filepath='cellspin.in')
[bond_tab, n_spin, n_bond] = IO.read_bond_table(filepath='bond.log')
print(f"读取键表: 共 {n_bond} 个键")
print(f"键表内容示例: {bond_tab[:5]}")

# from Hamiltonian_jax import Ham_jax
# # 构建哈密顿量
# H = Ham_jax(L1, L2, k1, k2, A1, A2, A3, B1, B2, B3, lambda_param, h, J1plus, J2plus, J3plus, bond_tab, n_bond)
# print(f"哈密顿量形状: {H.shape}")