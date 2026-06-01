import IO
from Hamiltonian_jax import Ham_jax
import numpy as np
import jax.numpy as jnp

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
k1 = jnp.array([0.1])
k2 = jnp.array([0.1])
Nsites = 10
h = 1.0 / Nsites

A1 = A2 = A3 = 0.49126303j
B1 = B2 = B3 = 0.22640955
lambda_param = 0.94176189
    
# 读取晶格信息
[bond_tab, n_spin, n_bond] = IO.create_bond_table_file(unit_vector_filepath='unit_vector.in', cellspin_filepath='cellspin.in')
[bond_tab, n_spin, n_bond] = IO.read_bond_table(filepath='bond.log')

# JAX 不能直接处理 object dtype，转为数值表: [i, j, site_i, site_j, rij_x, rij_y, type]
bond_tab_numeric = np.zeros((n_bond, 7), dtype=np.float64)
for b in range(n_bond):
	bond = bond_tab[b]
	bond_tab_numeric[b, 0] = int(bond[0])
	bond_tab_numeric[b, 1] = int(bond[1])
	bond_tab_numeric[b, 2] = int(bond[2])
	bond_tab_numeric[b, 3] = int(bond[3])
	if len(bond) >= 7:
		rij_frac = np.asarray(bond[5], dtype=np.float64)
		bond_tab_numeric[b, 6] = int(bond[6])
	else:
		rij_frac = np.asarray(bond[4], dtype=np.float64)
		bond_tab_numeric[b, 6] = float(b % 3)
	bond_tab_numeric[b, 4] = rij_frac[0]
	bond_tab_numeric[b, 5] = rij_frac[1]

bond_tab_jax = jnp.asarray(bond_tab_numeric)

print(f"读取键表: 共 {n_bond} 个键")
print(f"键表内容示例: {bond_tab[:5]}")

from Hamiltonian_jax import Ham_jax
# 构建哈密顿量
H = Ham_jax(
	omega=0.0,
	k1=k1,
	k2=k2,
	A1=A1,
	A2=A2,
	A3=A3,
	B1=B1,
	B2=B2,
	B3=B3,
	lambda_param=lambda_param,
	h=h,
	J1plus=J1plus,
	J2plus=J2plus,
	J3plus=J3plus,
	bond_tab=bond_tab_jax,
	spin_n=n_spin,
	bond_n=n_bond,
)
print(f"哈密顿量形状: {H.shape}")