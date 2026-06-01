import IO

[dim, unit_vectors] = IO.read_unit_vectors(filepath='unit_vector.in')
print(f"维度: {dim}")
print("基础矢量:")
print(unit_vectors)

[spin_dim, spin_num, cell_spins] = IO.read_spin_in_cell(dim=dim, filepath='cellspin.in')
print(f"自旋晶胞维度: {spin_dim}")
print(f"自旋数目: {spin_num}")
print("自旋晶胞平移矢量:")
print(cell_spins)

IO.gen_bond_table(spin_positions=cell_spins, unit_vectors=unit_vectors)
IO.read_bond_table(filepath='bond.log')