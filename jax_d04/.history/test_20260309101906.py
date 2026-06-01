import IO

[dim, unit_vectors] = IO.read_unit_vector(filepath='unit_vector.in')
print(f"维度: {dim}")
print("基础矢量:")
print(unit_vectors)

[spin_dim, cell_spins] = IO.read_spin_in_cell(dim=dim, filepath='cellspin.in')