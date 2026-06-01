import IO
from Hamiltonian_jax import Ham_jax

[bond_tab,bond_n]=IO.create_bond_table_file(unit_vector_filepath='unit_vector.in', cellspin_filepath='cellspin.in')
J1=