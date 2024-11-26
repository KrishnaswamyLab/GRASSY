# maps atomic name to numeric index for conversion to one-hot encoding
# retrived from EMD: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/configs/datasets_config.py

# unique_atoms = ['H', 'Mg', 'Tc', 'Bi', 'B', 'Hg', 'Fe', 'Cu', 'Al', 'C', 'N', 'Cl', 'Mn', 'V', 'Br', 'Co', 'As', 'Si', 'Ag', 'S', 'Au', 'K', 'O', 'P', 'Na', 'Se', 'W', 'F', 'Zn', 'I', 'Gd', 'Ca', 'Mo']
unique_atoms = ['Ta', 'Ho', 'La', 'Hg', 'H', 'Pt', 'O', 'Ac', 'Ge', 'Li', 'Lu', 'N', 'Co', 'I', 'Ni', 'Br', 'Bk', 'Sm', 'Au', 'Xe', 'Te', 'Cu', 'Sr', 'V', 'Hf', 'Ga', 'Mn', 'P', 'Cs', 'Ca', 'Ba', 'Zn', 'Cl', 'Al', 'Gd', 'Se', 'Mo', 'Tc', 'Ne', 'Pd', 'Kr', 'C', 'Na', 'Fe', 'Ru', 'Ti', 'Ce', 'Ag', 'Si', 'Cd', 'Be', 'Bi', 'F', 'S', 'Nd', 'Re', 'Y', 'In', 'Nb', 'Os', 'Rb', 'Pb', 'K', 'Ra', 'Zr', 'Sb', 'As', 'He', 'Sn', 'Tl', 'Cr', 'Mg', 'W', 'B']

atom_idx_map = {key: i for i, key in enumerate(unique_atoms)} # create indexes for all unique atoms
# print (len(atom_idx_map))

NUM_ATOM_TYPES = len(atom_idx_map) # confirm the number of total atoms considered
# print (NUM_ATOM_TYPES)