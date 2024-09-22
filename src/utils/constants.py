NUM_ATOM_TYPES = 19 # TODO: confirm the number of total atoms considered

# maps atomic name to numeric index for conversion to one-hot encoding
# retrived from EMD: https://github.com/ehoogeboom/e3_diffusion_for_molecules/blob/main/configs/datasets_config.py
atom_idx_map = {
    'H': 0, 
    'C': 1, 
    'N': 2, 
    'O': 3, 
    'S': 4,
    'F': 5
}