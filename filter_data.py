import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

PATH = "data/raw/sample.csv"
df = pd.read_csv(PATH)

final_rows = []

for i in range(len(df)):
    row = df.iloc[i]
    smiles_string = row['smiles']
    try:
        molecule = Chem.MolFromSmiles(smiles_string)
        molecule = Chem.AddHs(molecule) # adds explicit Hydrogen atoms to heavy atoms to complete molecule
        num_atoms = molecule.GetNumAtoms()
        all_atoms = molecule.GetAtoms()

        mol_fsp3 = Chem.rdMolDescriptors.CalcFractionCSP3(molecule)
        molwt = Chem.rdMolDescriptors.CalcExactMolWt(molecule)

        AllChem.EmbedMolecule(molecule, useRandomCoords=False)
        AllChem.UFFOptimizeMolecule(molecule)
        atom_coordinates_np = molecule.GetConformer().GetPositions()
        
        final_rows.append(row)
    except Exception as e:
        print (f"{smiles_string} at index {i} failed")
        print (e)
        print ()
        print ()

print (len(final_rows))
with open("data/raw/sample_filtere2.csv", "a") as f:
    f.write(f"smiles,molwt,fsp3,logp,binding_affinity,cypi,tpsa\n")
    for row in final_rows:
        smiles = row['smiles']
        f.write(f"{smiles}, {row['molwt']}, {row['fsp3']}, {row['logp']}, {row['binding_affinity']}, {row['cypi']}, {row['tpsa']}\n")