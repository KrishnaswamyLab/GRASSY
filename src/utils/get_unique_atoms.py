import pandas as pd
from rdkit import Chem

PATH = "../../data/processed/drugbank_processed_with_props.csv"

df = pd.read_csv(PATH)

elems = set()

for i in range(len(df)):
    row = df.iloc[i]
    smiles = row['smiles']
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    symbols = list(map(lambda atom : atom.GetSymbol(), atoms))
    for s in symbols:
        elems.add(s)

print (elems)
print (len(elems))
