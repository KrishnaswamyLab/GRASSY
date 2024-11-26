import pandas as pd
from rdkit import Chem
from src.analysis.prop_helpers import get_molecular_properties

PATH = "data/raw/drugbank_12k.csv"
df = pd.read_csv(PATH)

final_csv = []

for i in range(len(df)):
    row = df.iloc[i]
    smiles = row['SMILES']
    name = row['Name']
    
    if isinstance(smiles, str) and len(smiles) > 0:
        try:
            molecule = Chem.MolFromSmiles(smiles)
            molecule = Chem.AddHs(molecule)
            properties = get_molecular_properties(molecule)

            if len(molecule.GetBonds()) > 0:
                molwt = properties['molwt']
                tpsa = properties['tpsa']
                qed = properties['qed']
                fsp3 = properties['fsp3']
                logp = properties['logp']
                # strain = properties['strain']

                final_csv.append([
                    i, 
                    name,
                    smiles, 
                    molwt,
                    logp,
                    qed,
                    fsp3,
                    tpsa,
                    # strain
                ])
        except:
            print (f"Bad molecule at index {i}:", smiles)

df_new = pd.DataFrame(final_csv, columns=["index", "name", "smiles", "molwt", "logp", "qed", "fsp3", "tpsa"])
df_new.to_csv("data/processed/drugbank_12k_processed_with_props.csv", index=False)

# import pandas as pd
# from rdkit import Chem

# PATH = "data/processed/drugbank_12k_processed_with_props.csv"
# df = pd.read_csv(PATH)

# final_csv = []

# for i in range(len(df)):
#     row = df.iloc[i]
#     smiles = row['smiles']

#     molecule = Chem.MolFromSmiles(smiles)
#     molecule = Chem.AddHs(molecule)
#     atoms = molecule.GetAtoms()
#     assert len(molecule.GetBonds()) > 0, f"{i}, {smiles}"

# df_new = pd.DataFrame(final_csv, columns=["index", "name", "smiles", "molwt", "logp", "qed", "fsp3", "tpsa"])
# df_new.to_csv("data/processed/drugbank_12k_processed_with_props.csv", index=False)    