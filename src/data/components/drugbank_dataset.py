import pandas as pd
import numpy as np
from rdkit import Chem

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected, from_networkx
from torch_geometric.data import Data
from torch_cluster import knn_graph

import networkx as nx
from src.utils.constants import NUM_ATOM_TYPES, atom_idx_map

class DrugBankDataset(Dataset):
    def __init__(
            self,
            data_cfg,
            is_training: bool = True,
        ):
        
        super().__init__()
        # self.save_hyperparameters(logger=False)

        self.data_cfg = data_cfg
        self.is_training = is_training
        if is_training:
            self.metadata_csv = pd.read_csv(data_cfg.metadata_path_train)
        else:
            self.metadata_csv = pd.read_csv(data_cfg.metadata_path_test)

    def len(self) -> int:
        """
        get the total number of samples in dataset

        :return: number of unique instances in dataset
        """

        return len(self.metadata_csv.index)
    
    def __len__(self) -> int:
        """
        get the total number of samples in dataset

        :return: number of unique instances in dataset
        """

        return len(self.metadata_csv.index)

    def __getitem__(self, idx: int) -> Data:
        """
        Retrieves and returns molecule corresponding to queried row in CSV.

        :param idx: row index in dataset CSV to return

        :return: tg.data.Data object for queried instance at row `idx` in CSV
        """
        
        csv_row = self.metadata_csv.iloc[idx]
        csv_row = csv_row.fillna(-1)
        
        smiles_string = csv_row['smiles']

        """
        NOTE: 
        torch_geometric.utils has `from_smiles` that converts SMILES to tg.data.Data(...) objects.
        Decide whether to use that or write an explicit converter where we can control how the adjacency is constructed.

        NOTE:
        Decide whether to convert from SMILES->Data when queried OR during preprocessing in __init__().
        """

        molecule = Chem.MolFromSmiles(smiles_string)
        molecule = Chem.AddHs(molecule) # adds explicit Hydrogen atoms to heavy atoms to complete molecule
        num_atoms = molecule.GetNumAtoms()
        all_atoms = molecule.GetAtoms()
        
        # get node features
        atom_symbols = list(map(lambda atom : atom.GetSymbol(), all_atoms)) # get atomic names as characters
        atom_symbol_indexes = torch.Tensor(list(map(lambda sym : atom_idx_map[sym], atom_symbols))).long() # map atomic name to corresponding numeric index
        atom_ids_ohe = F.one_hot(atom_symbol_indexes, num_classes=NUM_ATOM_TYPES) # convert to one-hot encodings

        if self.is_training:
            properties = csv_row[['molwt', 'logp', 'qed', 'fsp3', 'tpsa']].to_numpy(dtype=np.float32).tolist()
            properties = torch.tensor(properties).float().view(1, -1)

        """
        NOTE: The following creates a 2D graph using RDKit-inferred bonds
        """
        nx_G = nx.Graph()
        for atom in molecule.GetAtoms(): 
            nx_G.add_node(atom.GetIdx(), symbol=atom.GetSymbol())  
        for bond in molecule.GetBonds(): 
            nx_G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type=bond.GetBondType())
        edge_index = from_networkx(nx_G).edge_index
        # assert edge_index.shape[1] > 0
        assert len(molecule.GetBonds()) > 0

        """
        NOTE: Uncomment the following if using 3D conformers to extract 2D graph
        
        # embed 2D graph in 3D space and run universal forcefield relaxation to get a conformer
        try:
            AllChem.EmbedMolecule(molecule, useRandomCoords=False)
            AllChem.UFFOptimizeMolecule(molecule)
        except Exception as e:
            print (f"{e} : [{smiles_string}]")

        # retrieve 3D structure and adjacency from SMILES string using RDKit (in numpy format)
        atom_coordinates_np = molecule.GetConformer().GetPositions()
        atom_coordinates = torch.from_numpy(atom_coordinates_np)
        edge_index = knn_graph(atom_coordinates, k=self.data_cfg.k, loop=False)
        edge_index = to_undirected(edge_index) # convert to undirected edges
        """

        if self.is_training:
            return Data(
                    x=atom_ids_ohe, # atomic identities as one-hot-encoded vectors
                    y=properties, # molecular properties to predict
                    edge_index=edge_index, # graph adjacency using k-NN construction
                    num_nodes=num_atoms
                )
        else:
            return Data(
                    x=atom_ids_ohe, # atomic identities as one-hot-encoded vectors
                    edge_index=edge_index, # graph adjacency using k-NN construction
                    num_nodes=num_atoms
                )

