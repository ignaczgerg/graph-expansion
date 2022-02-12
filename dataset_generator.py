import pandas as pd
from rdkit import Chem
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from tqdm import tqdm
import networkx as nx


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, target, expand=False, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.expand = expand
        self.target = target
        self.test = test
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass


    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, mol in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol_obj = Chem.MolFromSmiles(mol['smiles'])
            # Get node features
            node_feats = self._get_node_features(mol_obj)
            # Get edge features
            edge_feats = self._get_edge_features(mol_obj)
            # Get adjacency info
            edge_index = self._get_adjacency_info(mol_obj)
            # Get labels info
            label = self._get_labels(mol[self.target])

            # Create data object
            if self.expand:
                data = Data(x=node_feats, 
                            edge_index=edge_index,
                            edge_attr=torch.zeros(1), # edge_feats has been taken out for the graph expansion model,
                            y=label,
                            smiles=mol['smiles']
                            ) 
                if self.test:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_test_{index}.pt'))
                else:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_{index}.pt'))
            
            else:
                data = Data(x=node_feats, 
                        edge_index=edge_index,
                        edge_attr=edge_feats,
                        y=label,
                        smiles=mol["smiles"]
                        ) 
                if self.test:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_test_{index}.pt'))
                else:
                    torch.save(data, 
                        os.path.join(self.processed_dir, 
                                    f'data_{index}.pt'))

    def _get_node_features(self, mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number
            node_feats.append(atom.GetAtomicNum())
            # Feature 2: Atom degree
            node_feats.append(atom.GetDegree())
            # Feature 3: Formal charge
            node_feats.append(atom.GetFormalCharge())
            # Feature 4: Hybridization
            node_feats.append(atom.GetHybridization())
            # Feature 5: Aromaticity
            node_feats.append(atom.GetIsAromatic())
            # Feature 6: Total Num Hs
            node_feats.append(atom.GetTotalNumHs())
            # Feature 7: Radical Electrons
            node_feats.append(atom.GetNumRadicalElectrons())
            # Feature 8: In Ring
            node_feats.append(atom.IsInRing())
            # Feature 9: Chirality
            node_feats.append(atom.GetChiralTag())

            # Append node features to matrix
            all_node_feats.append(node_feats)
        
        all_node_feats = np.asarray(all_node_feats)

        if self.expand:
            l = []
            for i in range(len(all_node_feats)):
                l.append(0)
            for elem in all_node_feats:
                for char in elem:
                    l.append(char)
                    
            all_node_feats = l

        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            # Feature 1: Bond type (as double)
            edge_feats.append(bond.GetBondTypeAsDouble())
            # Feature 2: Rings
            edge_feats.append(bond.IsInRing())
            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    
    def _add_self_loop(self, adj_matrix: np.array):
        num_edges = adj_matrix[0] + adj_matrix[1]
        num_edge = 0

        for row in range(adj_matrix.shape[0]):
            for column in range(adj_matrix.shape[1]):
                if row == column:
                    adj_matrix[row][column] = 1

        return adj_matrix

    def _get_adjacency_info(self, mol):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        # Yeah, this code below is nice, but we don't have time for this. 
        # edge_indices = []
        # for bond in mol.GetBonds():
        #     i = bond.GetBeginAtomIdx()
        #     j = bond.GetEndAtomIdx()
        #     edge_indices += [[i, j], [j, i]]

        adj_matrix = Chem.GetAdjacencyMatrix(mol)
        adj_matrix = self._add_self_loop(adj_matrix=adj_matrix)

        if self.expand:
            adj_matrix = self._graph_expansion(G = adj_matrix, new_edges = 9) # The new_edges value is absolutely hard-coded. Sorry. 
                                                                         # new_edges: the number of edge_features the matrix has.  

        #adj_matrix = self._to_edge_list(adj_matrix)
        graph = nx.Graph(adj_matrix)
        edge_index = list(graph.edges)
        # edge_indices = torch.tensor(edge_indices)
        # print(adj_matrix.shape)

        # row, col, edge_attr = adj_matrix.t().coo()
        # edge_indices = torch.stack([row, col], dim=0)
        # edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return torch.tensor(edge_index)
        

    def _to_edge_list(self, adj_matrix: np.array):
            num_edges = adj_matrix[0] + adj_matrix[1]

            num_edge = 0
            edge_set = set()

            for row in range(adj_matrix.shape[0]):
                for column in range(adj_matrix.shape[1]):
                    if adj_matrix.item(row,column) == 1: # we don't get rid of the repeated edges!
                        num_edge += 1
                        edge_set.add((row,column))
            return torch.tensor(list(edge_set))


    def _graph_expansion(self, G, new_edges) -> np.array:
        """
        The function adds nodes to already existing nodes.
        :graph: numpy array
        :new_edges: int 
        """
        num_edges = len(list(G[0]))
        for i in range(num_edges):
            num_edges = len(list(G[0]))
            support_matrix = np.zeros(((num_edges-new_edges*i) + new_edges * (i + 1), (num_edges-new_edges*i) + new_edges * (i + 1)))
            support_matrix[i,:] = 1
            support_matrix[:,i] = 1
            support_matrix[:num_edges, :num_edges] = G
            G = support_matrix
        return G

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data
