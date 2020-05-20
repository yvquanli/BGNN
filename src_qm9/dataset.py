import os
import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import networkx as nx
import pandas as pd
from tqdm import tqdm

fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


class QM9Dataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(QM9Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv']

    @property
    def processed_file_names(self):
        return 'processed_qm9.pt'

    def download(self):
        raise NotImplementedError('please download and unzip dataset from \
        https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/qm9.zip , \
        and put it at data/raw')

    def process(self):
        data_path = self.raw_paths[0]
        target_path = self.raw_paths[1]
        self.property_names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0',
                               'u298', 'h298', 'g298', 'cv']
        self.target = pd.read_csv(target_path, index_col='mol_id')
        self.target = self.target[self.property_names]
        supplier = Chem.SDMolSupplier(data_path, removeHs=False)
        data_list = []
        for i, mol in tqdm(enumerate(supplier)):
            data = self.mol2graph(mol)
            if data is not None:
                data.y = torch.FloatTensor(self.target.iloc[i, :])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', ]]
            h_t.append(d['a_num'])
            h_t.append(d['acceptor'])
            h_t.append(d['donor'])
            h_t.append(int(d['aromatic']))
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            # 5 more
            h_t.append(d['ExplicitValence'])
            h_t.append(d['FormalCharge'])
            h_t.append(d['ImplicitValence'])
            h_t.append(d['NumExplicitHs'])
            h_t.append(d['NumRadicalElectrons'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def get_edges(self, g):
        e = {}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                   for x in (Chem.rdchem.BondType.SINGLE, \
                             Chem.rdchem.BondType.DOUBLE, \
                             Chem.rdchem.BondType.TRIPLE, \
                             Chem.rdchem.BondType.AROMATIC)]

            e_t.append(int(d['IsConjugated'] == False))
            e_t.append(int(d['IsConjugated'] == True))
            e[(n1, n2)] = e_t
        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    def mol2graph(self, mol):
        if mol is None:
            return None
        feats = chem_feature_factory.GetFeaturesForMol(mol)
        g = nx.DiGraph()

        # Create nodes
        assert len(mol.GetConformers()) == 1
        geom = mol.GetConformers()[0].GetPositions()
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i,
                       a_type=atom_i.GetSymbol(),
                       a_num=atom_i.GetAtomicNum(),
                       acceptor=0,
                       donor=0,
                       aromatic=atom_i.GetIsAromatic(),
                       hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(),

                       # 5 more node features
                       ExplicitValence=atom_i.GetExplicitValence(),
                       FormalCharge=atom_i.GetFormalCharge(),
                       ImplicitValence=atom_i.GetImplicitValence(),
                       NumExplicitHs=atom_i.GetNumExplicitHs(),
                       NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
                       )

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['acceptor'] = 1

        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j,
                               b_type=e_ij.GetBondType(),
                               # 1 more edge features 2 dim
                               IsConjugated=int(e_ij.GetIsConjugated()),
                               )

        node_attr = self.get_nodes(g)
        edge_index, edge_attr = self.get_edges(g)

        # Get Labels
        data = Data(
            x=node_attr,
            pos=torch.FloatTensor(geom),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=None  # as a placeholder
        )
        return data


if __name__ == '__main__':
    dataset = QM9Dataset('..' + '/QM9_dataset_moleculenet')
