import pickle
from rdkit import Chem
from abc import abstractmethod
from graph import get_num_of_heavy_atoms, HyperGraph, HyperNode
from collections import Counter
from typing import Union


class OverlapTokenizer:
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        all_vocabs = pickle.load(open(vocab_path, 'rb'))
        self.vocab_keys = ['atom_vocab', 'bond_vocab', 'aromatic_ring_vocab', 'non_aromatic_ring_vocab', 'hyper_token_vocab']
        self.all_vocabs = {k: all_vocabs[k] for k in self.vocab_keys}

        self.kekulize = False
        self.isomeric = False

        self.vocab_dict = {}
        # bi mapping for smiles and idx
        self.idx2smiles, self.smiles2idx = [], {}
        # self.max_num_nodes = 0

        # self.pad, self.end, self.mask, self.unk = '<pad>', '<s>', '<mask>', '<unk>'
        # for smi in [self.pad, self.end, self.mask, self.unk]:
        #     self.smiles2idx[smi] = len(self.idx2smiles)
        #     self.idx2smiles.append(smi)
        self.atom, self.bond, self.aring, self.naring, self.hyper = '<atom>', '<bond>', '<aring>', '<naring>', '<hyper>'
        for smi in [self.atom, self.bond, self.aring, self.naring, self.hyper]:
            self.smiles2idx[smi] = len(self.idx2smiles)
            self.idx2smiles.append(smi)

        for key, vocab in self.all_vocabs.items():
            for smiles, freq in vocab.items():
                atom_num = get_num_of_heavy_atoms(smiles)
                assert smiles not in self.vocab_dict, 'every smiles in vocab should be unique'
                self.vocab_dict[smiles] = (atom_num, freq, key[:-6])
                # self.max_num_nodes = max(self.max_num_nodes, atom_num)
                self.smiles2idx[smiles] = len(self.idx2smiles)
                self.idx2smiles.append(smiles)
        
        # self.max_num_nodes += 2 # start, padding

    @abstractmethod
    def tokenize(self, mol: Chem.Mol):
        pass

    def idx_to_smiles(self, idx):
        return self.idx2smiles[idx]

    def smiles_to_idx(self, smiles):
        return self.smiles2idx[smiles]
    
    def pad_idx(self):
        return self.smiles2idx[self.pad]
    
    def end_idx(self):
        return self.smiles2idx[self.end]
    
    def __call__(self, mol: Chem.Mol):
        return self.tokenize(mol)
    
    def __len__(self):
        return len(self.idx2smiles)
    

class MergeTokenizer(OverlapTokenizer):
    def __init__(self, vocab_path):
        super().__init__(vocab_path)

    def contract_basic_tokens_from_vocab(self, hg: HyperGraph):
        aromatic_ring_infos = []
        for atom_ids, smiles in hg.aromatic_ring_smiles.items():
            if smiles not in self.vocab_dict:
                smiles = self.aring
            aromatic_ring_infos.append(HyperNode(smiles=smiles, atom_ids=atom_ids, hid=None))

        # TODO: handle large rings        
        non_aromatic_ring_infos = []
        for atom_ids, smiles in hg.non_aromatic_ring_smiles.items():
            if smiles not in self.vocab_dict:
                smiles = self.naring
            non_aromatic_ring_infos.append(HyperNode(smiles=smiles, atom_ids=atom_ids, hid=None))

        # Add non-aromatic bonds as HyperNodes, excluding those entirely within rings
        bond_infos = []
        for atom_ids, smiles in hg.bond_smiles.items():
            # Skip if both atoms are in the same ring
            is_in_same_ring = False
            for ring in aromatic_ring_infos + non_aromatic_ring_infos:
                if atom_ids[0] in ring.atom_ids and atom_ids[1] in ring.atom_ids:
                    is_in_same_ring = True
                    break
            if is_in_same_ring:
                continue
            if smiles not in self.vocab_dict:
                continue
            bond_infos.append(HyperNode(smiles=smiles, atom_ids=atom_ids, hid=None))

        visited_atom_ids = set()
        for node in aromatic_ring_infos + non_aromatic_ring_infos + bond_infos:
            visited_atom_ids.update(node.atom_ids)

        atom_infos = []
        for aid, atom_smiles in enumerate(hg.atom_smiles):
            if aid not in visited_atom_ids:
                if atom_smiles not in self.vocab_dict:
                    atom_smiles = self.atom
                atom_infos.append(HyperNode(smiles=atom_smiles, atom_ids=(aid,), hid=None))

        hg.hyper_nodes = aromatic_ring_infos + non_aromatic_ring_infos + bond_infos + atom_infos
        hg.get_mapping()
        hg.get_hyper_pairs()

    def tokenize(self, mol: Chem.Mol, _idx: Union[int, str] = None, smiles: str = None, _name: str = None) -> HyperGraph:
        hg = HyperGraph(mol, _idx, smiles, _name, self.kekulize, self.isomeric)        
        self.contract_basic_tokens_from_vocab(hg)
        hg.check_coverage()

        while True:
            max_freq, merge_smiles = -1, ''
            for hyper_pair in hg.hyper_pairs.values():
                smiles = hyper_pair.smiles
                if smiles in self.vocab_dict:
                    freq = self.vocab_dict[smiles][1]
                    if freq > max_freq:
                        max_freq, merge_smiles = freq, smiles
            if max_freq == -1:
                break
            # print('contracting', merge_smiles, max_freq)
            hg.contract_hyper_pair(merge_smiles)
            hg.check_coverage()

        return hg
    
    def __call__(self, mol: Chem.Mol, smiles: str = None):
        hg = self.tokenize(mol, smiles=smiles)
        return hg

    def to_count_feat(self, hg: HyperGraph):
        feat = [0] * len(self.idx2smiles)
        for node in hg.hyper_nodes:
            token_id = self.smiles_to_idx(node.smiles)
            feat[token_id] += 1
        return feat


if __name__ == '__main__':
    vocab_path = './data/lba/vocab_213.pkl'
    tokenizer = MergeTokenizer(vocab_path)

    for idx, (k, v) in enumerate(tokenizer.vocab_dict.items()):
        print(idx, k, v)
