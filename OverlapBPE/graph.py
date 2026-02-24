import pickle
import functools

import rdkit
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType as HT
from typing import Optional, List, Dict, Tuple, Union

from tqdm import tqdm
from collections import Counter

import numpy as np

from dataclasses import dataclass

import networkx as nx
import matplotlib.pyplot as plt


BOND_SYMBOL = {BT.SINGLE: '-', BT.DOUBLE: '=', BT.TRIPLE: '#', BT.AROMATIC: ':', '<OTHER>': '?'}

def get_bond_symbol(bond_type):
    if bond_type in BOND_SYMBOL:
        return BOND_SYMBOL[bond_type]
    else:
        return BOND_SYMBOL['<OTHER>']

def get_num_of_heavy_atoms(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        return sum([atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()])
    except Exception as e:
        print(f'Error getting number of heavy atoms: {e}')
        return 0

# def get_frag_elements(smiles):
#     mol = Chem.MolFromSmiles(smiles, sanitize=False)
#     heavy_atoms = [atom.GetSymbol() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
#     sorted_heavy_atoms = sorted(heavy_atoms)
#     return ''.join(sorted_heavy_atoms)

def is_ring(sub_mol, size):
    bonds = sub_mol.GetBonds()
    if len(bonds) != size:
        return False
    for i in range(size-1):
        if sub_mol.GetBondBetweenAtoms(i, i+1) is None:
            return False
    if sub_mol.GetBondBetweenAtoms(0, size-1) is None:
        return False
    return True

def is_aromatic_ring(mol, ring):
    # TODO: check if ring is ring
    non_aromatic_bonds = []

    for i in range(len(ring)):
        a1_idx = ring[i]
        a2_idx = ring[(i + 1) % len(ring)]

        # check atoms are aromatic
        atom1 = mol.GetAtomWithIdx(a1_idx)
        atom2 = mol.GetAtomWithIdx(a2_idx)
        if not (atom1.GetIsAromatic() and atom2.GetIsAromatic()):
            return False

        # check bond is aromatic
        bond = mol.GetBondBetweenAtoms(a1_idx, a2_idx)
        if not bond.GetIsAromatic():
            non_aromatic_bonds.append(bond)

    # fix non-aromatic bonds in aromatic rings
    for bond in non_aromatic_bonds:
        bond.SetIsAromatic(True)
    return True

def get_atom_smiles(atomic_symbol, is_aromatic=False, charge=0):
    atom_smiles = atomic_symbol
    if is_aromatic:
        atom_smiles = atom_smiles.lower()
    if charge > 0:
        if charge == 1:
            atom_smiles = f'[{atom_smiles}+]'
        else:
            atom_smiles = f'[{atom_smiles}+{charge}]'
    elif charge < 0:
        if charge == -1:
            atom_smiles = f'[{atom_smiles}-]'
        else:
            atom_smiles = f'[{atom_smiles}{charge}]'
    return atom_smiles

def get_atom_set_tuple(frag_atom_ids: List[int]):
    return tuple(set(sorted(frag_atom_ids)))

def get_frag_smiles(mol, frag_node_idxs, isomeric=False):
    assert len(frag_node_idxs) > 1, 'for single atom, use atom_smiles'
    frag_smiles = Chem.MolFragmentToSmiles(mol, frag_node_idxs, kekuleSmiles=False, isomericSmiles=isomeric, canonical=True)
    frag_mol = Chem.MolFromSmiles(frag_smiles, sanitize=False)
    frag_smiles_fix = Chem.MolToSmiles(frag_mol, isomericSmiles=isomeric, canonical=True)

    # rdkit will get cC and Cc as canonical smiles
    if get_num_of_heavy_atoms(frag_smiles_fix) == 2 and len(frag_smiles_fix) == 2:
        frag_smiles_fix = ''.join(sorted(list(frag_smiles_fix)))

    return frag_smiles_fix

if rdkit.__version__ < '2023.09.4':
    print(f'rdkit version {rdkit.__version__} < 2023.09.4, using GetSymmSSSR')
    get_rings = Chem.GetSymmSSSR
else:
    print(f'rdkit version {rdkit.__version__} >= 2023.09.4, using GetSymmSSSR')
    get_rings = Chem.GetSymmSSSR

def extract_rings(mol: Chem.Mol):
    aromatic_rings = []
    non_aromatic_rings = []
    for ring in get_rings(mol):
        if is_aromatic_ring(mol, ring):
            aromatic_rings.append(list(ring))
        else:
            non_aromatic_rings.append(list(ring))
    return aromatic_rings, non_aromatic_rings

def get_2d_mol_image(mol, kekulize=False, legend=None, atoms=[], bonds=[], bond_colors={}, size=(300, 300)):
    img = Draw.MolToImage(mol, kekulize=kekulize, legend=legend, highlightAtoms=atoms, highlightBonds=bonds, highlightBondColors=bond_colors, size=size)
    return img

def concat_images_in_grid(images, molsPerRow=10, subImgSize=(150, 150)):
    img_width, img_height = subImgSize
    if len(images) < molsPerRow:
        cols = len(images)
        rows = 1
    else:
        cols = molsPerRow
        rows = (len(images) + molsPerRow - 1) // molsPerRow

    from IPython.display import display
    from PIL import Image

    grid_image = Image.new('RGB', (cols * img_width, rows * img_height))
    for i, img in enumerate(images):
        x = (i % cols) * img_width
        y = (i // cols) * img_height
        grid_image.paste(img, (x, y))
    
    display(grid_image)

class MolGraph:
    def __init__(self, mol: Chem.Mol, _idx: Union[int, str] = None, smiles: str = None,
                 _name: str = None, kekulize: bool = False, isomeric: bool = False):
        self.kekulize = kekulize
        self.isomeric = isomeric

        self.mol = mol
        self._idx = _idx
        if smiles is None or smiles == '':
            self.smiles = Chem.MolToSmiles(mol, kekulize=kekulize, isomericSmiles=isomeric)
        else:
            self.smiles = smiles
        self._name = _name

        self.num_atoms = mol.GetNumAtoms()
        self.num_bonds = mol.GetNumBonds()

        self._init_atom_info()
        self._init_bond_info()
        self._init_basic_tokens()

    def _init_atom_info(self):
        try:
            conf = self.mol.GetConformer()
        except:
            try:
                AllChem.Compute2DCoords(self.mol)
                conf = self.mol.GetConformer()
            except Exception as e:
                raise ValueError(f"Failed to compute 2D coordinates: {str(e)}")
        
        pos = conf.GetPositions()
        self.pos = np.array(pos)

        self.atom_smiles = []
        self.atomic_symbols = []
        self.atomic_numbers = []
        self.aromatics = []
        self.charges = []
        self.hybrid_types = []
        self.num_hs = []

        for atom in self.mol.GetAtoms():
            atomic_symbol = atom.GetSymbol()
            atomic_number = atom.GetAtomicNum()
            is_aromatic = atom.GetIsAromatic()
            charge = atom.GetFormalCharge()
            hybridization = atom.GetHybridization()
            num_hs = atom.GetNumExplicitHs()

            self.atomic_symbols.append(atomic_symbol)
            self.atomic_numbers.append(atomic_number)
            self.aromatics.append(is_aromatic)
            self.charges.append(charge)
            
            atom_smiles = get_atom_smiles(atomic_symbol, is_aromatic, charge)
            self.atom_smiles.append(atom_smiles)
            hybrid_type = 0
            hybridization = atom.GetHybridization()
            if hybridization == HT.SP:
                hybrid_type = 1
            elif hybridization == HT.SP2:
                hybrid_type = 2
            elif hybridization == HT.SP3:
                hybrid_type = 3

            self.hybrid_types.append(hybrid_type)
            self.num_hs.append(num_hs)

        self.atomic_numbers = np.array(self.atomic_numbers)
        self.aromatics = np.array(self.aromatics)
        self.hybrid_types = np.array(self.hybrid_types)
        self.num_hs = np.array(self.num_hs)

    def _init_bond_info(self):
        self.rows, self.cols, bond_symbols, bonds = [], [], [], []
        for bond in self.mol.GetBonds():
            # different GraphBPE
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            # _bond = mol.GetBondBetweenAtoms(end, start)
            # if _bond is None:
            #     print(start, end, 'has a bond', end, start, 'has no bond')
            start, end = get_atom_set_tuple([start, end])
            self.rows += [start]
            self.cols += [end]
            bond_symbols += [get_bond_symbol(bond.GetBondType())]
            bonds += [bond]
        
        self.rows = np.array(self.rows)
        self.cols = np.array(self.cols)
        perm = np.lexsort((self.cols, self.rows))

        self.bond_symbols = [bond_symbols[_] for _ in perm]
        self.bonds = {}
        for i in perm:
            a1, a2 = self.rows[i], self.cols[i]
            bd = bonds[i]
            if bd.GetIsAromatic():
                continue
            k = get_atom_set_tuple([int(a1), int(a2)])
            assert k not in self.bonds
            self.bonds[k] = bd

    def _init_basic_tokens(self):
        self.bond_smiles = {}
        self.aromatic_ring_smiles, self.non_aromatic_ring_smiles = {}, {}
        self.extract_basic_tokens()

    def is_aromatic_ring(self, ring):
        return is_aromatic_ring(self.mol, ring)
    
    def extract_basic_tokens(self):
        aromatic_rings, non_aromatic_rings = extract_rings(self.mol)
        for ring in aromatic_rings:
            atom_ids = get_atom_set_tuple(ring)
            ring_smiles = self.get_frag_smiles(atom_ids)
            self.aromatic_ring_smiles[atom_ids] = ring_smiles
        for ring in non_aromatic_rings:
            atom_ids = get_atom_set_tuple(ring)
            ring_smiles = self.get_frag_smiles(atom_ids)
            self.non_aromatic_ring_smiles[atom_ids] = ring_smiles

        for (a1, a2), bond in self.bonds.items():
            is_in_same_ring = False
            for ring in aromatic_rings + non_aromatic_rings:
                if a1 in ring and a2 in ring:
                    is_in_same_ring = True
                    break
            if is_in_same_ring:
                continue
            atom_ids = get_atom_set_tuple([a1, a2])
            bond_smiles = self.get_frag_smiles(atom_ids)
            self.bond_smiles[atom_ids] = bond_smiles

    @functools.lru_cache(maxsize=128)
    def get_frag_smiles(self, frag_atom_ids: Tuple[int]):
        frag = tuple(set([int(a) for a in sorted(frag_atom_ids)]))
        return get_frag_smiles(self.mol, frag, isomeric=self.isomeric)

@dataclass
class HyperNode:
    smiles: str
    atom_ids: Tuple[int]
    hid: int
    contracted_from: Optional[List[str]] = None

@dataclass
class HyperPair:
    smiles: str
    atom_ids: Tuple[int]
    h1: HyperNode
    h2: HyperNode
    edge_type: str

    @property
    def h1_idx(self):
        return self.h1.hid

    @property
    def h2_idx(self):
        return self.h2.hid

    @property
    def h1_smiles(self):
        return self.h1.smiles

    @property
    def h2_smiles(self):
        return self.h2.smiles

class HyperGraph(MolGraph):
    @classmethod
    def from_mol_graph(cls, mol_graph: MolGraph):
        return cls(
            mol_graph.mol,
            mol_graph._idx,
            mol_graph.smiles,
            mol_graph._name,
            mol_graph.kekulize,
            mol_graph.isomeric,
        )
    
    def __init__(self, mol: Chem.Mol, _idx: Union[int, str] = None, smiles: str = None,
                 _name: str = None, kekulize: bool = False, isomeric: bool = False):
        super().__init__(mol, _idx, smiles, _name, kekulize, isomeric)


    def _get_bonds_between_hyper_nodes(self, h1_idx: int, h2_idx: int):
        h1_nodes_set = set(self.hid_to_aids[h1_idx])
        h2_nodes_set = set(self.hid_to_aids[h2_idx])

        hh_bonds = []
        # enumerate all bonds between h1 and h2
        for a1_idx in sorted(h1_nodes_set - h2_nodes_set):
            for a2_idx in sorted(h2_nodes_set - h1_nodes_set):
                bond = self.mol.GetBondBetweenAtoms(a1_idx, a2_idx)
                if bond is None:
                    continue

                # if a1 and a2 are partially connected by h3, do not consider as a connecting bond
                a1_hids = set(self.aid_to_hids[a1_idx])
                a2_hids = set(self.aid_to_hids[a2_idx])
                h3_idxs = a1_hids & a2_hids - {h1_idx, h2_idx}

                a1_a2_bond_in_another_hyper_node = False
                for h3_idx in h3_idxs:
                    if len(set(self.hid_to_aids[h3_idx]) - {a1_idx, a2_idx}) > 0:
                        a1_a2_bond_in_another_hyper_node = True
                        break
                        
                if a1_a2_bond_in_another_hyper_node:
                    continue

                a1_smiles = self.atom_smiles[a1_idx]
                a2_smiles = self.atom_smiles[a2_idx]
                a1_smiles, a2_smiles = sorted([a1_smiles, a2_smiles])
                bond_type = get_bond_symbol(bond.GetBondType())
                hh_bonds.append(f'{a1_smiles}{bond_type}{a2_smiles}')
        return hh_bonds
    
    def get_hyper_pairs(self):
        hyper_pairs = {}
        # Find connected hyper nodes through shared atoms
        for h1_idx in range(len(self.hid_to_aids)):
            for h2_idx in range(h1_idx + 1, len(self.hid_to_aids)):
                h1_atoms = set(self.hid_to_aids[h1_idx])
                h2_atoms = set(self.hid_to_aids[h2_idx])
                
                connecting_bonds = self._get_bonds_between_hyper_nodes(h1_idx, h2_idx)

                # Get shared atoms between two hyper nodes
                shared_atoms = sorted(h1_atoms & h2_atoms)

                if len(connecting_bonds) == 0 and len(shared_atoms) == 0:
                    continue
                
                # edge type contains all shared atoms and connecting bonds
                # TODO: recognize shared bonds from shared atoms
                edge_type = []

                for bond in connecting_bonds:
                    edge_type.append(bond)

                for aid in shared_atoms:
                    atom_smiles = self.atom_smiles[aid]
                    edge_type.append(f'<{atom_smiles}>')

                # for a1_idx in sorted(h1_atoms - h2_atoms):
                #     for a2_idx in sorted(h2_atoms - h1_atoms):
                #         bond_id = get_atom_set_tuple([a1_idx, a2_idx])
                #         if bond_id in self.bond_smiles:
                #             bond_smiles = self.bond_smiles[bond_id]
                #             edge_type.append(f'{bond_smiles}')

                edge_type = ';'.join(sorted(edge_type))

                atom_ids = get_atom_set_tuple(h1_atoms | h2_atoms)
                frag_smiles = self.get_frag_smiles(atom_ids)
                h1 = self.hyper_nodes[h1_idx]
                h2 = self.hyper_nodes[h2_idx]
                hyper_pair = HyperPair(smiles=frag_smiles, atom_ids=atom_ids,
                                       h1=h1, h2=h2, edge_type=edge_type)
                hyper_pairs[(h1_idx, h2_idx)] = hyper_pair
        
        self.hyper_pairs = hyper_pairs
        return hyper_pairs
    
    def get_mapping(self):
        for hid, node in enumerate(self.hyper_nodes):
            node.hid = hid
        self.hid_to_aids = []
        self.aid_to_hids = [[] for _ in range(self.num_atoms)]
        # Build mapping from atom to hyper nodes
        for hid, node in enumerate(self.hyper_nodes):
            node.hid = hid
            self.hid_to_aids.append(node.atom_ids)
            for atom_idx in node.atom_ids:
                self.aid_to_hids[atom_idx].append(hid)

    def get_isolated_atoms(self):
        covered_atom_idxs = set()
        for node in self.hyper_nodes:
            covered_atom_idxs.update(node.atom_ids)

        atom_infos = []
        # add isolated atoms like Cl-
        if len(covered_atom_idxs) < self.num_atoms:
            for atom_idx in range(self.num_atoms):
                if atom_idx in covered_atom_idxs:
                    continue
                atom_infos.append(HyperNode(smiles=self.atom_smiles[atom_idx], atom_ids=(atom_idx,), hid=None))

        return atom_infos
    
    def contract_basic_tokens(self):
        aromatic_ring_infos = []
        for atom_ids, smiles in self.aromatic_ring_smiles.items():
            aromatic_ring_infos.append(HyperNode(smiles=smiles, atom_ids=atom_ids, hid=None))

        non_aromatic_ring_infos = []
        for atom_ids, smiles in self.non_aromatic_ring_smiles.items():
            non_aromatic_ring_infos.append(HyperNode(smiles=smiles, atom_ids=atom_ids, hid=None))

        # Add non-aromatic bonds as HyperNodes, excluding those entirely within rings
        bond_infos = []
        for atom_ids, smiles in self.bond_smiles.items():
            # Skip if both atoms are in the same ring
            is_in_same_ring = False
            for ring in aromatic_ring_infos + non_aromatic_ring_infos:
                if atom_ids[0] in ring.atom_ids and atom_ids[1] in ring.atom_ids:
                    is_in_same_ring = True
                    break
            if is_in_same_ring:
                continue
            bond_infos.append(HyperNode(smiles=smiles, atom_ids=atom_ids, hid=None))

        self.hyper_nodes = aromatic_ring_infos + non_aromatic_ring_infos + bond_infos
        
        atom_infos = self.get_isolated_atoms()
        self.hyper_nodes = self.hyper_nodes + atom_infos
        
        self.get_mapping()
        self.get_hyper_pairs()

    # when smiles is determined, there are 2 cases of sharing_hyper_node
    # 1. sharing_hyper_node = False, we want to remove other pairs including h1_idx or h2_idx, which means a hyper node is contracted only once
    # 2. sharing_hyper_node = True, we will keep all pairs except pair_atom_ids is a subset of another pair_atom_ids
    def contract_hyper_pair(self, smiles: str, sharing_hyper_node: bool = True) -> bool:
        contracted_pair_atom_ids = set()
        contracted_hyper_ids = set()
        contracted_hyper_pairs = set()
        hyper_nodes_to_add = []
        
        for (h1_idx, h2_idx), hyper_pair in self.hyper_pairs.items():
            if hyper_pair.smiles != smiles:
                continue

            if not sharing_hyper_node:
                # even though (h1, h2) constitute the smiles, h1 or h2 is already contracted in prev pairs, this pair will be skipped
                if h1_idx in contracted_hyper_ids or h2_idx in contracted_hyper_ids:
                    continue

            # pair_atom_ids is guaranteed to be sorted set
            pair_atom_ids = hyper_pair.atom_ids
            # Check if this frag is already contracted
            if tuple(pair_atom_ids) in contracted_pair_atom_ids:
                continue

            # Skip if this frag is a subset of another frag
            is_subset_of_another_pair = False
            for atom_ids in contracted_pair_atom_ids:
                if set(pair_atom_ids).issubset(set(atom_ids)):
                    is_subset_of_another_pair = True
                    break
            if is_subset_of_another_pair:
                continue

            # Skip if sharing not allowed and pair already contracted
            if not sharing_hyper_node and ((h1_idx, h2_idx) in contracted_hyper_pairs):
                continue

            contracted_pair_atom_ids.add(pair_atom_ids)
            contracted_hyper_ids.add(h1_idx)
            contracted_hyper_ids.add(h2_idx)
            contracted_hyper_pairs.add((h1_idx, h2_idx))

            # Create new contracted hyper node
            hyper_nodes_to_add.append(HyperNode(
                smiles=smiles,
                atom_ids=pair_atom_ids,
                hid=None,
                contracted_from=sorted([self.hyper_nodes[h1_idx].smiles, self.hyper_nodes[h2_idx].smiles]),
            ))

        if len(hyper_nodes_to_add) == 0:
            return False

        # Keep uncontracted nodes that aren't subsets of contractions
        hyper_nodes_to_retain = []
        for node in self.hyper_nodes:
            if not sharing_hyper_node:
                if node.hid in contracted_hyper_ids:
                    continue

            # if sharing_hyper_node is True, we will keep all nodes which have other uncontracted pairs

            should_retain = False
            # Check if node is part of any uncontracted pair
            for (h1_idx, h2_idx), hyper_pair in self.hyper_pairs.items():
                if node.hid in (h1_idx, h2_idx) and (h1_idx, h2_idx) not in contracted_hyper_pairs:
                    should_retain = True
                    break

            # all pairs including this node are contracted, skip
            if not should_retain:
                continue

            # remove this node if subset of contracted nodes
            is_subset_of_another_pair = False
            for pair_atom_ids in contracted_pair_atom_ids:
                if set(node.atom_ids).issubset(set(pair_atom_ids)):
                    is_subset_of_another_pair = True
                    break
            if is_subset_of_another_pair:
                continue

            node.contracted_from = None
            hyper_nodes_to_retain.append(node)

        # Update hyper nodes and recompute edges
        self.hyper_nodes = hyper_nodes_to_add + hyper_nodes_to_retain

        atom_infos = self.get_isolated_atoms()
        self.hyper_nodes = self.hyper_nodes + atom_infos

        # Rebuild mappings and edges
        self.get_mapping()
        self.get_hyper_pairs()
        return True
    
    def check_coverage(self):
        covered_atom_idxs = set()
        covered_bond_idxs = set()
        for hyper_idx, hyper_node in enumerate(self.hyper_nodes):
            atom_ids = hyper_node.atom_ids
            covered_atom_idxs.update(atom_ids)

        assert len(covered_atom_idxs) == self.num_atoms, f'{self._idx} = {self.smiles} atoms not fully covered!'

        for hyper_node in self.hyper_nodes:
            atom_ids = hyper_node.atom_ids
            for i in range(len(atom_ids)):
                for j in range(i + 1, len(atom_ids)):
                    bond = self.mol.GetBondBetweenAtoms(atom_ids[i], atom_ids[j])
                    if bond is not None:
                        covered_bond_idxs.add(bond.GetIdx())

        for (h1_idx, h2_idx), hyper_pair in self.hyper_pairs.items():
            h1_atoms = set(self.hid_to_aids[h1_idx])
            h2_atoms = set(self.hid_to_aids[h2_idx])
            for a1_idx in sorted(h1_atoms - h2_atoms):
                for a2_idx in sorted(h2_atoms - h1_atoms):
                    bond = self.mol.GetBondBetweenAtoms(a1_idx, a2_idx)
                    if bond is not None:
                        covered_bond_idxs.add(bond.GetIdx())

        # uncovered_bond_idxs = set(range(self.num_bonds)) - covered_bond_idxs
        # from utils import show_2d_mol
        # show_2d_mol(self.mol, bonds=list(uncovered_bond_idxs))
        # print(covered_bond_idxs, len(covered_bond_idxs), self.num_bonds)

        assert len(covered_bond_idxs) == self.num_bonds, f'{self._idx} = {self.smiles} bonds not fully covered!'
            
    
    def show_tokenization(self, show_node=True, show_edge=True):
        if show_node:
            images = []
            for hyper_idx, hyper_node in enumerate(self.hyper_nodes):
                atom_ids = hyper_node.atom_ids
                bond_idxs = []
                for i in range(len(atom_ids)):
                    for j in range(i + 1, len(atom_ids)):
                        # should not use self.bonds or self.bond_smiles, because aromatic bonds are not included
                        bond = self.mol.GetBondBetweenAtoms(atom_ids[i], atom_ids[j])
                        if bond is not None:
                            bond_idxs.append(bond.GetIdx())

                legend = f'h-{hyper_idx}: {hyper_node.smiles}'
                if hyper_node.contracted_from is not None:
                    legend += f' <- {hyper_node.contracted_from}'
                img = get_2d_mol_image(self.mol, atoms=atom_ids, bonds=bond_idxs, legend=legend, size=(200, 200))
                images.append(img)
            concat_images_in_grid(images, molsPerRow=10, subImgSize=(200, 200))

        if show_edge:
            images = []
            for (h1_idx, h2_idx), hyper_pair in self.hyper_pairs.items():
                h1_h2_bond_idxs = []
                
                h1_atoms = set(self.hid_to_aids[h1_idx])
                h2_atoms = set(self.hid_to_aids[h2_idx])
                
                for a1_idx in sorted(h1_atoms - h2_atoms):
                    for a2_idx in sorted(h2_atoms - h1_atoms):
                        bond = self.mol.GetBondBetweenAtoms(a1_idx, a2_idx)
                        if bond is not None:
                            h1_h2_bond_idxs.append(bond.GetIdx())

                edge_type_str = hyper_pair.edge_type
                legend = f'h-{h1_idx} & h-{h2_idx} -> {edge_type_str}'
                img = get_2d_mol_image(self.mol, atoms=list(h1_atoms & h2_atoms), bonds=h1_h2_bond_idxs, legend=legend, size=(200, 200))
                images.append(img)
            concat_images_in_grid(images, molsPerRow=10, subImgSize=(200, 200))

if __name__ == '__main__':
    prefix = '/data/yanruqu2/MolBPE/data/lba'
    all_data = pickle.load(open(f'{prefix}/train_mols.pkl', 'rb'))
    for idx, mol in enumerate(all_data):
        smiles = Chem.MolToSmiles(mol)
        mol_graph = MolGraph(mol, idx, smiles)

        hyper_graph = HyperGraph.from_mol_graph(mol_graph)
        for h_id, h_info in enumerate(hyper_graph.hyper_nodes):
            print(h_id, h_info.smiles, h_info.atom_ids)
        
        for k, v in hyper_graph.hyper_pairs.items():
            print(k, v.smiles, v.atom_ids, v.h1_idx, v.h2_idx, v.h1_smiles, v.h2_smiles)
        break
