from collections import Counter
import matplotlib.pyplot as plt
import pickle

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType as HT

import numpy as np

from tqdm import tqdm
from graph import HyperGraph, get_2d_mol_image, concat_images_in_grid, MolGraph
from typing import Union


def show_vocab_2d(vocab: Union[dict, Counter]):
    token_imgs = []
    items = vocab.most_common() if isinstance(vocab, Counter) else vocab.items()
    for k, v in items:
        mol = Chem.MolFromSmiles(k, sanitize=False)
        img = get_2d_mol_image(mol, legend=f'{k}: {v}', size=(150, 150))
        token_imgs.append(img)

    concat_images_in_grid(token_imgs, molsPerRow=10, subImgSize=(150, 150))


def plot_init_node_edge_num(hyper_graphs: list[HyperGraph]):
    mol_node_counter = Counter()
    mol_edge_counter = Counter()
    hyper_node_counter = Counter()
    hyper_edge_counter = Counter()
    mol_ratios = []
    hyper_ratios = []

    for hyper_g in hyper_graphs:
        mol_node_counter[hyper_g.num_atoms] += 1
        mol_edge_counter[hyper_g.num_bonds] += 1
        mol_ratios.append(hyper_g.num_bonds / hyper_g.num_atoms)
        hyper_node_counter[len(hyper_g.hyper_nodes)] += 1
        hyper_edge_counter[len(hyper_g.hyper_pairs)] += 1
        hyper_ratios.append(len(hyper_g.hyper_pairs) / len(hyper_g.hyper_nodes))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.bar(mol_node_counter.keys(), mol_node_counter.values(), label='mol G')
    ax1.bar(hyper_node_counter.keys(), hyper_node_counter.values(), label='hyper G')
    ax1.set_title('# nodes')
    ax1.legend()
    ax2.bar(mol_edge_counter.keys(), mol_edge_counter.values(), label='mol G')
    ax2.bar(hyper_edge_counter.keys(), hyper_edge_counter.values(), label='hyper G')
    ax2.set_title('# edges')
    ax2.legend()
    ax3.hist(mol_ratios, bins=100, alpha=0.5, label=f'mol G, avg = {np.mean(mol_ratios):.2f}')
    ax3.hist(hyper_ratios, bins=100, alpha=0.5, label=f'hyper G, avg = {np.mean(hyper_ratios):.2f}')
    ax3.set_title('# edges : # nodes')
    ax3.legend()
    plt.tight_layout()
    plt.show()


def count_atom_freq(mol_graphs: list[MolGraph]):
    atom_freq = Counter()
    for mol_g in mol_graphs:
        atom_freq.update(mol_g.atom_smiles)
    return atom_freq


def count_basic_token_freq(mol_graphs: list[MolGraph]):
    bond_freq = Counter()
    aromatic_ring_freq = Counter()
    non_aromatic_ring_freq = Counter()
    for mol_g in mol_graphs:
        bond_freq.update(mol_g.bond_smiles.values())
        aromatic_ring_freq.update(mol_g.aromatic_ring_smiles.values())
        non_aromatic_ring_freq.update(mol_g.non_aromatic_ring_smiles.values())
    return bond_freq, aromatic_ring_freq, non_aromatic_ring_freq


def count_hyper_node_freq(hyper_graphs: list[HyperGraph]):
    hyper_node_smiles_counter = Counter()
    for hyper_g in hyper_graphs:
        for hyper_node in hyper_g.hyper_nodes:
            hyper_node_smiles_counter[hyper_node.smiles] += 1
    return hyper_node_smiles_counter


def count_hyper_pair_freq(hyper_graphs: list[HyperGraph]):
    hyper_pair_smiles_counter = Counter()
    for hyper_g in hyper_graphs:
        for hyper_pair in hyper_g.hyper_pairs.values():
            hyper_pair_smiles_counter[hyper_pair.smiles] += 1
    return hyper_pair_smiles_counter
