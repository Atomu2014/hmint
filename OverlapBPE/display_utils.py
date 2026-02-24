from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType as HT

from IPython.display import display
from PIL import Image
import py3Dmol

import matplotlib.pyplot as plt


def get_2d_mol_image(mol, kekulize=False, legend=None, atoms=[], bonds=[], bond_colors={}, size=(300, 300)):
    img = Draw.MolToImage(mol, kekulize=kekulize, legend=legend, highlightAtoms=atoms, highlightBonds=bonds, highlightBondColors=bond_colors, size=size)
    return img


def show_2d_mol(mol, kekulize=False, legend=None, atoms=[], bonds=[], bond_colors=[]):
    img = get_2d_mol_image(mol, kekulize=kekulize, legend=legend, atoms=atoms, bonds=bonds, bond_colors=bond_colors)
    display(img)


def show_2d_smiles(smiles, kekulize=False, legend=None):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    show_2d_mol(mol, kekulize=kekulize, legend=legend)


def show_counter_topk(counter, topk=0):
    freq = counter.most_common()
    for k, v in freq[:topk] if topk > 0 else freq:
        print(k, v)


def show_counter_firstk(counter, firstk=0):
    cnt = 0
    for k, v in counter.items():
        print(k, v)
        cnt += 1
        if cnt == firstk:
            break


def plot_counter(counter, title=''):
    freq = [(k, v) for k, v in sorted(counter.items(), key=lambda x: x[1], reverse=True)]
    x, y = zip(*freq)
    plt.bar(x, y)
    plt.title(title)
    plt.show()


def show_2d_smiles_grid(smiles_list, legends=None, molsPerRow=10, subImgSize=(150, 150)):
    mols = [Chem.MolFromSmiles(smiles, sanitize=False) for smiles in smiles_list]

    img_width, img_height = subImgSize
    mol_images = [Draw.MolToImage(mol, kekulize=False, legend=legend, size=subImgSize) for mol, legend in zip(mols, legends)]
    rows = (len(mols) + molsPerRow - 1) // molsPerRow  # 总行数
    cols = molsPerRow  # 每行分子数量

    grid_image = Image.new('RGB', (cols * img_width, rows * img_height))
    for i, img in enumerate(mol_images):
        x = (i % cols) * img_width
        y = (i // cols) * img_height
        grid_image.paste(img, (x, y))
    
    display(grid_image)


def get_num_of_heavy_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    return sum([atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()])


def display_ring_by_size(ring_counter):
    ring_by_size = {}

    # legends = []
    for k, v in ring_counter.items():
        num_of_heavy_atoms = get_num_of_heavy_atoms(k)
        if num_of_heavy_atoms not in ring_by_size:
            ring_by_size[num_of_heavy_atoms] = []
        ring_by_size[num_of_heavy_atoms].append((k, v))

    for k in sorted(ring_by_size.keys()):
        v = ring_by_size[k]
        print(f'# {k}-ring aromatic', len(v), 'total freq', sum([x[1] for x in v]))

        ring_smiles = []
        ring_freq = []
        for _k, _v in v:
            ring_smiles.append(_k)
            ring_freq.append(_v)
        ring_legends = [f'{k} {v}' for k, v in zip(ring_smiles, ring_freq)]
        show_2d_smiles_grid(ring_smiles, legends=ring_legends)


def concat_images_in_grid(images, molsPerRow=10, subImgSize=(150, 150)):
    img_width, img_height = subImgSize
    rows = (len(images) + molsPerRow - 1) // molsPerRow  # 总行数
    cols = molsPerRow  # 每行分子数量

    grid_image = Image.new('RGB', (cols * img_width, rows * img_height))
    for i, img in enumerate(images):
        x = (i % cols) * img_width
        y = (i // cols) * img_height
        grid_image.paste(img, (x, y))
    
    display(grid_image)


def show_3d_mol(mol):
    block = Chem.MolToMolBlock(mol)

    view = py3Dmol.view(width=300, height=300)
    view.addModel(block, "mol")
    view.setStyle({"stick": {}})
    view.zoomTo()
    view.show()


def show_vocab_brief(vocab, tail_only=False):
    if len(vocab) < 6:
        for k, v in vocab.items():
            print(k, v)
    else:
        items = list(vocab.items())
        if not tail_only:
            for k, v in items[:3]:
                print(k, v)
        print('...')
        for k, v in items[-3:]:
            print(k, v)


def is_aromatic_ring(mol, ring):
    # TODO: add ring check
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
