# hmint

## Setup

### Environment

We follow the environment of GET
```bash
mamba env create -f env.yml
conda activate hmint
python -c "import rdkit; print(rdkit.__version__)"
```

### Datasets

#### Binding Affinity Prediction

Prerequisite: preprocessing ``PDBBind`` and ``LBA`` depends on tokenization vocabulary. For your convenience, we provide the vocabulary (built from LBA training set) at ``OverlapBPE/data/lba/vocab_213.pkl``.

```bash
cp ./OverlapBPE/data/lba/vocab_213.pkl ./data/tokenizer/vocabs/vocab_213.pkl
```

1. PDBBind Benchmark (established splits, following GET)

Download raw dataset (1.9G).
```bash
mkdir -p ./datasets/PDBBind
wget "https://zenodo.org/record/8102783/files/pdbbind_raw.tar.gz?download=1" -O ./datasets/PDBBind/pdbbind_raw.tar.gz
tar zxvf ./datasets/PDBBind/pdbbind_raw.tar.gz -C ./datasets/PDBBind
rm ./datasets/PDBBind/pdbbind_raw.tar.gz
```

```bash
python scripts/data_process/process_PDBbind_benchmark.py \
    --benchmark_dir ./datasets/PDBBind/pdbbind \
    --fragment VOCAB_213 \
    --out_dir ./datasets/PDBBind/processed_vocab_213
```

2. Ligand Binding Affinity (LBA, following GET)

Download raw dataset (540M).
```bash
mkdir ./datasets/LBA
wget "https://zenodo.org/record/4914718/files/LBA-split-by-sequence-identity-30.tar.gz?download=1" -O ./datasets/LBA/LBA-split-by-sequence-identity-30.tar.gz
tar zxvf ./datasets/LBA/LBA-split-by-sequence-identity-30.tar.gz -C ./datasets/LBA
rm ./datasets/LBA/LBA-split-by-sequence-identity-30.tar.gz
```

3. Build your own vocabulary from a set of molecules.

We provide a notebook ``OverlapBPE/build_vocab_lba.ipynb`` and ``OverlapBPE/data/lba/train_mols.pkl`` to illustrate the complete process. Simply run all cells to produce the results. See detailed instructions in notebook about how to customize your own vocabulary.

Note: important parameters include: data path, ISOMERIC, MAX_VOCAB_SIZE, min_freq.

### Experiments

### Reference
