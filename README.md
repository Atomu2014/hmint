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

```bash
cp ./OverlapBPE/data/lba/vocab_213.pkl ./data/tokenizer/vocabs/vocab_213.pkl
```

### Experiments

### Reference
