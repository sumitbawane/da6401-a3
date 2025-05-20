# Seq2Seq Transliteration with Attention

A PyTorch-based implementation of sequence-to-sequence transliteration models (vanilla and attention) using the Dakshina dataset for Hindi. This project covers data preprocessing, model definitions, training with Weights & Biases (W\&B) integration, and evaluation (including attention heatmaps).

---

## Project Overview

* **Goal**: Transliterate Latin-script words to Devanagari (Hindi) script.
* **Models**:

  * **Vanilla Seq2Seq** (`model_vanilla.py`)
  * **Attention-based Seq2Seq** (`model_attention.py`)
* **Scripts**:

  * **Preprocessing**: `PreProcess.py`
  * **Training**: `train_vanilla.py`, `train_attention.py`
  * **Evaluation**: `evaluate_vanilla.py`, `evaluate_attention.py`
* **Visualization**: Sample grids and attention heatmaps to inspect alignments.

---

## File Structure

```
├── PreProcess.py             # Data loading, vocab creation, tensor encoding
├── model_vanilla.py          # Vanilla Seq2Seq model
├── model_attention.py        # Seq2Seq model with attention
├── train_vanilla.py          # Training script (vanilla) with W&B
├── train_attention.py        # Training script (attention) with W&B
├── evaluate_vanilla.py       # Evaluation script (vanilla)
├── evaluate_attention.py     # Evaluation + heatmap visualization
├── best_model_*.pt           # Saved vanilla model weights 
├── best_attention_model_*.pt # Saved attention model weights 
└── dakshina_dataset_v1.0/     # External dataset directory
    └── hi/lexicons/*.tsv      # Translit data splits
```

**Download saved weights**: The pretrained model weights are large; you can download them from the following Drive link once available:

[Download model weights](your_drive_link_here)

---

## Requirements

* Python 3.6+
* PyTorch
* pandas
* numpy
* matplotlib
* seaborn
* tqdm
* wandb (Weights & Biases)

Install via:

```bash
pip install torch pandas numpy matplotlib seaborn tqdm wandb
```

---

## Installation

1. Clone this repository:

   ```bash
     git clone repo_url
   ```



````
2. Download the Dakshina dataset and place it under `dakshina_dataset_v1.0/`:
   ```bash
# should contain hi/lexicons/*.tsv files
````

3. Install dependencies (see Requirements).

---

## Usage

### 1. Data Preprocessing

```python
from PreProcess import PreProcess

pre = PreProcess(data_path='dakshina_dataset_v1.0', output_path='', language_code='hi')
pre.load_data()
pre.build_vocab()
train_data, test_data, val_data = pre.get_tensors()
```

### 2. Training

#### Vanilla Model

```bash
python train_vanilla.py
```

To run a W\&B hyperparameter sweep:

```bash
python train_vanilla.py --sweep --count 10
```

#### Attention Model

```bash
python train_attention.py
```

To run a W\&B sweep:

```bash
python train_attention.py --sweep --count 10
```

---

## Evaluation

#### Vanilla Model

```bash
python evaluate_vanilla.py
```

* Generates a CSV of predictions and prints a 3×3 sample grid.

#### Attention Model

```bash
python evaluate_attention.py
```

* Reports character/word accuracy and saves:

  * Individual heatmaps (`heatmaps/sample_*.png`)
  * 3×3 grid (`heatmaps/attention_grid_3x3.png`)
  * Position-indexed heatmaps

## links for 
best_model_*.pt           # Saved vanilla model weights 
best_attention_model_*.pt # Saved attention model weights
link1
link2
