## ChemInfo-LatentRep

# LSTM–Transformer–VAE (LTVAE) for Molecular SMILES

This project implements a hybrid Variational Autoencoder (**LTVAE**) for molecular SMILES strings, using an **Bi-directional LSTM encoder** followed by **VAE reparameterization trick** and a **Transformer decoder**. It supports teacher-forcing training, greedy and beam search decoding, and standard molecular metrics (validity, Levenshtein distance, reconstruction accuracy).

---

## Project Structure

- **`data_utils.py`**  
  Utilities for SMILES tokenization, vocabulary building, and data augmentation (randomized SMILES).

- **`dataset.py`**  
  Dataset wrapper for tokenized SMILES. Handles padding, special tokens, and augmentation on the fly.

- **`model_bs.py`**  
  Model definitions:
  - Encoder (BiLSTM)
  - Transformer/GRU decoder
  - VAE wrapper (`LSTM_VAE_Trans`)
  - Beam search decoding

- **`train.py`**  
  Training loop with:
  - Teacher forcing  
  - KL annealing  
  - Validation with greedy decoding every epoch  
  - Beam search evaluation every N epochs  
  - Early stopping + LR scheduling

- **`metrics.py`**  
  Loss functions (reconstruction + KL), token accuracy, chemical validity (RDKit), Levenshtein distance.

- **`inference.py`**  
  Utilities for reconstruction and evaluation of trained models. Includes conversion from token tensors back to SMILES strings.

- **`environment.yml`**  
  Conda environment with Python, PyTorch, RDKit, and dependencies to reproduce results.

---

## How to Run

1. **Set up environment**
   ```bash
   conda env create -f environment.yml
   conda activate cheminf

2. **Prepare data**  
Put your training, validation, and test CSV files under `Data/`.  
Each file should have a column named `smiles`.

3. **Update config**  
Edit `cfg` in `main.ipynb` notebook.

4. **Train**  
Train the model by running:
   ```python
   model, history = run_training(cfg, token_to_idx, idx_to_token)

5. **Inference**
   Evaluate the model by running:
   ```python
   from inference import reconstruct_smiles_table
   df_rec = reconstruct_smiles_table(test_csv="Data/Test.csv", model=model,
                                  token_to_idx=token_to_idx, idx_to_token=idx_to_token,
                                  seq_length=cfg["seq_length"],
                                  pad_idx=cfg["pad_idx"], sos_idx=cfg["sos_idx"], eos_idx=cfg["eos_idx"],
                                  device="cuda", mode="beam", beam_size=cfg["beam_size"])
   print(df_rec.head())

### Reproducibility
- Training/validation splits are fixed by the provided CSVs.
-	Vocabulary is cached in vocab.json for consistency.
-	All configs (cfg) are centralized so experiments can be rerun easily with different hyperparameters.
