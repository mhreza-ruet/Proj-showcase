## VOC Sensing Dye Detection Models

This project explores deep generative modeling approaches for analyzing dye–VOC interactions, with the goal of advancing data-driven cheminformatics for chemical sensing applications. I implemented and compared three architectures:  

- **VAE (Vanilla Variational Autoencoder)**  
- **LTVAE (LSTM-based VAE)**  
- **Trans-VAE (Transformer-based VAE)**  

All models were trained, validated, and tested on the same curated dataset to ensure a fair comparison. While their overall performance metrics were similar, the models varied significantly in terms of architecture design and number of trainable parameters, highlighting trade-offs between complexity and efficiency.  

### Key Features
- Implemented **custom SMILES tokenization** for sequence modeling.  
- Built models in **PyTorch** using **VAE, Bidirectional LSTM, and Transformer** layers.  
- Deployed training on a **remote VM with dual NVIDIA A100 GPUs**, gaining hands-on experience in distributed deep learning.  
- Developed reproducible workflows for model training, validation, and evaluation.  

### Skills Demonstrated
- Generative modeling (VAE, LSTM, Transformer)  
- Sequence tokenization for chemical data  
- High-performance computing on GPU clusters  
- Model architecture design and comparison  
- Research workflow development in cheminformatics
