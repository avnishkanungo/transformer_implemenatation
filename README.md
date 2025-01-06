# Transformer Implementation for Machine Translation

## Overview
This project implements a Transformer model for machine translation using PyTorch. The Transformer is a state-of-the-art neural network architecture introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017.

## Features
- Implementation of the Transformer model for machine translation
- Supports training and inference on bilingual datasets
- Uses PyTorch for efficient computation and automatic differentiation
- Includes data preprocessing and tokenization using the Hugging Face Tokenizers library
- Supports greedy decoding for generating translations

## Requirements
- Python 3.8+
- PyTorch 1.9+
- Hugging Face Tokenizers library
- `datasets` library for loading and manipulating datasets

## Training and Inference
To train the model on a bilingual dataset, follow these steps: 
1. Download the dataset using the `datasets` library.
2. Load the dataset using the `BilingualDataset` class. This class preprocesses the data and tokenizes it using the Hugging Face Tokenizers library.
3. Split the dataset into training and validation sets.
4. Create a DataLoader for the training and validation sets.
5. Train the model using the `train` function.
6. Evaluate the model on the validation set using the `evaluate` function.
7. Use the model for inference by calling the `greedy_decode` function.

## Model Architecture
The Transformer model implemented in this project consists of the following components:
1. Encoder: takes in source language input and outputs a continuous representation
2. Decoder: takes in the encoder output and generates target language output
3. Attention mechanism: allows the model to focus on different parts of the input sequence when generating output

## Dataset
This project uses the OPUS Books dataset, a large-scale bilingual dataset for machine translation. You can download the dataset using the datasets library.

## Tokenization
This project uses the Hugging Face Tokenizers library for tokenization, we have utilized the ByteLevelBPETokenizer. You can customize the tokenization process by modifying the get_build_tokenizer function in tokenizers.py.

## Greedy Decoding
This project implements greedy decoding for generating translations, by picking the token with the highest probability at each step. You can modify the decoding process by modifying the greedy_decode function in inference.py.

## Acknowledgements
This project is based on the Transformer implementation by [Hugging Face](https://huggingface.co/transformers/) and Umar Jamil's [Machine Translation with PyTorch](https://github.com/umarjamil/machine-translation-pytorch).
