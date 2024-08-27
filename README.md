
# Misogyny Detection with BERT

This project utilizes the BERT (Bidirectional Encoder Representations from Transformers) model to detect misogynistic content in text data. It's specifically trained on the MIMIC2024 dataset, which includes labels for Misogyny, Objectification, Prejudice, and Humiliation. 

## Dataset

The MIMIC2024 dataset is used for training and evaluation. It consists of text extracted from images, along with binary labels indicating the presence of the aforementioned categories of harmful content.

## Model

The BERT base multilingual cased model is employed for sequence classification. It's fine-tuned on the MIMIC2024 dataset for each label category separately:

- Misogyny
- Objectification
- Prejudice
- Humiliation

## Requirements

- Python 3.7+
- pandas
- datasets
- transformers
- scikit-learn
- torch

You can install the necessary libraries using:

```bash
pip install pandas datasets transformers scikit-learn torch
