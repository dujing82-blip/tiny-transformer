# Tiny Transformer — GPT-2 Mini + Medical LoRA

Educational TensorFlow / Keras notebook based on the original **GPT2_Mini_Chat.ipynb** (8 layers, 8 attention heads, embedding dimension 512, context 256, GPT-2 BPE).

## Notebooks

- [GPT2_Mini_Chat.ipynb](GPT2_Mini_Chat.ipynb): the original SODA pretraining code, preserved for reference.
- [GPT2_Mini_MedDialog_LoRA.ipynb](GPT2_Mini_MedDialog_LoRA.ipynb): loads the existing GPT weights, freezes the base, adds Q/V LoRA, trains on short MedDialog patient/doctor turns, plots train/validation loss, compares LoRA ON/OFF and saves only the adapter.

**Open in Colab:** https://colab.research.google.com/github/dujing82-blip/tiny-transformer/blob/main/GPT2_Mini_MedDialog_LoRA.ipynb

## Getting started

1. Use a GPU Colab session and run cells in order.
2. Keep your original `gpt_chat.weights.h5` in Google Drive. The notebook expects `MyDrive/Colab Notebooks/Teaching/Trained_Models/gpt_chat.weights.h5`; edit `BASE_PATH` if yours differs. GitHub does **not** contain the large trained weights.
3. The training examples come from the Hugging Face dataset [OpenMed/MedDialog](https://huggingface.co/datasets/OpenMed/MedDialog), streamed and restricted to full conversations fitting 256 tokens. No medical data is committed here.
4. First try 500–3,000 training examples; increase if GPU time permits. Change `RANK` to study low-rank adaptation.

## What students learn

A trained base model can be reused for multiple domains without copying or retraining its full weights. LoRA trains tiny Q/V matrices of rank `r`; the adapter can be toggled on/off, saved separately and compared against the original model.

## Important limitations

This is a classroom demo, **not a medical advice system**. Generated medical text can be wrong or unsafe. Do not use for diagnosis, treatment, triage, or on real patient records. A low language-modeling loss does not imply clinical accuracy. Reproduction requires the **original base checkpoint** and compatible TensorFlow/Keras version. Dataset schema and availability may change.

The notebook contains a numerical sanity check that the frozen standard Keras attention and the custom Q/V LoRA wrapper agree before training; if the check fails on a future Keras release, stop and inspect the implementation rather than training with mismatched attention.
