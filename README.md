# Multimodal Fallacy Detection in Political Debates

Detecting and classifying argumentative fallacies in political speech using fused audio and text signals.

---

## Overview

Political debates are full of logical fallacies — arguments that sound convincing but don't hold up under scrutiny. This project tackles the problem of **automatically detecting and classifying these fallacies** using a multimodal deep learning approach that combines what is *said* (text) with *how* it is said (audio).

Built on the **MM-USED-fallacy dataset** from the SemEval shared task, this project addresses two tasks:

- **Task A — Fallacy Detection:** Binary classification — does this utterance contain a fallacy or not?
- **Task B — Fallacy Classification:** 6-class classification — which type of fallacy is it?

---

## Results

| Task | Modality | F1 Score |
|---|---|---|
| Task A — Detection | Multimodal (Text + Audio) | **0.90** |
| Task B — Classification | Multimodal (Text + Audio) | **0.65** |


The multimodal fusion consistently outperformed text-only baselines on both tasks, demonstrating that audio signals carry meaningful information beyond the transcript alone.

---

## Architecture
```
Audio Input ──► WavLM (base) ──► Audio Embeddings ──┐
                                                      ├──► Fusion Layer ──► Classifier
Text Input  ──► MM-RoBERTa   ──► Text Embeddings  ──┘
```

- **Text encoder:** MM-RoBERTa — a multimodal-aware variant of RoBERTa fine-tuned for debate understanding
- **Audio encoder:** WavLM — a self-supervised speech model from Microsoft that captures acoustic and prosodic features
- **Fusion:** Concatenation of pooled embeddings from both modalities, passed through a classification head
- **Training:** Fine-tuned end-to-end on the MM-USED-fallacy dataset with cross-entropy loss

---

## Dataset

**MM-USED-fallacy** — a multimodal dataset of political debate utterances annotated for argumentative fallacies. Each sample contains:
- Raw audio clip
- Transcript text
- Fallacy label (binary for Task A, 6-class for Task B)

The 6 fallacy classes in Task B include: Appeal to Emotion, False Cause, Slippery Slope, Slogans, Ad Hominem, and others.

---

## Repository Structure
```
├── Multimodal_(Text_+_Audio)_for_Tasks_A(final).ipynb   # Full multimodal pipeline — Task A
├── Multimodal_(Text_+_Audio)_for_Tasks_B(final).ipynb   # Full multimodal pipeline — Task B
├── Task_A_(Text_only_Fallacy_Detection).ipynb            # Text-only baseline — Task A
├── Task_B_Fallacy_Classification_(Text_only).ipynb       # Text-only baseline — Task B
├── MultimodelTesting.ipynb                               # Evaluation and testing notebook
└── Testing_Text.ipynb                                    # Text baseline testing
```

---

## Setup & Requirements
```bash
pip install transformers torch torchaudio datasets scikit-learn
```

Models used:
- `microsoft/wavlm-base` — audio encoder
- `roberta-base` (MM-RoBERTa variant) — text encoder

Run on **Google Colab** with GPU runtime for best performance.

---

## Key Takeaways

- Audio features (tone, prosody, pacing) add measurable signal to fallacy detection beyond text alone
- WavLM + RoBERTa fusion is a strong baseline for multimodal argument mining
- The gap between Task A (0.90) and Task B (0.65) reflects the inherent difficulty of fine-grained fallacy classification vs. binary detection

---

## Tech Stack

`Python` `PyTorch` `HuggingFace Transformers` `WavLM` `RoBERTa` `torchaudio` `scikit-learn` `Google Colab`

---

## Author

**Syeda Azka Qadir** — MS AI @ NUST Islamabad  
**Irsa Aziz** — MS AI @ NUST Islamabad  
**Roshni** — MS AI @ NUST Islamabad  
