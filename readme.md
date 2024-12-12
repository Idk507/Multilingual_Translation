# Tamil-English Translation Model

This repository contains a complete pipeline for training and deploying a Tamil-to-English translation model using Hugging Face's `transformers` library and the Helsinki-NLP pre-trained translation models.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Preparation](#dataset-preparation)
3. [Model Training](#model-training)
4. [Evaluation and Metrics](#evaluation-and-metrics)
5. [Deployment](#deployment)
6. [Usage](#usage)
7. [Requirements](#requirements)
8. [Setup Instructions](#setup-instructions)
9. [References](#references)

---

## Overview

The project demonstrates a full workflow for creating a Tamil-to-English translation model:

1. **Dataset Preparation:** Preprocessing and aligning Tamil-English sentence pairs.
2. **Model Training:** Using the Helsinki-NLP/opus-mt pre-trained models.
3. **Evaluation:** Computing BLEU scores to evaluate model performance.
4. **Deployment:** Saving and using the trained model for translations.

---

## Dataset Preparation

1. **Dataset Location:** Place the dataset in the `Tamil-English-Dataset-master` folder.
2. **Structure:**
   - Tamil sentence files start with `data.ta`.
   - English sentence files start with `data.en`.
3. **Steps:**
   - Merge all Tamil and English files into `merged_tamil.txt` and `merged_english.txt`.
   - Align the files to ensure each Tamil sentence maps correctly to an English sentence.
   - Split the dataset into training, validation, and testing sets.

   ```python
   random.shuffle(data)
   train_size = int(0.8 * len(data))
   val_size = int(0.1 * len(data))
   ```

4. **Outputs:**
   - Training files: `train_tamil.txt`, `train_english.txt`
   - Validation files: `val_tamil.txt`, `val_english.txt`
   - Test files: `test_tamil.txt`, `test_english.txt`

---

## Model Training

1. **Pre-trained Model:** `Helsinki-NLP/opus-mt-ta-en`
2. **Training Parameters:**
   - Batch Size: 8
   - Learning Rate: 2e-5
   - Epochs: 3
   - Device: GPU (if available)

3. **Tokenization:**
   Tokenize inputs (Tamil) and outputs (English) with truncation and padding to a maximum length of 128 tokens.

   ```python
   tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ta-en")
   ```

4. **Training Arguments:**
   ```python
   Seq2SeqTrainingArguments(
       output_dir="./tamil_english_translation_model",
       evaluation_strategy="epoch",
       save_strategy="epoch",
       learning_rate=2e-5,
       per_device_train_batch_size=8,
       num_train_epochs=3,
   )
   ```

5. **Trainer:**
   Use the `Seq2SeqTrainer` class with a BLEU score metric.

---

## Evaluation and Metrics

- **Metric Used:** BLEU score (via `evaluate` library).
- **Computation:**

  ```python
  metric = evaluate.load("sacrebleu")
  result = metric.compute(predictions=decoded_preds, references=[[label]])
  ```

---

## Deployment

1. Save the trained model:
   ```python
   trainer.save_model("./tamil_english_translation_model")
   ```

2. **Translation Model Class:**
   A Python class for performing single and batch translations using the trained model.

   ```python
   class TranslationModel:
       def __init__(self, model_path):
           self.tokenizer = AutoTokenizer.from_pretrained(model_path)
           self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
   ```

---

## Usage

### Single Sentence Translation

```python
translator = TranslationModel("./tamil_english_translation_model")
tamil_text = "வணக்கம் உலகம்!"
english_translation = translator.translate(tamil_text)
print(english_translation)
```

### Batch Translation

```python
tamil_texts = ["வணக்கம் உலகம்!", "தமிழ் மொழி அருமையான மொழி"]
batch_translations = translator.batch_translate(tamil_texts)
```

---

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Datasets
- Evaluate

Install requirements:

```bash
pip install torch transformers datasets evaluate
```

---

## Setup Instructions

1. Clone the repository and navigate to the directory.
2. Place the dataset in the appropriate folder structure.
3. Run the preprocessing script to align and split the dataset.
4. Train the model with the training script.
5. Use the saved model for inference.

---

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Helsinki-NLP Models](https://huggingface.co/Helsinki-NLP)
- [SacreBLEU Documentation](https://github.com/mjpost/sacrebleu)

---

For questions or contributions, feel free to reach out!

To access the complete code : https://drive.google.com/drive/folders/1iXGSrFQXHBWfU6E3Uaz-3auSduUzPML-?usp=sharing
