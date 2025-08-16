# RecycleVision — Garbage Image Classification (6–12 classes)

A complete, modular deep-learning project that classifies garbage images into categories (e.g., plastic, metal, glass, paper, cardboard, organic/biological, batteries, clothes, shoes, green-glass, brown-glass, white-glass, trash).

## Quickstart

```bash
# (optional) create venv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# 1) Put/keep your dataset zip at the path shown (already uploaded in this workspace):
#    archive.zip
#    or download from Kaggle and place under data/raw/
#
# 2) Unzip + split into train/val/test (stratified)
python src/data_prep.py --zip_path "archive.zip" --out_dir data/processed --val_split 0.15 --test_split 0.15

# 3) (Optional) Run EDA – saves plots under reports/
python src/eda.py --data_dir data/processed/train

# 4) Train (transfer learning model: mobilenetv2 / resnet50 / efficientnetb0)
python src/train.py --data_dir data/processed --model mobilenetv2 --epochs 10 --batch_size 32 --fine_tune --save_format keras

# 5) Evaluate on test set (classification report + confusion matrix)
python src/evaluate.py --data_dir data/processed --weights artifacts/best_model.keras

# 6) Run the Streamlit app
streamlit run app.py
```

## Project Structure

```
RecycleVision/
├── app.py
├── artifacts/                # saved models, label map, training history
├── data/
│   ├── raw/                  # raw/unzipped dataset here
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
├── reports/                  # EDA charts, confusion matrix
├── src/
│   ├── config.py
│   ├── data_prep.py
│   ├── eda.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
└── requirements.txt
```

## Notes

- The pipeline reads class names directly from the folder names, so it works for both 6-class and 12-class versions (or any subset) as long as the dataset is organized in subfolders by class.
- Images are resized to 224×224, normalized, and augmented during training.
- We freeze the base of the transfer-learning backbone for a few epochs and optionally unfreeze top layers for fine-tuning.
- Metrics: Accuracy, Precision, Recall, F1; we also save a confusion matrix and misclassification report.
