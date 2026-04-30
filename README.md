
# Dermatology Classification Model with MLP

## 📌About the project
This project implements a complete Machine Learning pipeline for the classification of dermatological diseases using a Multi-Layer Perceptron (MLP) architecture.

The goal is to build a reproducible training, validation, and monitoring flow using MLOps best practices.




## 📚Tools Used

This project is used by the following companies:

- Python 3.14.3
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Weights & Biases (W&B)


## 📂Project Struture
```bash
projeto_avaliacao/
├── data/
│   ├── raw/
│   └── processed/
│
├── figures/
│
├── models/
│   └── best_model_final.pt
│
├── notebooks/
│   └── projeto.ipynb
│
├── src/
│   ├── data_loading.py
│   ├── pre_processing.py
│   ├── feature_engineering.py
│   ├── split.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── .env
```
## 🔄Project Pipeline

The pipeline follows these steps:

- Data Collection
- Data Cleaning
- Handling Missing Values
- Feature Engineering
- Feature Selection
- Train/Test Split
- Model Training
- Hyperparameter Tuning (W&B Sweeps)
- Model Evaluation
- Experiment Monitoring
## ⚙️Installation


### 1. Clone the repository

```bash
git clone https://github.com/rayssacoelho/learning_ai.git
cd learning_ai
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate environment

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a file `.env`:

```env
WANDB_API_KEY=your_key
```

### 6. Perform training

```bash
python notebook/projeto.ipynb
```
## 📈 Monitoring with W&B

All experiments are tracked in Weights & Biases:

🔗 Project link: [W&B Dashboard](https://wandb.ai/raycoelho-ufrn/mlops-project-dermatology?nw=nwuserraycoelho)## 📊 Results

### Data Distribution

![EDA](./figures/histograms_features.png)

### Feature Ranking

![Features](./figures/feature_ranking.png)

### Correlation Matrix

![Correlation](./figures/correlation_matrix.png)

### Training Curves in Sweep

![Training](./figures/training_sweep.png)

## 🎯 Final Metrics

- Accuracy: 83,7%
- Precision: 84,4%
- Recall: 83,7%
- F1-Score: 83,3%

## 🔍 Key Learnings

- Building reproducible pipelines
- Experiment monitoring
- Hyperparameter optimization
- MLOps best practices

## 👨‍💻 Autor

**Rayssa Coelho Silva**
## Optimizations

What optimizations did you make in your code? E.g. refactors, performance improvements, accessibility

