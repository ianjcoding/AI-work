README.md
🥗 Semi-Supervised Food Insecurity Prediction
Using Machine Learning to Support SDG 2: Zero Hunger
📌 Project Overview
Food insecurity remains one of Africa’s biggest challenges, especially in rural areas where real-time data is limited. Many regions lack labeled datasets that clearly classify households as “Food Secure” or “At Risk”.

This project uses semi-supervised machine learning to solve that challenge.

The algorithm learns from:

✔ A small set of labeled data (regions where food insecurity status is known)

✔ A large set of unlabeled data (regions with no labels)

By using Label Spreading + Self-Training with Random Forest, the model is able to:

🟢 Predict food insecurity risk for unlabeled regions

🟢 Improve performance using pseudo-labels

🟢 Provide a scalable tool for governments & NGOs

This project directly supports SDG 2: Zero Hunger, especially target 2.1 — End hunger and ensure access to safe, nutritious food for all.

📂 Project Structure
Ai folder/
│
├── semi_supervised_food_insecurity.py   # Main ML script
│
├── data/                                # Auto-generated dataset output
│     └── regions_features.csv
│
├── models/                              # Saved model pipeline (.joblib)
│     └── semi_supervised_pipeline.joblib
│
├── screenshots/                         # Screenshots of demo output
│     └── screenshot1.png
│     └── screenshot2.png
│
└── README.md
⚙️ Installation Requirements
Install dependencies using:

pip install scikit-learn pandas numpy matplotlib joblib
Required Libraries:

scikit-learn

pandas

numpy

matplotlib

joblib

▶️ How to Run the Project
Open a terminal inside the project folder:

cd "Ai folder"
Run the script:

python semi_supervised_food_insecurity.py
You should see output similar to:

Generated sample dataset at: data/regions_features.csv
Evaluation on test set: {'accuracy': ..., 'f1': ..., 'precision': ..., 'recall': ..., 'roc_auc': ...}
📊 Model Workflow
1. Generate a Synthetic Dataset
Because real food insecurity datasets are limited, the script auto-creates a fake dataset:

200 regions

10 random features

40% unlabeled (simulated missing labels)

2. Preprocessing
Missing values imputed

Features scaled

Labels split into:

Labeled set (train/test)

Unlabeled set

3. Label Spreading
Uses RBF kernel to generate pseudo-labels for unlabeled regions.

4. Self-Training Random Forest
A Random Forest model is trained on:

Real labels

Confident pseudo labels (> 85% confidence)

5. Evaluation
Model is evaluated on unseen labeled regions.

6. Save Model Pipeline
Saved to:

models/semi_supervised_pipeline.joblib
🖼️ Screenshots
Sample Output Screenshot
Add your screenshot to:

screenshots/
Then reference it like this:

<img width="1920" height="1080" alt="Screenshot (326)" src="https://github.com/user-attachments/assets/6d00f5ae-5fa6-4ac6-85ce-923e2b51228d" />

Example:

🎯 How This Project Supports SDG 2 (Zero Hunger)
This model helps governments, NGOs, and researchers:

✔ Identify at-risk regions early
✔ Predict food insecurity without full datasets
✔ Reduce the cost of large surveys
✔ Focus resources on areas that need it most
By combining semi-supervised learning and synthetic data generation, this project demonstrates a scalable approach to improving food security analytics across developing regions.

📌 How to Add Screenshots to README
<img width="1581" height="113" alt="Screenshot 2025-11-23 123430" src="https://github.com/user-attachments/assets/08b20ea2-9f3e-4b46-a0e4-7d13a97f7b2a" />
<img width="1919" height="557" alt="Screenshot 2025-11-23 194632" src="https://github.com/user-attachments/assets/e8c7fffe-d9e6-4449-993a-66fa24ea26e5" />
<img width="1280" height="71" alt="Screenshot 2025-11-23 195347" src="https://github.com/user-attachments/assets/7cc03f76-792f-460b-8d57-7999607e547f" />

