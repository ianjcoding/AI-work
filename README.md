Using Semi-Supervised Machine Learning to Predict Food Insecurity Risk (SDG 2: Zero Hunger)
By [James K M]
1. Introduction — The SDG Problem Being Solved

This project addresses Sustainable Development Goal 2 (SDG 2): Zero Hunger, specifically the challenge of identifying regions at risk of food insecurity.

Governments and NGOs struggle because:

Field surveys are expensive

Ground-truth labels (“At risk” vs “Not at risk”) are limited

However, satellite, climate, economic, and geographic data are abundant but unlabeled

This creates the perfect opportunity for semi-supervised machine learning — a method that can learn effectively from a small amount of labeled data and a large amount of unlabeled data.

This project demonstrates exactly that.

**2. What the Project Does**

This project builds a semi-supervised machine learning pipeline that:

Loads a per-region dataset

Uses Label Spreading to infer pseudo-labels for unlabeled regions

Selects high-confidence pseudo-labels

Trains a RandomForestClassifier using real + pseudo labels

Evaluates performance on a clean test set

Saves the complete pipeline using joblib

The system ultimately produces a model that can predict whether a region is “At Risk” or “Not at Risk” of food insecurity even with limited real labels.

This is extremely valuable in real-world hunger-monitoring systems, especially across rural African counties and districts.

**3. Dataset Generation (As Implemented in the Code)**

To allow the project to run anywhere without requiring real government datasets, the code automatically generates a fake dataset:

num_regions = 200  
num_features = 10  
labels include: 0 = Not at Risk, 1 = At Risk, NaN = Unlabeled (40%)


The script then saves this dataset to:

data/regions_features.csv


This simulates what real data would look like while avoiding the need to expose sensitive government files.

**4. Semi-Supervised Learning Pipeline (Matches Code)**  
Step 1 — Preprocessing

Median imputation using SimpleImputer

Standardization using StandardScaler

Matching lines in the code:

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

Step 2 — Separate Labeled vs Unlabeled

Your code splits data like this:

Labeled → rows where label is 0 or 1

Unlabeled → rows where label = NaN

Then it creates a small labeled set (approximately 10%) to simulate real-world scarcity, which equals:

n_labeled_small = max(20, 10% of labeled data)


This is excellent and realistic.

Step 3 — Label Spreading (Semi-Supervised Learning)

The heart of the project:

label_spread = LabelSpreading(kernel='rbf', alpha=0.2, max_iter=1000)
label_spread.fit(X_pool, y_pool)


This model spreads labels from the few labeled regions across the feature space to infer new ones.

Step 4 — High-Confidence Pseudo Labels

The code selects only pseudo-labels where:

max probability > 0.85


This ensures the model doesn’t learn from noisy or uncertain guesses.

Step 5 — Supervised Model Training

Using both real and confident pseudo-labels:

clf = RandomForestClassifier(n_estimators=200)
clf.fit(X_self_train, y_self_train)


This creates a strong supervised model on expanded data.

Step 6 — Evaluation

The model is evaluated on a pure labeled test set:

metrics = {
    accuracy, f1, precision, recall, roc_auc
}


This ensures your final performance score is honest and not inflated by unlabeled data.

Step 7 — Saving the Pipeline

The following dictionary is saved:

{
    "imputer": imputer,
    "scaler": scaler,
    "label_spread": label_spread,
    "classifier": clf
}


Stored in:

models/semi_supervised_pipeline.joblib


This makes your model usable for deployment or future predictions.

5. How This Solves the SDG 2 Problem
✔ Predicts food insecurity early

Regions that show rising risk can be targeted with interventions before extreme hunger occurs.

✔ Works even with limited survey data

Agricultural ministries rarely have enough labeled samples — your approach solves this bottleneck.

✔ Reduces cost of monitoring

Satellite + environmental data is cheap; your ML system turns it into actionable insights.

✔ Enables data-driven decision-making

Counties, NGOs, and global partners can allocate food aid more efficiently.

✔ Fully automatable

Once deployed, predictions can run weekly or monthly with new data.

6. Conclusion

This project demonstrates how semi-supervised machine learning can strengthen hunger early-warning systems, especially in regions where labeled food security assessments are limited. By combining Label Spreading and Random Forests, the model becomes more accurate while relying on minimal labeled data.

This approach directly contributes to achieving SDG 2: Zero Hunger, and shows how artificial intelligence can be used responsibly to support vulnerable communities.


**1. Screenshot of the Folder Structure**

<img width="1920" height="1080" alt="Screenshot (324)" src="https://github.com/user-attachments/assets/5cd74caa-380d-4863-a135-90b01bfdb2f5" />


**2. Screenshot of the “Dataset Generated” Output**

<img width="1280" height="71" alt="Screenshot 2025-11-23 195347" src="https://github.com/user-attachments/assets/fff4dcad-8349-4550-a016-0ea889b37fd6" />

**3. Screenshot of the Model Training Logs**

<img width="1581" height="113" alt="Screenshot 2025-11-23 123430" src="https://github.com/user-attachments/assets/f3cac64e-319f-4287-a401-a7d25ca4e3f5" />

**4. Screenshot of the CSV File Preview**

<img width="1919" height="557" alt="Screenshot 2025-11-23 194632" src="https://github.com/user-attachments/assets/c9316f39-35cf-4e18-97ef-c6c632fe0de0" />

**5. Screenshot of the Models Folder**

Showing:

semi_supervised_pipeline.joblib
