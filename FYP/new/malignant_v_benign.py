import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ✅ LOAD THE DATASETS
# Load S11 Data
malignant_s11 = pd.read_csv("prev/data/s11_14ghz_malignant_tumor.csv")
benign_s11 = pd.read_csv("prev/data/s11_14ghz_benign_tumor.csv")
print(malignant_s11.head())
print("*************************")
print(benign_s11.head())
print("*************************")

# Load Phase Data
malignant_phase = pd.read_excel("prev/data/s11_14ghz_malignant_phases.xlsx")
benign_phase = pd.read_excel("prev/data/s11_14ghz_benigntumor_phases (1).xlsx")
print(malignant_phase.head())
print("*************************")
print(benign_phase.head())
print("*************************")


# ✅ ADD LABELS
benign_s11["label"] = 0
malignant_s11["label"] = 1

# ✅ MERGE DATASETS ON FREQUENCY
benign = pd.merge(benign_s11, benign_phase, on='Freq [GHz]')
malignant = pd.merge(malignant_s11, malignant_phase, on='Freq [GHz]')

# ✅ CONCATENATE BOTH DATASETS
data = pd.concat([benign, malignant], ignore_index=True)

print(data)

# ✅ DEFINE FEATURES AND TARGET
# We'll use both S11 and Phase as features
X = data.iloc[:, 1:-1]  # Drop frequency column and label
y = data["label"]

# print(data.head())
# print("*************************")

# print(X.head())
# print("*************************")

# print(y.head())
# print("*************************")

# ✅ SCALE THE FEATURES
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ TRAIN SVM MODEL
print("Training SVM Model...")
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_y_pred = svm_model.predict(X_test)

# ✅ TRAIN RANDOM FOREST MODEL
print("Training Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)

# ✅ EVALUATE SVM MODEL
print("SVM Model Performance:")
print("Accuracy:", accuracy_score(y_test, svm_y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_y_pred))
print("Classification Report:\n", classification_report(y_test, svm_y_pred))

# ✅ EVALUATE RANDOM FOREST MODEL
print("\nRandom Forest Model Performance:")
print("Accuracy:", accuracy_score(y_test, rf_y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_y_pred))
print("Classification Report:\n", classification_report(y_test, rf_y_pred))

# ✅ Plot the Confusion Matrix for SVM
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
svm_conf_matrix = confusion_matrix(y_test, svm_y_pred)
sns.heatmap(svm_conf_matrix, annot=True, fmt='g', cmap='Purples')
plt.title('Confusion Matrix - SVM Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(ticks=[0.5, 1.5], labels=['Benign', 'Malignant'])
plt.yticks(ticks=[0.5, 1.5], labels=['Benign', 'Malignant'])

# ✅ Plot the Confusion Matrix for Random Forest
plt.subplot(1, 2, 2)
rf_conf_matrix = confusion_matrix(y_test, rf_y_pred)
sns.heatmap(rf_conf_matrix, annot=True, fmt='g', cmap='Greens')
plt.title('Confusion Matrix - Random Forest Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(ticks=[0.5, 1.5], labels=['Benign', 'Malignant'])
plt.yticks(ticks=[0.5, 1.5], labels=['Benign', 'Malignant'])

# ✅ Display the plots
plt.tight_layout()
plt.show()

os.makedirs("new", exist_ok=True)

# Save the full model object using pickle
with open("new/malignant_v_benign_rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("\n✅ Model saved as: new/malignant_v_benign_rf_model.pkl")
