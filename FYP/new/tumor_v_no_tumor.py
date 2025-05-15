import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim

tumor_data = pd.read_csv("prev/data/s11_14_tumor.csv")
no_tumor_data = pd.read_csv("prev/data/S11_simulated_1.4ghz_withouttumor.csv")

# print(tumor_data.head())
# print(no_tumor_data.head())

'''
Initial Plot:
'''

# plt.plot(no_tumor_data["Freq [GHz]"], no_tumor_data["S11 (dB)"], label="No Tumor - S11 (dB)", alpha=0.7)

# # Plot tumor data
# plt.plot(tumor_data["Freq [GHz]"], tumor_data["S11 (dB)"], label="Tumor - S11 (dB)", alpha=0.7)

# # Labels and legend
# plt.xlabel("Frequency (GHz)")
# plt.ylabel("S11 Parameter")
# plt.title("Comparison of S11 Parameter with and without Tumor")
# plt.legend()
# plt.grid()
# plt.show()

# Add labels
no_tumor_data['Label'] = 0
tumor_data['Label'] = 1

# Combine datasets
data = pd.concat([no_tumor_data, tumor_data], ignore_index=True)
# print(data.head())
# print("\n******************\n")
# print(data.head)
# print("\n******************\n")
# print(data.describe())

'''
Preprocessing
'''

features = data.drop(columns=["Label"])  # Label hatao
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# update df with scaled features,
data_scaled = pd.DataFrame(features_scaled, columns=features.columns)
data_scaled["Label"] = data["Label"]
# print(data_scaled.head)  # Verify

# Gradient for new features
data_scaled["S11_Gradient"] = data_scaled["S11 (dB)"].diff() / data_scaled["Freq [GHz]"].diff()
data_scaled["Phase_Gradient"] = data_scaled["Phase"].diff() / data_scaled["Freq [GHz]"].diff()

# Fill NaN values caused by differencing
data_scaled.fillna(0, inplace=True)

'''
Model Design and Implementation:
'''

# Train - Test Split:

X = data_scaled.drop(columns=["Label"])
y = data_scaled["Label"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

# print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

'''Model : Random Forest'''

# Initialize Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate
print("Random Forest Metrics:")
print(classification_report(y_test, y_pred_rf))

ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Save Model and Scaler with custom names
os.makedirs("new", exist_ok=True)

with open("new/tumor_v_no_tumor_rf_model.pkl", "wb") as model_file:
    pickle.dump(rf_model, model_file)

with open("new/tumor_v_no_tumor_rf_scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("\n✅ Model and scaler saved as:")
print("  - new/tumor_v_no_tumor_rf_model.pkl")
print("  - new/tumor_v_no_tumor_rf_scaler.pkl")

''' Model : Neural Network '''

# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
#         super(NeuralNet, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.fc3 = nn.Linear(hidden_size2, output_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x
    
# input_size = X_train.shape[1]
# hidden_size1 = 128
# hidden_size2 = 64
# output_size = 1

# model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size)

# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# train_losses, val_losses = [], []
# train_accuracies, val_accuracies = [], []

# num_epochs = 50

# for epoch in range(num_epochs):
#     model.train()
    
#     optimizer.zero_grad()
#     outputs_train = model(torch.tensor(X_train.values).float())
#     loss_train = criterion(outputs_train.squeeze(), torch.tensor(y_train.values).float())
#     loss_train.backward()
#     optimizer.step()

#     predicted_train = (outputs_train.squeeze() > 0.5).float()
#     train_accuracy = (predicted_train == torch.tensor(y_train.values).float()).float().mean().item()

#     model.eval()
#     with torch.no_grad():
#         outputs_val = model(torch.tensor(X_val.values).float())
#         loss_val = criterion(outputs_val.squeeze(), torch.tensor(y_val.values).float())

        
#         predicted_val = (outputs_val.squeeze() > 0.5).float()
#         val_accuracy = (predicted_val == torch.tensor(y_val.values).float()).float().mean().item()

    
#     train_losses.append(loss_train.item())
#     val_losses.append(loss_val.item())
#     train_accuracies.append(train_accuracy)
#     val_accuracies.append(val_accuracy)

#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss_train.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {loss_val.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

# # Ensure the directory exists
# os.makedirs("new", exist_ok=True)

# # Save the full model object using pickle
# with open("new/tumor_v_no_tumor_nn_model.pkl", "wb") as f:
#     pickle.dump(model, f)

# print("\n✅ Neural network model saved as: new/tumor_v_no_tumor_nn_model.pkl")
