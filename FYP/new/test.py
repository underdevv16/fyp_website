import streamlit as st
import numpy as np
import pandas as pd
import time
import torch
import pickle
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

# Load the CSV files
malignant_benign_df = pd.read_excel("FYP/new/malignant_benign.xlsx")
tumor_no_tumor_df = pd.read_excel("FYP/new/tumor_no_tumor.xlsx")


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Load models and scaler
# with open('new\\models\\malignant_v_benign_rf_model.pkl', 'rb') as model_file:
#     rf_malignant_benign_model = pickle.load(model_file)

# with open('new\\tumor_v_no_tumor_rf_model.pkl', 'rb') as model_file:
#     rf_tumor_no_tumor_model = pickle.load(model_file)

# with open('new\\tumor_v_no_tumor_rf_scaler.pkl', 'rb') as scaler_file:
#     tumor_no_tumor_scaler = pickle.load(scaler_file)

# # Load the neural network model, ensuring the class definition is present
# with open('new\\models\\tumor_v_no_tumor_nn_model.pkl', 'rb') as model_file:
#     nn_tumor_no_tumor_model = pickle.load(model_file)

# Define base path
base_path = os.path.join("FYP", "new")

# Load models and scaler
with open(os.path.join(base_path, "models", "malignant_v_benign_rf_model.pkl"), "rb") as model_file:
    rf_malignant_benign_model = pickle.load(model_file)

with open(os.path.join(base_path, "tumor_v_no_tumor_rf_model.pkl"), "rb") as model_file:
    rf_tumor_no_tumor_model = pickle.load(model_file)

with open(os.path.join(base_path, "tumor_v_no_tumor_rf_scaler.pkl"), "rb") as scaler_file:
    tumor_no_tumor_scaler = pickle.load(scaler_file)

# Load the neural network model, ensuring the class definition is present
with open(os.path.join(base_path, "models", "tumor_v_no_tumor_nn_model.pkl"), "rb") as model_file:
    nn_tumor_no_tumor_model = pickle.load(model_file)

# Set up default state
if 'prediction_type' not in st.session_state:
    st.session_state.prediction_type = None

# Default slider values
default_slider_values = {
    's11': -9.0,
    'phase': 0.0,
    'frequency': 10.0
}

def reset_slider_values():
    for key, value in default_slider_values.items():
        st.session_state[key] = value

st.markdown("<h1 style='text-align: center;'> 🧠 Tumor Detection & Type Prediction WebApp </h1>", unsafe_allow_html=True)

st.write(" ")


# Greet the user
st.write("Welcome to the Tumor Detection and Type Prediction App!")
st.write("Please select the parameters below to make a prediction.")


# Prediction type selection
st.write("### Choose Prediction Type:")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔬 Malignant vs Benign"):
        st.session_state.prediction_type = 'Malignant vs Benign'
        reset_slider_values()

with col2:
    if st.button("🧠 Tumor vs No Tumor"):
        st.session_state.prediction_type = 'Tumor vs No Tumor'
        reset_slider_values()

prediction_type = st.session_state.prediction_type

# Show sliders and prediction logic
if prediction_type:

    if prediction_type == 'Malignant vs Benign':
        s11 = st.slider("S11 (in dB)", -20.5, 1.5, step=0.001, value= -9.5, key='s11')
        phase = st.slider("Phase (in radians)", -1.40, 1.50, step=0.01, value= 0.05, key='phase')
        frequency = st.slider("Frequency (in GHz)", 0.0, 25.0, step=0.1, value= 12.5, key='frequency')

    else:
        s11 = st.slider("S11 (in dB)", -18.00000, 1.00000, step=0.00001, value= -8.50000, key='s11')
        phase = st.slider("Phase (in radians)", -3.5000, 3.5000, step=0.0001, value= 0.0000, key='phase')
        frequency = st.slider("Frequency (in GHz)", 0.00, 20.00, step=0.01, value= 10.00, key='frequency')


    if st.button("🔍 Predict"):
        with st.spinner("Thinking..."):
            time.sleep(1.5)  # simulate loading

            if prediction_type == 'Malignant vs Benign':
                # Check exact match in CSV
                match = malignant_benign_df[
                    (malignant_benign_df['S11(dB)'] == s11) &
                    (malignant_benign_df['Phase(rad)'] == phase)
                ]
                if not match.empty:
                    result = "Benign" if match.iloc[0]['Label'] == 0 else "Malignant"
                    st.success(f"✅ Exact Match Found: {result}")
                else:
                    # Use model
                    input_data = np.array([[s11, phase]])
                    pred = rf_malignant_benign_model.predict(input_data)
                    result = "Benign" if pred == 0 else "Malignant"
                    st.info(f"🤖 Predicted: {result}")

            elif prediction_type == 'Tumor vs No Tumor':
                # Check exact match in CSV
                match = tumor_no_tumor_df[
                    (tumor_no_tumor_df['S11 (dB)'] == s11) &
                    (tumor_no_tumor_df['Phase (rad)'] == phase) &
                    (tumor_no_tumor_df['Freq [GHz]'] == frequency)
                ]
                if not match.empty:
                    result = "No Tumor" if match.iloc[0]['Label'] == 0 else "Tumor"
                    st.success(f"✅ Exact Match Found: {result}")
                else:
                    s11_squared = s11 ** 2
                    phase_freq = phase * frequency
                    to_scale = np.array([[s11, phase, frequency]])
                    scaled_input = tumor_no_tumor_scaler.transform(to_scale)
                    input_data = np.hstack([scaled_input, [[s11_squared, phase_freq]]])

                    # RF prediction
                    rf_pred = rf_tumor_no_tumor_model.predict(input_data)
                    rf_result = "No Tumor" if rf_pred == 0 else "Tumor"

                    # NN prediction
                    nn_pred = nn_tumor_no_tumor_model(torch.tensor(input_data).float()).detach().numpy()
                    nn_result = "No Tumor" if nn_pred <= 0.5 else "Tumor"

                    st.info(f"🤖 RF Model: {nn_result}")
                    st.info(f"🧠 NN Model: {nn_result}")
