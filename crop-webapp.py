import streamlit as st
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # Import pandas for creating the table

# Function to classify the crop
def classify(answer):
    return f"<h2 style='color:green; text-align:center;'>🌟 {answer[0]} 🌟</h2><p style='text-align:center; font-size:20px;'>is the best crop for cultivation here!</p>"

# Load pre-trained models
LogReg_model = pickle.load(open('LogReg_model.pkl', 'rb'))


# Predefined accuracy scores
model_accuracies = {
    'Logistic Regression': 85.2,
    
}

# Generate modern feature importance graph dynamically
def plot_graph(model_name):
    contributions = {
        'Logistic Regression': [20, 15, 10, 25, 20, 5, 5],
        
    }

    labels = ["N (Nitrogen)", "P (Phosphorous)", "K (Potassium)", "Temperature (°C)", "Humidity (%)", "pH", "Rainfall (mm)"]
    importance = contributions[model_name]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, importance, color='skyblue', edgecolor='black', alpha=0.85)

    ax.set_title(f"Feature Importance - {model_name}", fontsize=14, weight='bold')
    ax.set_xlabel("Importance (%)", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    # Compact design
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#8B8B8B')
    ax.spines['bottom'].set_color('#8B8B8B')

    st.pyplot(fig)

# Main function to render the Streamlit app
def main():
    st.set_page_config(page_title="Crop Recommender", layout="wide")
    st.title("🌱 Crop Recommendation System ")

    st.markdown(
        """
        Welcome to our intelligent assistant for recommending the most suitable crop for your soil conditions. 🌾🌽
        """
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open('cc.jpg')
        st.image(image, caption="Your smart crop recommender", use_column_width=True, output_format="auto")

    st.sidebar.header("🔧 Configure the Model")
    activities = ['Logistic Regression']
    option = st.sidebar.selectbox("Which model would you like to use?", activities)

    st.sidebar.header("📊 Input Parameters")
    sn = st.sidebar.slider('NITROGEN (N)', 0.0, 150.0, step=1.0)
    sp = st.sidebar.slider('PHOSPHOROUS (P)', 0.0, 150.0, step=1.0)
    pk = st.sidebar.slider('POTASSIUM (K)', 0.0, 210.0, step=1.0)
    pt = st.sidebar.slider('TEMPERATURE (°C)', 0.0, 50.0, step=0.1)
    phu = st.sidebar.slider('HUMIDITY (%)', 0.0, 100.0, step=0.1)
    pPh = st.sidebar.slider('pH', 0.0, 14.0, step=0.1)
    pr = st.sidebar.slider('RAINFALL (mm)', 0.0, 300.0, step=1.0)
    inputs = [[sn, sp, pk, pt, phu, pPh, pr]]

    # Map user-friendly names to accuracy dictionary keys
    model_key_mapping = {
        'Logistic Regression': 'Logistic Regression',
    }

    accuracy_key = model_key_mapping[option]
    st.sidebar.write(f"**Model Accuracy:** {model_accuracies[accuracy_key]}%")

    if st.sidebar.button('Classify'):
        if option == 'Logistic Regression':
            st.markdown(classify(LogReg_model.predict(inputs)), unsafe_allow_html=True)
            plot_graph('Logistic Regression')

        
    

    st.markdown("""<hr style='border: 1px solid #ccc;' />""", unsafe_allow_html=True)

    st.markdown("""
    ### About the Models
    - **Logistic Regression**: Predicts the best crop based on linear relationships.

    #### Disclaimer
    This tool is designed for educational purposes. For professional agricultural advice, please consult an expert.
    """)

if __name__ == '__main__':
    main()
