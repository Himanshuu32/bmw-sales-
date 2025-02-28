import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
@st.cache_data
def load_data():
    file_path = "bmw_sales.csv"
    df = pd.read_csv(file_path)
    # Clean data
    for col in ['Total Sales', 'BEV Sales', 'Other Vehicle Sales']:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float).astype('Int64')
    return df

data = load_data()

# Streamlit App Title
st.title("📊 BMW Sales Analysis in the US")
st.write("### BMW Sales Data Preview")
st.dataframe(data)

# Sales Trend Visualization
st.write("## 📈 Sales Trends Over Time")
fig, ax = plt.subplots()
ax.plot(data['Quarter'], data['Total Sales'], marker='o', linestyle='-', label='Total Sales')
ax.plot(data['Quarter'], data['BEV Sales'], marker='s', linestyle='--', label='BEV Sales')
ax.plot(data['Quarter'], data['Other Vehicle Sales'], marker='^', linestyle=':', label='Other Vehicle Sales')
ax.set_xlabel("Quarter")
ax.set_ylabel("Number of Vehicles Sold")
ax.legend()
st.pyplot(fig)

# User Input for Sales Prediction
st.sidebar.header("📊 Predict Future Sales")
quarter_input = st.sidebar.slider("Select Quarter", min_value=1, max_value=12, value=8)
prediction = data['Total Sales'].mean()  # Placeholder for a model
st.sidebar.write(f"**Estimated Sales:** {int(prediction)} vehicles") 
