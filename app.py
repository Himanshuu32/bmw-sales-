try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
except ImportError as e:
    st.error(f"Error importing libraries: {e}. Please check your `requirements.txt` file.")
    st.stop()

# Load the dataset
@st.cache_data
def load_data():
    file_path = "bmw_sales.csv"
    df = pd.read_csv(file_path)
    # Clean data
    for col in ['Total Sales', 'BEV Sales', 'Other Vehicle Sales']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float).astype('Int64')
    return df

data = load_data()

# Streamlit App Title
st.title("📊 BMW Sales Analysis and Prediction in the US")
st.write("### BMW Sales Data Preview")
st.dataframe(data)

# Sales Trend Visualization
st.write("## 📈 Sales Trends Over Time")
if 'Year' in data.columns and 'Total Sales' in data.columns:
    fig, ax = plt.subplots()
    ax.plot(data['Year'], data['Total Sales'], marker='o', linestyle='-', label='Total Sales')
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Vehicles Sold")
    ax.legend()
    st.pyplot(fig)
else:
    st.error("Required columns ('Year' and 'Total Sales') not found in the dataset.")

# Sales Prediction
st.write("## 🔮 Predict Future Sales")

# Prepare data for prediction
if 'Year' in data.columns and 'Total Sales' in data.columns:
    X = data[['Year']]  # Feature: Year
    y = data['Total Sales']  # Target: Total Sales

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # User input for prediction
    st.sidebar.header("Predict Future Sales")
    future_year = st.sidebar.number_input("Enter the year to predict sales:", min_value=2023, max_value=2050, value=2025)

    # Predict sales for the input year
    predicted_sales = model.predict([[future_year]])

    # Display prediction
    st.sidebar.write(f"**Predicted Sales for {future_year}:** {int(predicted_sales[0]):,} vehicles")

    # Plot the regression line
    st.write("### Linear Regression Model")
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Historical Sales')
    ax.plot(X, model.predict(X), color='red', label='Regression Line')
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Sales")
    ax.legend()
    st.pyplot(fig)
else:
    st.error("Required columns ('Year' and 'Total Sales') not found in the dataset.")

st.write("🚀 Future improvements: Use more advanced models for better predictions!")