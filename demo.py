import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set the title of the app
st.title('Advanced Modeling and Demography App')

# Sidebar for user input
st.sidebar.header('User Input Parameters')

# Function to get user input
def user_input_features(data):
    st.sidebar.subheader('Select Features for Modeling')
    features = st.sidebar.multiselect('Features', data.columns.tolist())
    return features

# Upload CSV data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Display the raw data
    st.subheader('Raw Data')
    st.write(data.head())

    # Get user-selected features
    features = user_input_features(data)

    if features:
        # Prepare the data
        X = data[features].dropna()

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Ask for number of clusters
        st.sidebar.subheader('Model Parameters')
        n_clusters = st.sidebar.slider('Number of Clusters (k)', 2, 10, 3)

        # Perform K-Means Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_scaled)
        labels = kmeans.labels_

        # Add cluster labels to the data
        data['Cluster'] = labels

        # Display clustered data
        st.subheader('Clustered Data')
        st.write(data.head())

        # Visualize the clusters
        st.subheader('Cluster Visualization')
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette='viridis')
        plt.xlabel(features[0])
        plt.ylabel(features[1])
        plt.title('K-Means Clustering')
        st.pyplot(plt)

    else:
        st.write('Please select at least one feature for modeling.')
else:
    st.write('Awaiting CSV file to be uploaded.')

# Footer
st.markdown("""
*Developed with Streamlit and deployed on Heroku.*
""")
