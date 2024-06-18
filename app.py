import streamlit as st
import pandas as pd
import pickle
from preprocess import preprocess_url
import datetime 
import sqlite3
#from log import add_data_to_sqlite 
#from log import display_all_data
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model = pickle.load(open('model.pkl', 'rb'))


def main():
    st.title("Malicious URL Detection :shield: :desktop_computer:")
    st.write("Enter a URL to predict if it's malicious or not:")
    url = st.text_input("URL:")
    
    if st.button("Predict"):
        if url:
            # Preprocess the URL
            url = preprocess_url(url)
            # Make predictions
            prediction = model.predict([url])  # Make sure to pass the url as a list or the required format for the model
            if prediction == 1:
                st.error("This URL is malicious!")
            else:
                st.success("This URL is safe.")
                st.button("Know More!")
        else:
            st.warning("Please enter a URL.")
    
    if st.button("Analyze Dataset"):
        file_path = "cleaned_malicious_url_dataset.csv"
        
        df=data.head(1000)
        # Add the functionality for "Analyze Dataset" button
        st.write("Dataset analysis functionality is not implemented yet.")
         # Visualization options
        # Distribution of the target variable
        plt.figure(figsize=(10, 6))
        sns.countplot(data['label'])
        plt.title('Distribution of Target Variable')
        plt.show()

        categorical_features = ['is_ip', 'contains_exe', 'ftp_used', 'js_used', 'css_used', 'is_domain_random']

    # Visualize categorical features
        for feature in categorical_features:
            plt.figure(figsize=(10, 6))
            sns.countplot(df[feature])
            plt.title(f'Distribution of {feature}')
            plt.show()
        numerical_features = ['url_length', 'digit_alphabet_ratio', 'specialchar_alphabet_ratio', 'uppercase_lowercase_ratio', 
                      'domain_url_ratio', 'numeric_char_count', 'sensitive_word_count', 'entropy']

    # Histograms
        for feature in numerical_features:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[feature], bins=30, kde=True)
            plt.title(f'Histogram of {feature}')
            plt.show()

    # Box plots
        for feature in numerical_features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=df[feature])
            plt.title(f'Boxplot of {feature}')
            plt.show()

        # Correlation matrix
        plt.figure(figsize=(12, 8))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()

            # Correlation matrix
        plt.figure(figsize=(12, 8))
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.show()
        # Pairplot
        sns.pairplot(df, hue='label')
        plt.show()

        for feature in numerical_features:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='label', y=feature, data=df)
            plt.title(f'{feature} by Label')
            plt.show()


if __name__ == "__main__":
    main()
