import pickle
import streamlit as st
import os
from PIL import Image
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import plotly.express as px
model_file_path = r'E:\data\DATA\sentiment_model.pkl'
vectorizer_file_path = r'E:\data\DATA\tfidf_vectorizer.pkl'
# Dictionary mapping product names to image paths
product_images = {
    "OnePlus Nord CE 2 5G (Gray Mirror, 8GB RAM, 128GB Storage)": r'E:\12.jpeg',
    "realme narzo 50A (Oxygen Blue , 4GB RAM + 64 GB Storage)": r'E:\13.jpg',
    "Redmi Note 11 (Space Black, 4GB RAM, 64GB Storage)": r'E:\14.jpg',
    "Redmi 10 Prime (Bifrost Blue 4GB RAM 64GB ROM": r'E:\16.jpg',
    "Redmi 9 Activ (Carbon Black, 4GB RAM, 64GB Storage)": r'E:\15.jpg',
    "Samsung Galaxy M32": r'E:\17.jpg',
    "vivo iQOO Z6 5G (Chromatic Blue, 6GB RAM, 128GB Storage)": r'E:\18.jpg',
    "OPPO A31 (Mystery Black, 6GB RAM, 128GB Storage)": r'E:\19.png',
    # Add more products as needed
}

def load_model():
    with open(model_file_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def fit_and_save_vectorizer(data):
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf_vectorizer.fit_transform(data)

    with open(vectorizer_file_path, 'wb') as vectorizer_file:
        pickle.dump(tfidf_vectorizer, vectorizer_file)

    return tfidf_vectorizer

def load_vectorizer():
    with open(vectorizer_file_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return vectorizer

def calculate_rating(positive_percentage, negative_percentage, rating_scale=5):
    rating = (positive_percentage / 100) * rating_scale
    return rating
def create_overall_rating(product_df):
    predicted_sentiment_col = 'Predicted_Sentiment'

    if predicted_sentiment_col not in product_df:
        # If the column doesn't exist, set a default value (you can change this as needed)
        product_df[predicted_sentiment_col] = 0  # Assuming default sentiment is negative

    num_positive_reviews = (product_df[predicted_sentiment_col] == 1).sum()
    num_negative_reviews = (product_df[predicted_sentiment_col] == 0).sum()
    total_reviews = len(product_df)

    overall_positive_percentage = (num_positive_reviews / total_reviews) * 100
    overall_negative_percentage = (num_negative_reviews / total_reviews) * 100

    overall_rating_out_of_5 = calculate_rating(overall_positive_percentage, overall_negative_percentage)

    return overall_rating_out_of_5

def create_ranking_table(product_df):
    categories = ['camera', 'performance', 'value for money', 'display', 'battery']

    ranking_data = {
        'Category': [],
        'Positive Percentage': [],
        'Negative Percentage': [],
        'Rating': [],
    }

    for category in categories:
        filtered_df = product_df[product_df['Predicted Label'] == category]

        if not filtered_df.empty and 'Predicted_Sentiment' in filtered_df:
            positive_percentage = (filtered_df['Predicted_Sentiment'] == 1).mean() * 100
            negative_percentage = (filtered_df['Predicted_Sentiment'] == 0).mean() * 100
            rating = calculate_rating(positive_percentage, negative_percentage)

            ranking_data['Category'].append(category)
            ranking_data['Positive Percentage'].append(positive_percentage)
            ranking_data['Negative Percentage'].append(negative_percentage)
            ranking_data['Rating'].append(rating)

    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values(by=['Rating'], ascending=False).reset_index(drop=True)

    return ranking_df

def create_category_ranking_table(product_df):
    categories = ['camera', 'performance', 'value for money', 'display', 'battery']

    ranking_data = {
        'Product Name': [],
        'Overall Rating': [],
        'Camera Rating': [],
        'Performance Rating': [],
        'Value for Money Rating': [],
        'Display Rating': [],
        'Battery Rating': [],
    }

    for product_name in product_df['Product Name'].unique():
        product_copy = product_df[product_df['Product Name'] == product_name].copy()

        X_tfidf_product_copy = tfidf_vectorizer.transform(product_copy['Cleaned_Review'])
        product_copy['Predicted_Sentiment'] = sentiment_classifier.predict(X_tfidf_product_copy)

        overall_rating = create_overall_rating(product_copy)

        rating_data = {
            'Product Name': product_name,
            'Overall Rating': overall_rating,
        }

        for category in categories:
            category_df = product_copy[product_copy['Predicted Label'] == category]
            if not category_df.empty and 'Predicted_Sentiment' in category_df:
                positive_percentage = (category_df['Predicted_Sentiment'] == 1).mean() * 100
                negative_percentage = (category_df['Predicted_Sentiment'] == 0).mean() * 100
                rating = calculate_rating(positive_percentage, negative_percentage)
                rating_data[f'{category} Rating'] = rating
            else:
                rating_data[f'{category} Rating'] = None

        ranking_data['Product Name'].append(product_name)
        ranking_data['Overall Rating'].append(overall_rating)
        ranking_data['Camera Rating'].append(rating_data['camera Rating'])
        ranking_data['Performance Rating'].append(rating_data['performance Rating'])
        ranking_data['Value for Money Rating'].append(rating_data['value for money Rating'])
        ranking_data['Display Rating'].append(rating_data['display Rating'])
        ranking_data['Battery Rating'].append(rating_data['battery Rating'])

    ranking_df = pd.DataFrame(ranking_data)
    ranking_df = ranking_df.sort_values(by=['Overall Rating'], ascending=False).reset_index(drop=True)

    # Add 'Rank' column based on 'Overall Rating'
    ranking_df['Rank'] = ranking_df.index + 1

    return ranking_df


def create_ring_progress_bar(percentage, positive_color, negative_color, is_negative=False):
    color = negative_color if is_negative else positive_color

    return f"""
        <div style="width: 150px; height: 150px; border-radius: 50%; background-color: #ddd; display: flex; align-items: center; justify-content: center; position: relative;">
            <div style="width: 130px; height: 130px; border-radius: 50%; background-color: {color}; display: flex; align-items: center; justify-content: center; position: absolute; overflow: hidden;">
                <div style="position: absolute; font-size: 18px; color: #fff; z-index: 1; transform: translate(-50%, -50%);">{percentage:.2f}%</div>
                <div style="position: absolute; border-radius: 50%; clip: rect(0, 65px, 130px, 0); width: 130px; height: 130px; background-color: #fff; transform: rotate({percentage * 3.6}deg);"></div>
                <div style="position: absolute; border-radius: 50%; clip: rect(0, 65px, 130px, 0); width: 130px; height: 130px; background-color: {color};"></div>
            </div>
        </div>
    """

def create_star_progress_bar(rating):
    num_full_stars = int(rating)
    remainder = rating - num_full_stars

    full_stars = '⭐️' * num_full_stars
    half_star = '⭐️' if remainder >= 0.5 else ''
    empty_stars = '☆' * (5 - num_full_stars - (1 if remainder >= 0.5 else 0))

    return f"""
        <div style="display: flex; align-items: center; font-size: 24px;">
            {full_stars}{half_star}{empty_stars}
            <span style="margin-left: 8px;">{rating:.2f}</span>
        </div>
    """

# Set custom CSS styles to make the page wider and add a background with an emoji
st.markdown(
    """
    <style>
    .reportview-container {
        max-width: 1900px;
        padding-top: 3rem;
        padding-right: 3rem;
        padding-left: 3rem;
        padding-bottom: 3rem;
        background-color: #f2f2f2; /* Light gray background color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load your image
image = Image.open('E:/Sentiment-analysis-HUB-Final.png')

st.image(image,caption=None, use_column_width=None)

st.title('Sentiment Analysis and Rating Prediction App')

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if 'model_fitted' not in st.session_state:
    st.session_state.model_fitted = False

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully.")

    if not st.session_state.model_fitted:
        tfidf_vectorizer = fit_and_save_vectorizer(df['Cleaned_Review'])
        X_tfidf = tfidf_vectorizer.transform(df['Cleaned_Review'])

        sentiment_classifier = load_model()
        sentiment_classifier.fit(X_tfidf, df['Sentiment'])

        st.session_state.model_fitted = True
        st.session_state.tfidf_vectorizer = tfidf_vectorizer
        st.session_state.sentiment_classifier = sentiment_classifier

    else:
        tfidf_vectorizer = st.session_state.tfidf_vectorizer
        sentiment_classifier = st.session_state.sentiment_classifier

    product_column_name = 'Product Name'

    selected_product = st.selectbox('Select a Product', df[product_column_name].unique())
    # Filter the DataFrame for the selected product
    product_df = df[df[product_column_name] == selected_product]
# Display product image based on the selected product
    if selected_product in product_images:
        product_image_path = product_images[selected_product]
        product_image = Image.open(product_image_path)
        st.image(product_image, caption=f"{selected_product}", use_column_width=None)

    # Calculate overall percentages based on the reviews for the selected product
    X_tfidf_product = tfidf_vectorizer.transform(product_df['Cleaned_Review'])
    product_df['Predicted_Sentiment'] = sentiment_classifier.predict(X_tfidf_product)

    num_positive_reviews_product = (product_df['Predicted_Sentiment'] == 1).sum()
    num_negative_reviews_product = (product_df['Predicted_Sentiment'] == 0).sum()
    total_reviews_product = len(product_df)

    overall_positive_percentage_product = (num_positive_reviews_product / total_reviews_product) * 100
    overall_negative_percentage_product = (num_negative_reviews_product / total_reviews_product) * 100
    st.write(f'Overall Positive for {selected_product}: {overall_positive_percentage_product:.2f}%')
    st.markdown(create_ring_progress_bar(overall_positive_percentage_product, '#2ecc71', '#3498db'), unsafe_allow_html=True)

    st.write(f'Overall Negative for {selected_product}: {overall_negative_percentage_product:.2f}%')
    st.markdown(create_ring_progress_bar(overall_negative_percentage_product, '#e74c3c', '#c0392b', is_negative=True), unsafe_allow_html=True)

    overall_rating_out_of_5_product = calculate_rating(overall_positive_percentage_product, overall_negative_percentage_product)
    st.write(f'Overall Rating out of 5 for {selected_product}:')
    st.markdown(create_star_progress_bar(overall_rating_out_of_5_product), unsafe_allow_html=True)
   
    unique_labels_product = product_df['Predicted Label'].unique()

    # Here, you can use the 'sentiment_option' variable to capture the user's selection for each label
    # You may want to store this information or perform further actions based on the user's choices

    # Display radio buttons for sentiment for each specific label
    specific_labels = ['camera', 'performance', 'value for money', 'display', 'battery']
    selected_specific_label = st.sidebar.radio("Select a Specific Label", specific_labels)

    specific_label_df = product_df[product_df['Predicted Label'] == selected_specific_label]

    X_tfidf_specific_label = tfidf_vectorizer.transform(specific_label_df['Cleaned_Review'])
    specific_label_df['Predicted_Sentiment'] = sentiment_classifier.predict(X_tfidf_specific_label)

    specific_label_positive_percentage = (specific_label_df['Predicted_Sentiment'] == 1).mean() * 100
    specific_label_negative_percentage = (specific_label_df['Predicted_Sentiment'] == 0).mean() * 100

    st.write(f'Positive for {selected_specific_label} in {selected_product}: {specific_label_positive_percentage:.2f}%')
    st.markdown(create_ring_progress_bar(specific_label_positive_percentage, '#2ecc71', '#3498db'), unsafe_allow_html=True)

    st.write(f'Negative for {selected_specific_label} in {selected_product}: {specific_label_negative_percentage:.2f}%')
    st.markdown(create_ring_progress_bar(specific_label_negative_percentage, '#e74c3c', '#c0392b', is_negative=True), unsafe_allow_html=True)

    specific_label_rating = calculate_rating(specific_label_positive_percentage, specific_label_negative_percentage)
    st.write(f'Rating out of 5 for {selected_specific_label} in {selected_product}:')
    st.markdown(create_star_progress_bar(specific_label_rating), unsafe_allow_html=True)
    # Create and display the ranking table for the selected product
    st.title(f"Ranking for {selected_product}")
    ranking_table_product = create_ranking_table(product_df)
    st.table(ranking_table_product)

    st.title('Mobile Phone Rankings Based on Category Ratings')
    ranking_table = create_category_ranking_table(df)
    st.table(ranking_table)

    # ...
else:
    st.info("Please upload a CSV file to begin sentiment analysis and rating prediction.")
