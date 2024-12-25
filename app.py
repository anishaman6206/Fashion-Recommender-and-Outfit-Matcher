import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
import tensorflow as tf
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import gdown
#import faiss

# Page Configuration
st.set_page_config(page_title='Fashion Product Recommender', page_icon="👗", layout="wide")

# Load Data and Prepare Global Variables
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('fashion_data-_1_.csv')
        df1 = df[df['masterCategory']=='Apparel']
        
        # Convert the missing IDs to a set of integers
        missing_ids = {'12347', '39403', '39401', '39410', '39425'}
        missing_ids = set(map(int, missing_ids))
        
        # Remove rows containing the missing IDs
        df_filtered = df1[~df1['id'].isin(missing_ids)].reset_index(drop=True)
        
        # Prepare filename
        df_filtered['filename'] = df_filtered['filename'].astype(str)
        df_filtered['filename'] = df_filtered['filename'].apply(lambda x: os.path.join("filtered_images/", x))+".jpg"
        
        
        return df_filtered
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Initialize AI and ML Models
@st.cache_resource
def load_models():
    try:
        # Configure Gemini AI
        genai.configure(api_key="AIzaSyCF5NyCk8LvDATLUtEsTmcS_NgHBi4Az3Q")
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")
      
      
        
        # Load precomputed embeddings
        #embeddings = np.load("image_embeddings1.npy")
        image_paths = np.load("image_paths1.npy")
        # Use FAISS for approximate nearest neighbors
        #embedding_dim = embeddings.shape[1]
        #index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index
        #index.add(embeddings)
        
        return {
            'gemini': gemini_model,
            #'embeddings': embeddings,
            'image_paths': image_paths,
            #'faiss_index': index
        }
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None



# Function to download files from Google Drive
def download_file(file_id, output_file, description):
    if not os.path.exists(output_file):
        try:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            gdown.download(download_url, output_file, quiet=False)
            st.success(f"Downloaded {description} successfully!")
        except Exception as e:
            st.error(f"Failed to download {description}: {e}")
            raise e

# Function to download the model file
def download_model():
    model_file = "final_articleType_model1.h5"
    model_file_id = "1z0IMZDLRNzZ-2sIRfcMewB4WNy2Vvp8z"
    download_file(model_file_id, model_file, "model")

# Function to download the embeddings file
def download_embeddings():
    embeddings_file = "image_embeddings1.npy"
    embeddings_file_id = "1YG20Q3jwn7L5MepReGK-GyQKHdoI8wEE"  # Replace with the file ID of your embeddings file
    download_file(embeddings_file_id, embeddings_file, "image embeddings")
            
            

# Recommendation Functions from First Code
def get_recommendations(user_input, data, vectorizer, tfidf_matrix):
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    indices = similarities.argsort()[-50:][::-1]
    recommended_items = data.iloc[indices]
    return recommended_items

# Image Recommendation Functions from Second Code
def preprocess_image(image, target_size=(180, 180)):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)



def recommend_similar_images(input_embedding, embeddings, image_paths, top_n=16):
    scores = cosine_similarity([input_embedding], embeddings)[0]
    recommended_indices = np.argsort(scores)[::-1][:top_n]
    recommendations = [(image_paths[i], scores[i]) for i in recommended_indices]
    return recommendations

# Combination Recommendation Functions from Third Code
def get_combination_feedback(user_selected_product_ids, data, model):
    selected_images = [Image.open(data[data['id'] == pid]['filename'].iloc[0]) for pid in user_selected_product_ids]
    
    contents = ["Here are images of the selected products:"]
    for i, img in enumerate(selected_images):
        contents.extend([f"Product {i+1}:", img])
    
    question = """
Based on their:

* **Style:** (e.g., casual, formal, sporty, bohemian)
* **Color:** (e.g., color palette, complementary, contrasting)
* **Material:** (e.g., fabrics, textures, compatibility)
* **Overall Appearance:** (e.g., visual weight, silhouette, design elements)

Can these products be combined effectively to create a visually appealing and cohesive outfit? 

**Note:** Combinations of items within the same primary category (e.g., two shirts, two pairs of pants) are generally not considered effective. Combinations of items from different categories (e.g., shirt and pants, shirt and jacket, skirt and top) are typically more successful.

Return a JSON object with:

* **decision:** "yes" or "no"
* **reason:** A concise explanation supporting the decision. 
"""
    contents.append(question)
    
    responses = model.generate_content(contents, stream=True)
    all_responses = [response.text for response in responses]
    
    try:
        concatenated_response = ''.join(all_responses)
        cleaned_response = concatenated_response.replace('```json', '').replace('```', '').strip()
        response_json = json.loads(cleaned_response)
        
        decision = response_json.get("decision", "Unknown")
        reason = response_json.get("reason", "No reason provided")
        return decision, reason
    except Exception as e:
        st.error(f"Error parsing response: {e}")
        return "Unknown", "Response parsing error"
    
def load_image_from_dataset(product_id, dataframe):
    # Function to load an image from the dataset based on product ID
    file_path = dataframe[dataframe['id'] == product_id]['filename'].iloc[0]
    try:
        return Image.open(file_path)
    except IOError:
        print(f"Cannot load image for product ID {product_id}")
        return None    
    
def recommend_complementary_products(user_selected_product_ids, decision, reason, data, model):
    if decision == 'yes':
        selected_images = [load_image_from_dataset(pid, data) for pid in user_selected_product_ids]

        # Extract color, usage, and gender info from the AI feedback
        ai_feedback = reason.lower()

        # Extract color and usage from AI feedback
        color_keywords = ['blue', 'white', 'black', 'red']
        extracted_color = next((color for color in color_keywords if color in ai_feedback), None)

        usage_keywords = ['casual', 'formal', 'party']
        extracted_usage = next((usage for usage in usage_keywords if usage in ai_feedback), None)

        # Extract gender from selected product details
        selected_details = data[data['id'].isin(user_selected_product_ids)].iloc[0]
        selected_gender = selected_details['gender'].lower()

        # Filter the dataset using the AI-extracted color, usage, and gender
        if extracted_color and extracted_usage:
            similar_products = data[(data['baseColour'].str.lower() == extracted_color) &
                                    (data['usage'].str.lower() == extracted_usage) &
                                    (data['gender'].str.lower() == selected_gender) &
                                    (data['subCategory'].str.lower() == selected_details['subCategory'].lower())]
        else:
            # Fallback to selected product details if AI didn't provide specific colors or usage
            selected_color = selected_details['baseColour']
            selected_usage = selected_details['usage']
            selected_subcategory = selected_details['subCategory']
            similar_products = data[(data['baseColour'] == selected_color) &
                                    (data['usage'] == selected_usage) &
                                    (data['gender'] == selected_gender) &
                                    (data['subCategory'] == selected_subcategory)]

        return similar_products[['id','productDisplayName', 'baseColour', 'usage', 'subCategory', 'gender']].set_index('id')
    else:
        return pd.DataFrame()    
    
    

# Main Streamlit App
def main():
    # Load data and models
    data = load_data()
    models = load_models()
    
    if data is None or models is None:
        st.error("Failed to load data or models. Please check your files.")
        return
    
    # Prepare TF-IDF for recommendations
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(data['combined_features'])
    
    # Sidebar Navigation
    app_mode = st.sidebar.selectbox("Choose App Mode", 
        [
            "Product Search & Filter", 
            "Outfit Combination Recommender",
            "Image-Based Recommendation"
            
        ]
    )
    
    # Product Search & Filter Mode
    if app_mode == "Product Search & Filter":
        st.title("Product Search & Filter")
        
        # Sidebar Filters
        st.sidebar.header("Filters")
        gender_filter = st.sidebar.multiselect("Gender", data['gender'].unique(), default=data['gender'].unique())
        price_range = st.sidebar.slider("Price Range", int(data['Price'].min()), int(data['Price'].max()), (int(data['Price'].min()), int(data['Price'].max())))
        size_filter = st.sidebar.multiselect("Size", set(size for sizes in data['SizeOption'].dropna() for size in sizes.split(", ")), default=None)
        
        # User Search Input
        user_query = st.text_input("Search for a product by name or description:", placeholder="e.g., 'Turtle Check Men Navy Blue Shirt'")
        
        # Apply Filters
        filtered_data = data[
            (data['gender'].isin(gender_filter)) &
            
            (data['Price'].between(price_range[0], price_range[1])) &
            (data['SizeOption'].apply(lambda x: any(size in str(x) for size in size_filter) if size_filter else True))
        ]
        
        if st.button("Search"):
            if user_query.strip() == "":
                results = filtered_data[:300]
            else:
                recommendations = get_recommendations(user_query, data, vectorizer, tfidf_matrix)
                results = recommendations[
                    (recommendations['gender'].isin(gender_filter)) &
                   
                    (recommendations['Price'].between(price_range[0], price_range[1])) &
                    (recommendations['SizeOption'].apply(lambda x: any(size in str(x) for size in size_filter) if size_filter else True))
                ]
            
            # Display Results
            if results.empty:
                st.write("No products found.")
            else:
                cols = st.columns(2)
                for idx, row in results.iterrows():
                    with cols[idx % 2]:
                        st.write(f"Product ID: {row['id']}")
                        st.image(row['link'], width=400)
                        st.write(f"{row['productDisplayName']}")
                        st.write(f"Price: ₹{row['Price']}")
                        st.write(f"Sizes: {row['SizeOption']}")
                        
                        st.write("---")


    # Outfit Combination Recommender Mode
    elif app_mode == "Outfit Combination Recommender":
        st.title("Outfit Combination Recommender")

        # Instruction for users
        st.markdown("""
                    ### Instructions:
                    - Select **two complementary product IDs** to see how they match.
                    - Few examples of complementary pairs include:
                        - **Shirt and Jeans** 
                        - **Shirt and Jacket** 
                        - **Top and Skirt** 
                    - You can find product IDs by navigating to **Product Search** mode.
                    """)
        
        # Get unique product IDs
        unique_product_ids = sorted(data['id'].unique())
        
        # Multi-select for product IDs
        user_selected_product_ids = st.multiselect(
            "Choose Product IDs", 
            unique_product_ids, 
            default=None,
            max_selections=2
        )
        
        if len(user_selected_product_ids) == 2:
            # Display selected product images
            col1, col2 = st.columns(2)
            
            with col1:
                st.header(f"Product {user_selected_product_ids[0]}")
                img1 = data[data['id'] == user_selected_product_ids[0]]['link'].iloc[0]
                st.image(img1, width=400)
            
            with col2:
                st.header(f"Product {user_selected_product_ids[1]}")
                img2 = data[data['id'] == user_selected_product_ids[1]]['link'].iloc[0]
                st.image(img2, width=400)
            
            # Get combination feedback
            decision, reason = get_combination_feedback(user_selected_product_ids, data, models['gemini'])
            
            st.subheader("AI Stylist's Decision")
            st.write(f"**Decision:** {decision}")
            st.write(f"**Reason:** {reason}")

            # Recommend complementary products if combination is good
            if decision == 'yes':
                st.header("Complementary Product Recommendations")
            
            # Get complementary products
                complementary_products = recommend_complementary_products(
                user_selected_product_ids, 
                decision, 
                reason, 
                data, 
                model= models['gemini']
            )

                if not complementary_products.empty:
                    st.dataframe(complementary_products)
                else:
                    st.write("No complementary products found.")
            else:
                st.warning("The products cannot be combined effectively.")
   
    
    # Image-Based Recommendation Mode
    elif app_mode == "Image-Based Recommendation":
        st.title("Image Classification & Recommendation")
    
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
        # Download necessary files
        if not os.path.exists("final_articleType_model1.h5"):
            with st.spinner("Downloading model..."):
                download_model()
        
        if not os.path.exists("image_embeddings1.npy"):
            with st.spinner("Downloading image embeddings..."):
                download_embeddings()
    
        # Load model after downloading
        try:
            # Load the model
            tf_model = tf.keras.models.load_model('final_articleType_model1.h5')
    
            # Load the embeddings
            embeddings = np.load("image_embeddings1.npy", allow_pickle=True).item()
    
            if uploaded_file is not None:
                # Create a container to center and resize the uploaded image
                col1, col2, col3 = st.columns([1, 3, 1])
    
                with col2:
                    st.image(uploaded_file, caption="Uploaded Image", width=400)
    
                try:
                    image = Image.open(uploaded_file)
                    processed_image = preprocess_image(image)
    
                    # Predict image class
                    predictions = tf_model.predict(processed_image)
                    le = LabelEncoder()
                    le.fit(data['articleType'])
    
                    # Make predictions
                    predicted_class = np.argmax(predictions)
    
                    # Get the corresponding class label
                    predicted_label = le.inverse_transform([predicted_class])[0]
                    st.write(f"**Predicted Class:** {predicted_label}")
    
                    # Extract embeddings for the input image
                    st.write("Extracting features...")
                    feature_extractor = tf.keras.Model(inputs=tf_model.input, outputs=tf_model.layers[-4].output)
                    input_embedding = feature_extractor.predict(processed_image).flatten()
    
                    # Recommend similar images
                    st.write("Fetching recommendations...")
                    recommendations = recommend_similar_images(input_embedding, embeddings['embeddings'], embeddings['image_paths'])
    
                    # Display recommended images in 2 images per row
                    st.write("**Recommended Images:**")
    
                    # Create rows with 2 images each
                    for i in range(0, len(recommendations), 2):
                        cols = st.columns(2)
    
                        # First image in the row
                        with cols[0]:
                            rec_path1, score1 = recommendations[i]
                            normalized_rec_path1 = rec_path1.replace("\\", "/")
                            
                            # Retrieve product details from data
                            link = data[data['filename'] == normalized_rec_path1]['link'].iloc[0]
                            product_id = data[data['filename'] == normalized_rec_path1]['id'].iloc[0]
                            product_name = data[data['filename'] == normalized_rec_path1]['productDisplayName'].iloc[0]
                            product_price = data[data['filename'] == normalized_rec_path1]['Price'].iloc[0]
                            product_sizes = data[data['filename'] == normalized_rec_path1]['SizeOption'].iloc[0]
    
                            st.image(link, caption=f"Similarity: {score1:.2f}", width=400)
                            st.write(f"Product ID: {product_id}")
                            st.write(f"{product_name}")
                            st.write(f"Price: ₹{product_price}")
                            st.write(f"Sizes: {product_sizes}")
                            st.write("---")
    
                        # Second image in the row (if available)
                        if i + 1 < len(recommendations):
                            with cols[1]:
                                rec_path2, score2 = recommendations[i + 1]
                                normalized_rec_path2 = rec_path2.replace("\\", "/")
                                
                                # Retrieve product details from data
                                link = data[data['filename'] == normalized_rec_path2]['link'].iloc[0]
                                product_id = data[data['filename'] == normalized_rec_path2]['id'].iloc[0]
                                product_name = data[data['filename'] == normalized_rec_path2]['productDisplayName'].iloc[0]
                                product_price = data[data['filename'] == normalized_rec_path2]['Price'].iloc[0]
                                product_sizes = data[data['filename'] == normalized_rec_path2]['SizeOption'].iloc[0]
    
                                st.image(link, caption=f"Similarity: {score2:.2f}", width=400)
                                st.write(f"Product ID: {product_id}")
                                st.write(f"{product_name}")
                                st.write(f"Price: ₹{product_price}")
                                st.write(f"Sizes: {product_sizes}")
                                st.write("---")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        except Exception as e:
            st.error(f"Failed to load the model: {e}")
if __name__ == "__main__":
    main()