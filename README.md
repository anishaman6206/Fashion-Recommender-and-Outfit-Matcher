# Fashion Product Recommendation System 🛍️  

This project demonstrates a **content-based fashion product recommender system** with advanced capabilities, including outfit compatibility evaluation, image-based recommendations, and product filtering. The application integrates cutting-edge machine learning techniques to provide an intuitive and fast recommendation experience.  

## Features 🌟  

1. **Product Search and Filtering**  
   - Utilizes **TF-IDF** to preprocess and classify over **21,000 apparel items** across **57 unique categories**.  
   - Includes filters for categories, price range, and product attributes for refined searches.  

2. **Outfit Compatibility Recommender**  
   - Integrates the **Google Gemini API** to analyze clothing attributes and recommend complementary outfit items.  

3. **Image-Based Recommendation**  
   - Fine-tunes the **ResNet50** model to extract image embeddings for precise similarity-based recommendations.  
   - **FAISS ANN** (Approximate Nearest Neighbors) integration reduces recommendation latency to **0.013 seconds**, achieving a **16.6x speed improvement** over traditional cosine similarity.  

## Results 📊  

- **Classification Accuracy**: Fine-tuning ResNet50 resulted in a **30% improvement** (from **43.8% to 73.0%**) in classification accuracy on test data.  
- **Latency Optimization**: FAISS-powered recommendations achieved real-time performance, making the system highly scalable.  

## Demo 🌐  

The application is deployed on **Streamlit** and offers:  

1. **Product Search**: Search by keywords or filter by category, price, and other attributes.  
2. **Outfit Combination Recommender**: Suggests complementary items for a given clothing piece.  
3. **Image-Based Recommendation**: Upload an image to find visually similar apparel items.  

## How to Use 🚀  

### Prerequisites  

- Install Python 3.8 or above  
- Clone this repository and navigate to the project folder  
- Install dependencies using:  
  ```bash  
  pip install -r requirements.txt  
  ```  

### Run the App  

1. Launch the Streamlit app:  
   ```bash  
   streamlit run app.py  
   ```  
2. Upload an image or search for products to explore the system's features.  

### Sample Data  

The project uses a **filtered dataset of 21,000+ apparel items**. Ensure the dataset and required pre-trained model files (`image_embeddings1.npy` and `final_articleType_model1.h5`) are present in the project directory.  

## Repository Structure 📂  

- **`app.py`**: Main Python script integrating all features and the Streamlit app.  
- **`data/`**: Dataset folder (21k+ apparel items).  
- **`models/`**: Pre-trained model files and embeddings.  
- **`requirements.txt`**: List of Python dependencies.  

## Technical Details 🔍  

- **Content-Based Filtering**: TF-IDF vectorization for text-based product search.  
- **ResNet50**: Fine-tuned for image classification and feature extraction.  
- **FAISS ANN**: Reduces latency for large-scale image similarity search.  
- **Google Gemini API**: Analyzes attributes to recommend complementary outfits.  
