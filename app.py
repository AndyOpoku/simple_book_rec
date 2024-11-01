import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random

data = pickle.load(open("book_data.pkl", 'rb'))
cosine_sim= pickle.load(open("cosine_sim.pkl", 'rb'))
#movies_list = movies['title'].values

# Note: You'll need to import your existing data and cosine_sim
# Make sure to load these before running the Streamlit app
# data = pd.read_csv('your_data.csv')
# cosine_sim = np.load('cosine_sim_matrix.npy')

def recommend_books(book_title, cosine_sim=cosine_sim):
    # Clean the book title (strip white spaces and convert to lowercase)
    book_title = book_title.strip().lower()
    
    # Get the index of the book that matches the title
    idx = data[data['title'].str.lower() == book_title].index
    
    if len(idx) == 0:
        return f"Book '{book_title}' not found in the dataset."
    idx = idx[0]
    # Get the cosine similarity scores for all books with this book
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the top 10 most similar books (excluding the input book)
    sim_scores = sim_scores[1:11]
    # Get the book indices
    book_indices = [i[0] for i in sim_scores]
    # Return the top 10 recommended books
    return data['title'].iloc[book_indices]

# Styling configurations
st.set_page_config(
    page_title="BookBuddy - Recommendation System", 
    page_icon="📚", 
    layout="wide"
)

def recommend_books(book_title, cosine_sim=cosine_sim):
    # Clean the book title (strip white spaces and convert to lowercase)
    book_title = book_title.strip().lower()
    
    # Get the index of the book that matches the title
    idx = data[data['title'].str.lower() == book_title].index
    
    if len(idx) == 0:
        return f"Book '{book_title}' not found in the dataset."
    idx = idx[0]
    # Get the cosine similarity scores for all books with this book
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the top 10 most similar books (excluding the input book)
    sim_scores = sim_scores[1:11]
    # Get the book indices
    book_indices = [i[0] for i in sim_scores]
    # Return the top 10 recommended books with similarity scores
    recommended_books = data.iloc[book_indices]
    return recommended_books

def get_random_emoji():
    book_emojis = [
        "📖", "🔖", "📚", "📕", "📗", "📘", "📙", "📓", "📒", "📃", 
        "📄", "📜", "📋", "📑", "🗒️"
    ]
    return random.choice(book_emojis)

def main():
    # Custom CSS for styling
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
        color: #2C3E50;
    }
    .recommendation-card {
        background-color: #F0F4F8;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        transition: transform 0.3s ease;
    }
    .recommendation-card:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #2980B9;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title with animated styling
    st.markdown("""
    <h1 style='text-align: center; color: #2C3E50; 
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    animation: fadeIn 2s;'>
    BookBuddy 📚 Recommendation System
    </h1>
    """, unsafe_allow_html=True)

    # Sidebar for additional features
    st.sidebar.title("BookBuddy Explorer")
    
    # Add a cool feature to discover random books
    if st.sidebar.button("🎲 Discover Random Book"):
        try:
            random_book = data.sample(1)['title'].values[0]
            st.sidebar.success(f"Random Book: {random_book}")
            
            # Automatically show recommendations for random book
            recommendations = recommend_books(random_book)
            st.session_state['recommendations'] = recommendations
            st.session_state['original_book'] = random_book
        except Exception as e:
            st.sidebar.error("Couldn't fetch a random book")

    # Book title input
    book_title = st.text_input(
        "Enter a Book Title:", 
        placeholder="e.g., To Kill a Mockingbird",
        help="Type a book title and get personalized recommendations!"
    )

    # Recommendation button with custom styling
    col1, col2, _ = st.columns([2,2,6])
    with col1:
        recommend_btn = st.button("🔍 Get Recommendations")
    with col2:
        clear_btn = st.button("🧹 Clear")

    # Clear recommendations if clear button is pressed
    if clear_btn:
        if 'recommendations' in st.session_state:
            del st.session_state['recommendations']
        if 'original_book' in st.session_state:
            del st.session_state['original_book']
        book_title = ""

    # Process recommendations
    if recommend_btn or 'recommendations' in st.session_state:
        # Use either newly entered book or previously discovered random book
        current_book = book_title or st.session_state.get('original_book', '')
        
        if current_book:
            with st.spinner('Finding similar books...'):
                # Get recommendations
                recommendations = recommend_books(current_book)
            
            # Check if recommendations were found
            if isinstance(recommendations, str):
                st.error(recommendations)
            else:
                # Store recommendations in session state
                st.session_state['recommendations'] = recommendations
                st.session_state['original_book'] = current_book

        # Display recommendations if available
        if 'recommendations' in st.session_state:
            original_book = st.session_state['original_book']
            recommendations = st.session_state['recommendations']
            
            st.subheader(f"📖 Recommendations for '{original_book}'")
            
            # Create a grid of recommendation cards
            for index, row in recommendations.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div class='recommendation-card'>
                        <p class='big-font'>{get_random_emoji()} {row['title']}</p>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# Streamlit run instructions
# 1. Install required libraries: streamlit, pandas, numpy
# 2. Load your data and cosine similarity matrix
# 3. Run with: streamlit run your_app.py