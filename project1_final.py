import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import time

# Load the CSV file into a DataFrame
df = pd.read_csv("hindi_songs.csv")

# Rename columns for easier access
df.rename(columns={
    'Track Name': 'track_name',
    'Artist Name': 'artist_name',
    'Track URI': 'spotify_link',
    'Album': 'album',
    'Duration (ms)': 'duration'  # Ensure the 'Duration' column exists in your CSV
}, inplace=True)

# Select necessary columns
df1 = df[['track_name', 'artist_name', 'spotify_link', 'album', 'duration']]

# Remove duplicates and reset index
df1.drop_duplicates(subset=['track_name', 'artist_name', 'album'], inplace=True)
df1.reset_index(drop=True, inplace=True)

# Fill NaN values in text columns with empty strings to avoid errors with TfidfVectorizer
df1['track_name'] = df1['track_name'].fillna("")
df1['artist_name'] = df1['artist_name'].fillna("")
df1['album'] = df1['album'].fillna("")

# Convert spotify URI to full URL
df1['spotify_link'] = df1['spotify_link'].apply(lambda x: f"https://open.spotify.com/track/{x.split(':')[-1]}")

# Fit the TfidfVectorizer on the combined relevant columns
tfidf = TfidfVectorizer()
combined_data = df1['track_name'] + ' ' + df1['artist_name'] + ' ' + df1['album']
vectorizer = tfidf.fit_transform(combined_data)

# Convert duration to minutes and seconds
df1['formatted_duration'] = df1['duration'].apply(lambda x: f"{int(x // 60000)}:{int((x % 60000) / 1000):02d}")

# Function to get recommendations based on user input
def get_recommendations(user_input, tfidf, vectorizer, df, num_recommendations=10):
    user_vector = tfidf.transform([user_input])
    user_similarity = cosine_similarity(user_vector, vectorizer)
    
    # Find the top matches
    similar_indices = user_similarity.argsort()[0][-num_recommendations:][::-1]
    
    # Retrieve and display recommendations
    recommendations = df.iloc[similar_indices][['track_name', 'artist_name', 'album', 'formatted_duration', 'spotify_link']]
    return recommendations

# Streamlit app layout
st.title("Hindi Hits: Discover Your Favorite Tunes!")
st.write("Search for a song name, artist name, or album to get recommendations:")

# Loading animation
def load_animation():
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"Loading... {i+1}%")
        time.sleep(0.01)
    status_text.empty()
    progress_bar.empty()

# Add custom CSS to adjust the font size and box alignment
st.markdown("""
    <style>
        .song-title {
            font-size: 16px !important;
            font-weight: bold;
        }
        .song-box {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            text-align: center;
            height: 100%;
        }
    </style>
    """, unsafe_allow_html=True)
st.sidebar.subheader("No Idea What To Listen ? Here are some popular artists!")
st.sidebar.markdown(
    """
    <iframe style="border-radius:12px" src="https://open.spotify.com/embed/artist/4oVMLzAqW6qhRpZWt8fNw4?utm_source=generator&theme=0" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
    <iframe style="border-radius:20px" src="https://open.spotify.com/embed/playlist/37i9dQZF1DZ06evO0FcUGj?utm_source=generator&theme=0" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
    <iframe style="border-radius:20px" src="https://open.spotify.com/embed/playlist/37i9dQZF1DZ06evO0lbUOX?utm_source=generator&theme=0" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
    <iframe style="border-radius:20px" src="https://open.spotify.com/embed/artist/2jqTyPt0UZGrthPF4KMpeN?utm_source=generator&theme=0" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
    """,
    unsafe_allow_html=True
)


# User input
user_input = st.text_input("Search:", placeholder="e.g. Kishore Kumar, R D Burman").strip()

if st.button("Get Recommendations"):
    load_animation()
    if user_input:
        recommendations = get_recommendations(user_input, tfidf, vectorizer, df1, num_recommendations=10)
        if not recommendations.empty:
            
            st.write("### Recommended Songs:")
            # Display recommendations in 2 rows with 5 columns each
            columns = st.columns(5)
            for idx, (index, row) in enumerate(recommendations.iterrows()):
                with columns[idx % 5]:  # Cycle through columns to arrange in rows
                    st.markdown(f"""
                    <div class="song-box">
                        <p class="song-title">{row['track_name']}</p>
                        <p>Artist: {row['artist_name']}</p>
                        <p>Album: {row['album']}</p>
                        <p>Duration: {row['formatted_duration']}</p>
                        <p><a href="{row['spotify_link']}" target="_blank" style="text-decoration: none; color: blue;">Listen on Spotify</a></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.write("No recommendations found.")
    else:
        st.write("Please enter a song name, artist name, or album.")
