import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rapidfuzz import process

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

# Convert Spotify URI to full URL
df1['spotify_link'] = df1['spotify_link'].apply(lambda x: f"https://open.spotify.com/track/{x.split(':')[-1]}")

# Fit the TfidfVectorizer on the combined relevant columns
tfidf = TfidfVectorizer()
combined_data = df1['track_name'] + ' ' + df1['artist_name'] + ' ' + df1['album']
vectorizer = tfidf.fit_transform(combined_data)

# Convert duration to minutes and seconds
df1['formatted_duration'] = df1['duration'].apply(lambda x: f"{int(x // 60000)}:{int((x % 60000) / 1000):02d}")

# Function to get recommendations based on user input with fuzzy matching
def get_recommendations(user_input, tfidf, vectorizer, df, num_recommendations=10):
    # Use fuzzy matching to find closest matches for user input
    all_choices = df['track_name'].tolist() + df['artist_name'].tolist() + df['album'].tolist()
    best_matches = process.extract(user_input, all_choices, limit=5, score_cutoff=70)
    
    # Provide a "Did you mean" suggestion if no exact matches are found
    if best_matches:
        top_match = best_matches[0][0]  # Get the top match
        if user_input.lower() != top_match.lower():  # If input and top match differ significantly
            st.write(f"Did you mean **{top_match}**?")
        combined_query = ' '.join([match[0] for match in best_matches])
    else:
        st.write("No close matches found. Please check your input.")
        return pd.DataFrame()  # Return an empty DataFrame if no matches
    
    # Transform the combined query and calculate cosine similarity
    user_vector = tfidf.transform([combined_query])
    user_similarity = cosine_similarity(user_vector, vectorizer)
    
    # Find the top matches
    similar_indices = user_similarity.argsort()[0][-num_recommendations:][::-1]
    
    # Retrieve and display recommendations
    recommendations = df.iloc[similar_indices][['track_name', 'artist_name', 'album', 'formatted_duration', 'spotify_link']]
    return recommendations

# Streamlit app layout
import streamlit as st
import time

st.title("Hindi Hits: Discover Your Favorite Tunes!")
st.write("Search for a song name, artist name, or album to get recommendations:")

# Loading animation
def load_animation():
    progress_bar = st.progress(0)
    status_text = st.empty()
    music_icon = "🎵"  # Musical note emoji for theme

    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"{music_icon} Loading... {i+1}% {music_icon}")
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

st.sidebar.subheader("No Idea What To Listen To? Here are some popular artists!")
st.sidebar.markdown(
    """
    <iframe style="border-radius:20px" src="https://open.spotify.com/embed/artist/4oVMLzAqW6qhRpZWt8fNw4?utm_source=generator&theme=0" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
    <iframe style="border-radius:20px" src="https://open.spotify.com/embed/playlist/37i9dQZF1DZ06evO0FcUGj?utm_source=generator&theme=0" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
    <iframe style="border-radius:20px" src="https://open.spotify.com/embed/playlist/37i9dQZF1DZ06evO0lbUOX?utm_source=generator&theme=0" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
    <iframe style="border-radius:20px" src="https://open.spotify.com/embed/artist/2jqTyPt0UZGrthPF4KMpeN?utm_source=generator&theme=0" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
    <iframe style="border-radius:20px" src="https://open.spotify.com/embed/artist/4YRxDV8wJFPHPTeXepOstw?utm_source=generator&theme=0" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>
    """,    
    unsafe_allow_html=True
)

# Add a sidebar "About" section for contributors
with st.sidebar:
    st.header("About")
    st.write("This app was developed by:")
    st.write("[Your Name](https://github.com/your-github-id)")
    st.write("[Teammate 1](https://github.com/teammate1-github-id)")
    st.write("[Teammate 2](https://github.com/teammate2-github-id)")
    st.write("Explore our profiles for more projects!")

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

# Add a footer for contributors
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <h4>Developed by :</h4>
    <p>
        <a href='https://github.com/MICKEY0107' target='_blank'>Aniket Mishra</a> |
        <a href='https://github.com/aaradhya1205' target='_blank'>Aaradhya Gupta</a> |
    </p>
</div>
""", unsafe_allow_html=True)
