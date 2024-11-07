import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

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

# Fit the TfidfVectorizer on the combined relevant columns
tfidf = TfidfVectorizer()
combined_data = df1['track_name'] + ' ' + df1['artist_name'] + ' ' + df1['album']
vectorizer = tfidf.fit_transform(combined_data)

# Convert duration to minutes and seconds
df1['formatted_duration'] = df1['duration'].apply(lambda x: f"{int(x // 60000)}:{int((x % 60000) / 1000):02d}")

# Function to get recommendations based on user input
def get_recommendations(user_input, tfidf, vectorizer, df, num_recommendations=5):
    user_vector = tfidf.transform([user_input])
    user_similarity = cosine_similarity(user_vector, vectorizer)
    
    # Find the top matches
    similar_indices = user_similarity.argsort()[0][-num_recommendations:][::-1]
    
    # Retrieve and display recommendations
    recommendations = df.iloc[similar_indices][['track_name', 'artist_name', 'album', 'formatted_duration', 'spotify_link']]
    return recommendations


# Streamlit app layout
st.title("Hindi Hits: Discover Your Favorite Tunes!")  # Changed title
st.write("Search for a song name, artist name, or album to get recommendations:")

# User input
user_input = st.text_input("Search:").strip()

if st.button("Get Recommendations"):
    if user_input:
        recommendations = get_recommendations(user_input, tfidf, vectorizer, df1)
        if not recommendations.empty:
            st.write("### Recommended Songs:")
            for index, row in recommendations.iterrows():
                st.write(f"**{row['track_name']}** by {row['artist_name']} - "
                         f"Duration: {row['formatted_duration']} - "
                         f"[Listen on Spotify]({row['spotify_link']})")
        else:
            st.write("No recommendations found.")
    else:
        st.write("Please enter a song name, artist name, or album.")
