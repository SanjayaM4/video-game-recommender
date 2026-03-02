import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import PROCESSED_PATH, load_and_clean_json

class SteamRecommender:
    def __init__(self):
        # Load the data
        if not os.path.exists(PROCESSED_PATH):
            print("Processed data not found. Running preprocessor...")
            self.df = load_and_clean_json()
        else:
            print("Loading optimized dataset...")
            self.df = pd.read_parquet(PROCESSED_PATH)

        # --- ADD THIS: Calculate Quality Score (Bayesian-ish Average) ---
        print("Calculating quality scores...")
        
        # Total number of reviews (votes)
        v = self.df['Positive'] + self.df['Negative']
        # Average rating (Percentage of positive reviews)
        R = self.df['Positive'] / (v + 1e-6) # 1e-6 avoids division by zero
        # Minimum reviews required to be "reliable" (we'll use the median)
        m = v.median()
        # The mean positive percentage across the whole dataset
        C = R.mean()
        
        # Weighted Rating formula
        self.df['quality_score'] = (v / (v + m) * R) + (m / (v + m) * C)
        # ---------------------------------------------------------------

        print("Vectorizing features...")
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_features"])
        print("Recommender ready.")
        

    def recommend(self, game_name, top_n=10):
        query = game_name.strip().lower()
        
        # Check for matches in the "Name" column
        matches = self.df[self.df["Name"].str.lower() == query]
        
        if matches.empty:
            matches = self.df[self.df["Name"].str.contains(query, case=False, na=False, regex=False)]

        if matches.empty:
            print(f"Could not find '{game_name}'.")
            return None

        idx = matches.index[0]
        
        # 1. Content Similarity (TF-IDF)
        content_scores = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        
        # 2. Hybrid Calculation
        # We multiply similarity by the quality score to "boost" good games
        # We use a small 'alpha' to ensure we don't ignore niche games entirely
        alpha = 0.5
        hybrid_scores = content_scores * (1 + alpha * self.df['quality_score'])

        # 3. Get Top Results
        similar_indices = hybrid_scores.argsort()[-(top_n + 1):][::-1]
        similar_indices = [i for i in similar_indices if i != idx]

        return self.df.iloc[similar_indices][["Name", "Positive", "Negative", "Price"]]

if __name__ == "__main__":
    recommender = SteamRecommender()
    while True:
        game_input = input("\nEnter a game (or 'q' to quit): ")
        if game_input.lower() == 'q': break
        
        results = recommender.recommend(game_input)
        if results is not None:
            print(results.to_string(index=False))