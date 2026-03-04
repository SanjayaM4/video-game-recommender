import pandas as pd
import os

RAW_PATH = "data/raw/games.json"
PROCESSED_PATH = "data/processed/games.parquet"

def load_and_clean_json():
    print(f"Reading {RAW_PATH}... this might take a minute.")
    
    # Load the JSON
    df = pd.read_json(RAW_PATH, orient='index')

    # Reset index to make the AppID a column
    df = df.reset_index().rename(columns={'index': 'AppID'})

    # Select columns and rename them immediately for easier filtering
    cols_to_keep = ["name", "genres", "tags", "categories", "positive", "negative", "header_image"]
    df = df[cols_to_keep].copy()
    
    df = df.rename(columns={
        "name": "Name",
        "positive": "Positive",
        "negative": "Negative",
        "header_image": "HeaderImage"
    })

    # Remove rows without a name
    df = df.dropna(subset=["Name"])
    df["Name"] = df["Name"].astype(str).str.strip()

    # Filter out games with < 30 total reviews
    # This significantly improves recommendation quality by removing "junk"
    total_votes = df['Positive'] + df['Negative']
    initial_count = len(df)
    df = df[total_votes >= 30].copy()
    
    print(f"Data Pruning: Removed {initial_count - len(df)} games with < 30 reviews.")
    print(f"Remaining games in dataset: {len(df)}")

    # Process genres, tags, and categories
    def process_features(val):
        if isinstance(val, list):
            return " ".join(val)
        if isinstance(val, dict):
            return " ".join(val.keys()) # Extract tag names, ignore vote counts
        return str(val) if pd.notnull(val) else ""

    for col in ["genres", "tags", "categories"]:
        df[col] = df[col].apply(process_features)

    # Create the combined search string
    df["combined_features"] = (
        df["genres"] + " " + 
        df["tags"] + " " + 
        df["categories"]
    ).str.lower()

    # Save to high-performance Parquet format
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    
    print(f"Success! Optimized dataset saved to {PROCESSED_PATH}")
    return df

if __name__ == "__main__":
    load_and_clean_json()