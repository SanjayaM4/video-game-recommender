import pandas as pd
import os

RAW_PATH = "data/raw/games.json"
PROCESSED_PATH = "data/processed/games.parquet"

def load_and_clean_json():
    print(f"Reading {RAW_PATH}... this might take a minute.")
    
    # Load the JSON (Steam JSONs are usually orient='index')
    df = pd.read_json(RAW_PATH, orient='index')

    # Reset index to make the AppID (the keys) a column
    df = df.reset_index().rename(columns={'index': 'AppID'})

    # Use names as they appear in your JSON
    cols_to_keep = ["name", "genres", "tags", "categories", "positive", "negative", "price", "header_image"]
    df = df[cols_to_keep].copy()

    # Cleanup Names
    df = df.dropna(subset=["name"])
    df["name"] = df["name"].astype(str).str.strip()

    # Handle JSON specific structures
    # 'genres' and 'categories' are lists: ["Action", "Indie"]
    # 'tags' is a dict: {"Action": 50, "Indie": 20}
    
    def process_features(val):
        if isinstance(val, list):
            return " ".join(val)
        if isinstance(val, dict):
            return " ".join(val.keys()) # Take only the tag names, ignore the counts
        return str(val) if pd.notnull(val) else ""

    for col in ["genres", "tags", "categories"]:
        df[col] = df[col].apply(process_features)

    # Create the search string
    df["combined_features"] = (
        df["genres"] + " " + 
        df["tags"] + " " + 
        df["categories"]
    ).str.lower()

    # Rename columns back to Capitalized for UI
    df = df.rename(columns={
        "name": "Name",
        "positive": "Positive",
        "negative": "Negative",
        "header_image": "HeaderImage"
    })

    # Save to Parquet
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"Success! Processed data saved to {PROCESSED_PATH}")
    return df

if __name__ == "__main__":
    load_and_clean_json()