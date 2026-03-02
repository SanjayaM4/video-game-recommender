import pandas as pd

def load_games():
    df = pd.read_csv("data/raw/games.csv")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)
    return df

if __name__ == "__main__":
    df = load_games()
    print("\nFirst 5 rows:")
    print(df.head())