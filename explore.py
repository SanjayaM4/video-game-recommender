import json

with open("data/raw/games.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("Number of games:", len(dataset))