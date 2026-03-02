import streamlit as st
import pandas as pd
from src.recommender import SteamRecommender

@st.cache_resource
def get_recommender():
    return SteamRecommender()

st.set_page_config(page_title="Steam Recommender", page_icon="🎮", layout="wide")

def main():
    st.title("🎮 Steam Game Recommender", anchor=False)
    
    with st.spinner("Loading Library..."):
        recommender = get_recommender()

    all_game_names = recommender.df['Name'].unique()
    selected_game = st.selectbox("Type a game you like:", [""] + list(all_game_names))

    if selected_game:
        st.divider()
        results = recommender.recommend(selected_game)

        if results is not None:
            cols = st.columns(3) 
            
            for i, (idx, row) in enumerate(results.iterrows()):
                with cols[i % 3]:
                    with st.container(border=True):
                        if pd.notnull(row['HeaderImage']):
                            st.image(row['HeaderImage'], use_container_width=True)
                        
                        st.subheader(row['Name'], anchor=False)
        
                        stat1, stat2 = st.columns(2)
                        stat1.metric("👍 Positive", f"{row['Positive']:,}")
                        stat2.metric("👎 Negative", f"{row['Negative']:,}")
        else:
            st.error("Game not found in database.")

if __name__ == "__main__":
    main()