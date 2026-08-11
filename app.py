import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)


# Load data
@st.cache_data
def load_data():
    with open("songs.pkl", "rb") as f:
        songs = pickle.load(f)

    with open("X.pkl", "rb") as f:
        X = pickle.load(f)

    return songs, X


songs, X = load_data()

songs["search_title"] = songs["track_name"].astype(str).str.lower()


@st.cache_data(ttl=86400)
def get_cover_url(song, artist):
    try:
        response = requests.get(
            "https://itunes.apple.com/search",
            params={
                "term": f"{song} {artist}",
                "media": "music",
                "entity": "song",
                "limit": 1
            },
            timeout=10
        )

        data = response.json()

        if data["resultCount"] == 0:
            return None

        image = data["results"][0]["artworkUrl100"]
        return image.replace("100x100", "600x600")

    except Exception:
        return None


def recommend_song(song_name, n=10):

    matches = songs[
        songs["search_title"] == song_name.lower()
    ]

    if matches.empty:
        return pd.DataFrame()

    index = matches.index[0]

    similarity = cosine_similarity(
        X[index],
        X
    ).flatten()

    results = songs.copy()
    results["similarity"] = similarity

    # Give a small bonus to songs from the same artist
    input_artists = set(
        str(songs.loc[index, "artists"]).split(";")
    )

    def artist_match(artists):
        current_artists = set(
            str(artists).split(";")
        )
        return bool(input_artists & current_artists)

    results["same_artist"] = results["artists"].apply(
        artist_match
    )

    results["score"] = (
        results["similarity"]
        + results["same_artist"].astype(float) * 0.01
    ).clip(upper=1.0)

    # Don't recommend the song itself
    results = results[results.index != index]

    # Remove different versions of the same song
    title = songs.loc[index, "clean_title"]
    results = results[results["clean_title"] != title]

    results = results.drop_duplicates(
        subset="clean_title"
    )

    results = results.sort_values(
        "score",
        ascending=False
    )

    return results.head(n)[
        [
            "track_name",
            "artists",
            "track_genre",
            "score"
        ]
    ].reset_index(drop=True)


# App
st.title("🎵 Music Recommendation System")

st.write(
    "Find songs similar to your favorite song "
    "using audio features and genre similarity."
)

song_search = st.text_input(
    "Search song name",
    placeholder="e.g. Believer"
)

song = None

if song_search.strip():

    search_text = song_search.strip().lower()

    matches = songs[
        songs["search_title"].str.contains(
            search_text,
            na=False,
            regex=False
        )
    ]["track_name"].drop_duplicates().head(10).tolist()

    if matches:
        song = st.selectbox(
            "Select your song",
            matches
        )
    else:
        st.warning("No songs found. Try a different spelling.")


if st.button(
    "🎧 Recommend Songs",
    use_container_width=True
):

    if song is None:
        st.warning("Search for a song first.")

    else:

        with st.spinner("Finding similar songs..."):
            recommendations = recommend_song(song)

        st.subheader(
            f"🎵 Songs similar to {song}"
        )

        if recommendations.empty:
            st.warning("No recommendations found.")

        else:

            for i, row in recommendations.iterrows():

                col1, col2 = st.columns([1, 4])

                with col1:

                    cover = get_cover_url(
                        row["track_name"],
                        row["artists"]
                    )

                    if cover:
                        st.image(cover, width=150)
                    else:
                        st.write("🎵")

                with col2:

                    st.markdown(
                        f"### {i + 1}. {row['track_name']}"
                    )

                    st.write(
                        f"**Artist:** {row['artists']}"
                    )

                    st.write(
                        f"**Genre:** {row['track_genre']}"
                    )

                    st.progress(
                        min(float(row["score"]), 1.0)
                    )

                    st.caption(
                        f"Recommendation score: {row['score']:.2%}"
                    )

                st.divider()