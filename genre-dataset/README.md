# Sources for genre taxonomies

AllMusic genres are taken from previous project (parsed with a script
https://github.com/MTG/metadb/blob/master/metadb/scrapers/allmusicgenre.py).

Discogs genres are taken from Discogs-Youtube dataset. They were parsed from the release metadata dump.

Wikipedia genres are semi-manually parsed from https://en.wikipedia.org/wiki/List_of_music_genres_and_styles. Only "Popular" category (popular music) is used for simplicity.

# Genre prototypes

`selected-genres.yaml` contains the list of preselected genres (common genres, present both in Discogs 400 styles and AllMusic).

We use tracks listed in the "Song Highlights" for each genre on AllMusic as prototypes ([example](https://www.allmusic.com/style/ambient-ma0000002571/songs)).


# Creating the genre dataset with Spotify API
- `selected-genres.yaml` contains a list of genres preselected for building the dataset.
- Create `selected-genres` file (each line string is a genre, see `src/test_genre_list` for an example) from `selected-genres.yaml`.
- Register an application in the Spotify API dashboard: https://developer.spotify.com/dashboard.
- Set your `SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET`, `SPOTIPY_REDIRECT_URI` environment variables.
- `cd spotify/src`
- Fetch Spotify API previews for each genre in the genre list `selected-genres`:
    ```
    ./query_genre_spotifyids.sh ../../selected-genres > ../../selected-genres.spotifyapi.tsv
    cat ../../selected-genres.spotifyapi.tsv | grep "genre:" | grep -v "query: " > ../../selected-genres.spotifyapi.clean.tsv
    ```
- Download Spotify API previews:
    ```
    mkdir ../../spotify-audio
    ./download_spotifyids.sh ../../selected-genres.spotifyapi.clean.tsv ../../spotify-audio
    ```
