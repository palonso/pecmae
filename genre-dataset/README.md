# Sources for genre taxonomies

AllMusic genres are taken from previous project (parsed with a script
https://github.com/MTG/metadb/blob/master/metadb/scrapers/allmusicgenre.py).

Discogs genres are taken from Discogs-Youtube dataset. They were parsed from the release metadata dump.

Wikipedia genres are semi-manually parsed from https://en.wikipedia.org/wiki/List_of_music_genres_and_styles. Only "Popular" category (popular music) is used for simplicity.

# Genre prototypes

`selected-genres.yaml` contains the list of preselected genres (common genres, present both in Discogs 400 styles and AllMusic).

We use tracks listed in the "Song Highlights" for each genre on AllMusic as prototypes ([example](https://www.allmusic.com/style/ambient-ma0000002571/songs)).


# Creating the genre dataset with Spotify API
- The file `selected-genres.yaml` contains a list of genres preselected for building the dataset.
- Create `selected-genres` file (line-delimited strings, each line string is a genre, see `src/test_genre_list` for an example) from `selected-genres.yaml`.
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
- Stats for the downloaded previews:
    ```
    find ../../spotify-audio/ -name "*.mp3" | sed 's/spotify-audio\///g' | sed 's/\/audio.*//' | uniq -c
    ```
- Search genre prototypes on Spotify API. If there is no candidate track with exact string match for artist name and title, the script asks to manually select the correct match. The script appends results to the `../../prototypes-AM-selected-genres.yaml.spotifyapi.tsv` file (FIXME filenames are hardcoded):
    ```
    python3 search_prototypes.py

    ```
- Number of prototypes successfully matched Spotify API metadata:
    ```
    cat ../../prototypes-AM-selected-genres.yaml.spotifyapi.tsv | grep  -v "NO MATCH" | wc -l
    ```
- Download previews for matched genre prototypes:
    ```
    mkdir ../../spotify-prototypes
    ./download_spotifyids_prototypes.sh ../../prototypes-AM-selected-genres.yaml.spotifyapi.tsv ../../spotify-prototypes
    ```
- Stats for the downloaded prototypes:
    ```
     find ../../spotify-prototypes/ -name "*.mp3" | sed 's/.*prototypes\///g' | sed 's/\/audio.*//' | uniq -c
    ```
