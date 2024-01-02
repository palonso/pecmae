SPOTIFYIDS_TSV=$1
OUTPUT_DIR=$2

if [ ! -d "$2" ]; then
    mkdir "$2"
fi

while IFS=$'\t' read -r genre artist track spotifyid artist_spotify track_spotify; do
    echo "Downloading: $genre -- $spotifyid -- $artist_spotify -- $track_spotify"
    if [ ! -d "$2/$genre" ]; then
        mkdir "$2/$genre"
    fi
    python3 fetch_tracks.py -t $spotifyid -o "$2/$genre" -u --skip_annotations
done < "$SPOTIFYIDS_TSV"

