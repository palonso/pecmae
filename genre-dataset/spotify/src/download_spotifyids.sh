SPOTIFYIDS_TSV=$1
OUTPUT_DIR=$2

if [ ! -d "$2" ]; then
    mkdir "$2"
fi

while IFS=$'\t' read -r genre spotifyid artist track; do
    genre="${genre#*:}"
    echo "Downloading: $genre -- $spotifyid -- $artist -- $track"
    if [ ! -d "$2/$genre" ]; then
        mkdir "$2/$genre"
    fi
    python3 fetch_tracks.py -t $spotifyid -o $2/$genre -u --skip_annotations
done < "$SPOTIFYIDS_TSV"

