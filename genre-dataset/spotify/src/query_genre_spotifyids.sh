query_genre_list=$1

while IFS= read -r query_genre
do
    echo "--------- $query_genre ----------"
    #for offset in {0..950..50}
    for offset in {0..950..50}
    do
        python3 search_by_query.py -q "$query_genre" -f genre -l 50 --offset=$offset
        # Rate limit is based on a rolling 30 second window.
        sleep 30
    done
done < "$query_genre_list"
