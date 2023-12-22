query_genre_list=$1

while IFS= read -r query_genre
do
    echo "--------- $query_genre ----------"
    #for offset in {0..950..50}
    for offset in {0..150..50}
    do
        python3 search_by_query.py -q "$query_genre" -f genre -l 50 --offset=$offset
    done
done < "$query_genre_list"
