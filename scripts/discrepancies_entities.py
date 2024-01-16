import sys

def main():
    file_1 = sys.argv[1]
    file_2 = sys.argv[2]

    file_1_set = set(); tweet_1 = set()
    file_2_set = set(); tweet_2 = set()

    with open(file_1) as fp:
        for line in fp:
            tw_id, ent_id =  line.strip().split('\t')
            file_1_set.add((tw_id, ent_id))
            tweet_1.add(tw_id)

    with open(file_2) as fp:
        for line in fp:
            tw_id, ent_id =  line.strip().split('\t')
            file_2_set.add((tw_id, ent_id))
            tweet_2.add(tw_id)

    print(len(file_1_set), len(file_2_set), len(file_1_set & file_2_set))
    print(len(tweet_1), len(tweet_2),  len(tweet_1 & tweet_2))

if __name__ == "__main__":
    main()
