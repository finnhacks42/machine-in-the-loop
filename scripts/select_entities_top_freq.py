import argparse
import json
from collections import Counter

def main(args):
    entity2lexform = json.load(open(args.entity2lexform))
    lexform2entities = {}
    lexforms = []

    with open(args.input) as fp:
        for line in fp:
            tw_id, ent_id = line.strip().split('\t')
            ent_str = entity2lexform[ent_id]
            if ent_str not in lexform2entities:
                lexform2entities[ent_str] = []
            lexform2entities[ent_str].append((tw_id, ent_id))
            lexforms.append(ent_str)

    counter = Counter(lexforms)
    ls = [strs for strs in counter if counter[strs] > 500]
    print(len(counter))
    print(len(ls))

    with open(args.output, "w") as fp:
        for strs in ls:
            ids = lexform2entities[strs]
            for (tw_id, ent_id) in ids:
                fp.write('{}\t{}\n'.format(tw_id, ent_id))
        fp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity2lexform', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    main(args)
