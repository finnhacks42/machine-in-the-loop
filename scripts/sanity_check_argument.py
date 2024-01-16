import argparse
from tqdm import tqdm

def parse_predicate(pred):
    name, args = pred.split('(')
    args = args[:-1].split(',')
    return name, args

def main(args):
    mentions_argument = []
    with open(args.input_preds) as fp:
        lines = fp.readlines()
        pbar = tqdm(total=len(lines), desc='reading predictions...')

        for line in lines:
            name, args = parse_predicate(line.strip())
            if name == "MentionsArgument":
                twid = args[0]
                theme = args[1]

                if theme.startswith('KMeans') or theme == 'Unknown':
                    continue

                mentions_argument.append(twid)

    print(len(mentions_argument), len(set(mentions_argument)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_preds', type=str, required=True)
    args = parser.parse_args()
    main(args)
