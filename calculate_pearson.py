from argparse import ArgumentParser

from metrics import pearson_r

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('predicted')
    parser.add_argument('ground_truths')

    args = parser.parse_args()

    with open(args.predicted) as fh:
        predicted = [float(r) for r in fh]

    with open(args.ground_truths) as fh:
        ground_truths = [float(r) for r in fh]

    print(pearson_r(predicted, ground_truths))
