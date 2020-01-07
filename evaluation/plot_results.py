import argparse

import matplotlib.pyplot as plt
import pandas


def plot_scores(outfile, x_values, pearson, fscore, title='Tytuł'):

    f, ax = plt.subplots(1)
    plt.title(title)
    ax.plot(x_values, pearson, '--bo', label='r')

    ax.set_ylim(bottom=0, top=max(pearson) * 1.1)
    ax.legend()
    plt.grid(linestyle='-')
    # if fscore[0] != '-':
    #     plt.plot(x_values, fscore * 10**2)
    plt.savefig(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('-t', '--title', default='Tytuł')
    args = parser.parse_args()

    df = pandas.read_csv(args.infile, sep='\t')

    plot_scores(args.outfile, df['param'], df['pearson'], df['fscore'])

