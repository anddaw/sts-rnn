import argparse

import matplotlib.pyplot as plt


import yaml
from torch import nn, optim

from data_loading import load_dataset
from models.pick_model import pick_model
from models.vrnn_linear import VRNNLinear

from scipy import stats

from random import sample


def load_config(config_file_path: str):
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file)

    return config


def eval_model(model, corpus, gt_scores):
    pred_scores = [model.predict_score(s) for s in corpus]
    r, _ = stats.pearsonr(pred_scores, gt_scores)
    return r * 100


def plot_scores(train_scores_, test_scores_, losses_):
    figure, subplots = plt.subplots(2, sharex=True)
    subplots[0].plot(train_scores_, label='trainset')
    subplots[0].plot(test_scores_, label='testset')

    for i, score in enumerate(train_scores_):
        subplots[0].annotate(f'{score:.2f}', (i, score), textcoords='data')
    for i, score in enumerate(test_scores_):
        subplots[0].annotate(f'{score:.2f}', (i, score), textcoords='data')

    subplots[0].set_title('Scores')
    subplots[0].legend()
    subplots[1].plot(losses_)
    subplots[1].set_title('Loss')
    figure.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    train_sentences, train_labels, train_similarities = load_dataset('trainset', config)
    test_sentences, test_labels, test_similarities = load_dataset('testset', config)

    model = pick_model(config)
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_scores = []
    test_scores = []
    losses = []
    for epoch in range(config['epochs']):
        print(f'Epoch {epoch}')
        running_loss = 0
        i = 0
        for train, label in sample(list(zip(train_sentences, train_labels)), len(train_sentences)):
            model.zero_grad()
            output = model.forward(train)
            loss = loss_func(output.view(1, -1), label.view(1, -1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            i += 1
            if i % 50 == 0:
                print(f'{i}/{len(train_labels)}')

        train_score = eval_model(model, train_sentences, train_similarities)
        train_scores.append(train_score)
        test_score = eval_model(model, test_sentences, test_similarities)
        test_scores.append(test_score)
        avg_loss = running_loss/len(train_labels)
        losses.append(avg_loss)
        print(f'Loss after epoch {epoch}: {avg_loss}')
        print(f'Score on training set: {train_score}')
        print(f'Score on test set: {test_score}')

    plot_scores(train_scores, test_scores, losses)
