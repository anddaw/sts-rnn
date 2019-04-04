import argparse
import re

import matplotlib.pyplot as plt


import yaml
from torch import nn, optim

from data_loading import load_dataset
from models.pick_model import pick_model

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

    max_train = max(train_scores_)
    for i, score in enumerate(train_scores_):
        if score == max_train:
            subplots[0].annotate(f'{score:.2f}', (i, score), textcoords='data')
    max_test = max(test_scores_)
    for i, score in enumerate(test_scores_):
        if score == max_test:
            subplots[0].annotate(f'{score:.2f}', (i, score), textcoords='data')

    subplots[0].set_title('Scores')
    subplots[0].legend()
    subplots[1].plot(losses_)
    subplots[1].set_title('Loss')
    figure.show()


def train_epoch(model, train_sentences, train_labels, loss_func, optimizer):
    i = 0
    loss_value = 0
    for train, label in sample(list(zip(train_sentences, train_labels)), len(train_sentences)):
        model.zero_grad()
        output = model.forward(train)
        loss = loss_func(output.view(1, -1), label.view(1, -1))
        loss.backward()
        optimizer.step()
        loss_value += loss.item()

        i += 1
        if i % 50 == 0:
            print(f'{i}/{len(train_labels)}')

    return loss_value / len(train_labels)


def stopping_criterion(scores, patience):
    return len(scores) > patience and max(scores[-patience:]) < scores[-patience-1]


def train_model(model, train_corpus, test_corpus, epochs, patience=3):

    train_sentences, train_labels, train_similarities = train_corpus
    test_sentences, test_labels, test_similarities = test_corpus
    loss_func = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters())

    train_scores = []
    test_scores = []
    losses = []

    for epoch in range(epochs):
        print(f'Epoch {epoch}')

        loss = train_epoch(model, train_sentences, train_labels, loss_func, optimizer)

        train_score = eval_model(model, train_sentences, train_similarities)
        test_score = eval_model(model, test_sentences, test_similarities)

        print(f'Loss after epoch {epoch}: {loss}')
        print(f'Score on training set: {train_score}')
        print(f'Score on test set: {test_score}')

        train_scores.append(train_score)
        test_scores.append(test_score)
        losses.append(loss)

        if stopping_criterion(test_scores, patience):
            break

    return train_scores, test_scores, losses


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    parser.add_argument('-o', '--option', nargs=2, action='append')

    args = parser.parse_args()
    config = load_config(args.config)

    options = args.option
    options = [] if not options else options
    for option in options:
        value = option[1]
        if re.match(r'-?[0-9]+\.?[0-9]*$', value):
            value = float(value) if '.' in value else int(value)
        config[option[0]] = value

    trainset = load_dataset('train', config)
    testset = load_dataset('test', config)

    scores = train_model(pick_model(config), trainset, testset, config['epochs'])

    plot_scores(*scores)
