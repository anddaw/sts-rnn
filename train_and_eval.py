import argparse

import matplotlib.pyplot as plt


import yaml
from torch import nn, optim

from data_loading import load_embeddings, load_labels, load_sentence_pairs, embed_sentences
from models.vrnn_linear import VRNNLinear

from scipy import stats


def load_config(config_file_path: str):
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file)

    return config


def eval_model(model, corpus, gt_scores):
    pred_scores = [model.predict_score(s) for s in corpus]
    r, _ = stats.pearsonr(pred_scores, gt_scores)
    return r * 100


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    print('Loading embeddings')
    embeddings = load_embeddings(config['embeddings'])
    print('Loading datasets')
    train_sentences_path, train_labels_path = config['trainset']
    train_sentences = embed_sentences(load_sentence_pairs(train_sentences_path), embeddings)
    train_labels, train_similarities = load_labels(train_labels_path)

    test_sentences_path, test_labels_path = config['testset']
    test_sentences = embed_sentences(load_sentence_pairs(test_sentences_path), embeddings)
    test_labels, test_similarities = load_labels(test_labels_path)

    model = VRNNLinear(hidden_size=50, embedding_size=50, output_size=6)
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_scores = []
    test_scores = []
    losses = []
    for epoch in range(50):
        print(f'Epoch {epoch}')
        running_loss = 0
        i = 0
        for train, label in zip(train_sentences, train_labels):
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

        figure, subplots = plt.subplots(2, sharex=True)
        subplots[0].plot(train_scores, label='trainset')
        subplots[0].plot(test_scores, label='testset')
        subplots[0].set_title('Scores')
        subplots[0].legend()
        subplots[1].plot(losses)
        subplots[1].set_title('Loss')
        figure.show()
