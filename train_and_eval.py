import argparse
import yaml

from data_loading import load_embeddings, load_labels, load_sentence_pairs, embed_sentences


def load_config(config_file_path: str):
    with open(config_file_path, 'r') as config_file:
        config = yaml.load(config_file)

    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    embeddings = load_embeddings(config['embeddings'])
    train_sentences_path, train_labels_path = config['trainset']
    train_sentences = embed_sentences(load_sentence_pairs(train_sentences_path), embeddings)
    train_labels = load_labels(train_labels_path)

    print()
