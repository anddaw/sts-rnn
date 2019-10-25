from typing import Tuple, List

import torch

from nltk.tree import Tree
from anytree import AnyNode, RenderTree

from data_preprocessor import DataPreprocessor


class TreeNode(AnyNode):
    def __init__(self, parent=None, word: str = '', **kwargs):

        super().__init__(parent, **kwargs)
        self.word = word

    def __str__(self):
        return str(RenderTree(self))

class TreeReader:

    def __init__(self, preprocessor: DataPreprocessor):

        self.preprocessor = preprocessor

    def _tree_from_string(self, tree_string: str) -> TreeNode:

        def to_anytree(nltk_tree, parent=None):

            node = TreeNode()
            if isinstance(nltk_tree, str):
                node.word = nltk_tree
                node.parent = parent
            else:
                subtrees = list(nltk_tree)
                if len(subtrees) == 1 and parent is not None:
                    to_anytree(subtrees[0], parent)
                else:
                    node.parent = parent
                    for t in subtrees:
                        to_anytree(t, node)

            return node

        tree = to_anytree(Tree.fromstring(tree_string))
        return tree

    @staticmethod
    def _split_contents(contents: str) -> List[str]:
        return contents.split('\n\n')

    def load_from_file(self, file_path: str) -> List[Tuple[TreeNode, TreeNode]]:

        with open(file_path, 'r') as tree_file:
            tree_strings = self._split_contents(tree_file.read())

        trees = [self._tree_from_string(s) for s in tree_strings if s]

        return [tuple(trees[i:i+2]) for i in range(0, len(trees), 2)]
