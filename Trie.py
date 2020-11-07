from __future__ import annotations
from typing import Tuple, Dict, List


class TrieNode:
    """
    Implementation of a trie node.
    """

    __slots__ = "children", "is_end"

    def __init__(self, arr_size: int = 26) -> None:
        """
        Constructs a TrieNode with arr_size slots for child nodes.
        :param arr_size: Number of slots to allocate for child nodes.
        :return: None
        """
        self.children = [None] * arr_size
        self.is_end = 0

    def __str__(self) -> str:
        """
        Represents a TrieNode as a string.
        :return: String representation of a TrieNode.
        """
        if self.empty():
            return "..."
        children = self.children  # to shorten proceeding line
        return str({chr(i + ord("a")) + "*"*min(children[i].is_end, 1): children[i] for i in range(26) if children[i]})

    def __repr__(self) -> str:
        """
        Represents a TrieNode as a string.
        :return: String representation of a TrieNode.
        """
        return self.__str__()

    def __eq__(self, other: TrieNode) -> bool:
        """
        Compares two TrieNodes for equality.
        :return: True if two TrieNodes are equal, else False
        """
        if not other or self.is_end != other.is_end:
            return False
        return self.children == other.children

    def empty(self) -> bool:
        """
        Checks to see if TrieNode is empty
        :return: Return value here.
        """
        for char in self.children:
            if char is not None:
                return False
        return True

    @staticmethod
    def _get_index(char: str) -> int:
        """
        Returns index of given character
        :param char: Param description here.
        :return: Return value here.
        """
        if ord(char) <= 90:
            return ord(char) - ord('A')
        return ord(char) - ord('a')

    def get_child(self, char: str) -> TrieNode:
        """
        Returns child of current node at given character index
        :param char: Param description here.
        :return: Return value here.
        """
        return self.children[self._get_index(char)]

    def set_child(self, char: str) -> None:
        """
        Sets child of current node at given character index to TrieNode()
        :param char: Param description here.
        :return: Return value here.
        """
        self.children[self._get_index(char)] = TrieNode()

    def delete_child(self, char: str) -> None:
        """
        Set child of current node at given character index to None (delete)
        :param char: Param description here.
        :return: Return value here.
        """
        self.children[self._get_index(char)] = None


class Trie:
    """
    Implementation of a trie.
    """

    __slots__ = "root", "unique", "size"

    def __init__(self) -> None:
        """
        Constructs an empty Trie.
        :return: None.
        """
        self.root = TrieNode()
        self.unique = 0
        self.size = 0

    def __str__(self) -> str:
        """
        Represents a Trie as a string.
        :return: String representation of a Trie.
        """
        return "Trie Visual:\n" + str(self.root)

    def __repr__(self) -> str:
        """
        Represents a Trie as a string.
        :return: String representation of a Trie.
        """
        return self.__str__()

    def __eq__(self, other: Trie) -> bool:
        """
        Compares two Tries for equality.
        :return: True if two Tries are equal, else False
        """
        return self.root == other.root

    def add(self, word: str) -> int:
        """
        Add word to this instance of a Trie
        :param word: Param description here.
        :return: Return value here.
        """
        def add_inner(node: TrieNode, index: int) -> int:
            if index == len(word):
                if node.is_end == 0:
                    self.unique += 1
                self.size += 1
                node.is_end += 1
                return node.is_end
            if node.get_child(word[index]):
                return add_inner(node.get_child(word[index]), index + 1)
            if not node.get_child(word[index]):
                node.set_child(word[index])
                return add_inner(node.get_child(word[index]), index + 1)
        return add_inner(self.root, 0)

    def search(self, word: str) -> int:
        """
        Search for word passed in throughout current Trie
        :param word: Param description here.
        :return: Return value here.
        """
        def search_inner(node: TrieNode, index: int) -> int:
            if index == len(word):
                return node.is_end
            if not node.get_child(word[index]):
                return 0
            if node.get_child(word[index]):
                return search_inner(node.get_child(word[index]), index + 1)
        return search_inner(self.root, 0)

    def delete(self, word: str) -> int:
        """
        Delete word passed in from the current Trie
        :param word: Param description here.
        :return: Return value here.
        """
        def delete_inner(node: TrieNode, index: int) -> Tuple[int, bool]:
            #Base Case
            if node and index == len(word) and node.is_end > 0:
                node_counter = node.is_end
                self.unique -= 1
                self.size -= node_counter
                node.is_end = 0
                if not node.empty():
                    return node_counter, False
                if node.empty():
                    return node_counter, True

            #Recursive Case
            if node.get_child(word[index]):
                occurrences, prune_children = delete_inner(node.get_child(word[index]), index + 1)

                if prune_children: # We do prune children
                    node.children[node._get_index(word[index])] = None #Prune Child

                    if node.is_end == 0 and node.empty():
                        prune_current = True
                    else:
                        prune_current = False

                else: # We do not prune children
                    prune_current = False

                return occurrences, prune_current

            else:
                return 0, False

        return delete_inner(self.root, 0)[0]

    def __len__(self) -> int:
        """
        Get size of Trie
        :return: Return value here.
        """
        return self.size

    def __contains__(self, word: str) -> bool:
        """
        Check if Trie contains word
        :param word: Param description here.
        :return: Return value here.
        """
        return self.search(word) > 0

    def empty(self) -> bool:
        """
        Check of size of Trie is zero
        :return: Return value here.
        """
        return self.size == 0

    def get_vocabulary(self, prefix: str = "") -> Dict[str, int]:
        """
        Get all words and their occurrences in current Trie
        :param prefix: Param description here.
        :return: Return value here.
        """
        vocab = dict()
        level = 0
        word = [None] * self.unique

        def get_vocabulary_inner(node: TrieNode, suffix: str):
            nonlocal vocab
            nonlocal level
            nonlocal word

            if node.is_end > 0:
                for i in range(level):
                    suffix += word[i]
                vocab[prefix + suffix] = node.is_end
                suffix = ""

            for i in range(26):
                if node.children[i]:
                    word[level] = chr(i + ord('a'))
                    level += 1
                    get_vocabulary_inner(node.children[i], suffix)
            level -= 1

        def get_prefix(node: TrieNode, index: int):
            if index == len(prefix):
                return node
            if not node.get_child(prefix[index]):
                return 0
            if node.get_child(prefix[index]):
                return get_prefix(node.get_child(prefix[index]), index + 1)

        parent = get_prefix(self.root, 0)
        if parent == 0:
            return vocab
        get_vocabulary_inner(parent, "")
        return vocab

    def autocomplete(self, word: str) -> Dict[str, int]:
        """
        Perform autocomplete to fill in missing spots in given word using words in current Trie
        :param word: Param description here.
        :return: Return value here.
        """
        vocab = dict()

        def autocomplete_inner(node: TrieNode, prefix: str, index: int):
            nonlocal vocab

            # Base Case
            if node and index + 1 == len(word) - 1 and (word[index+1] == '.' or (node.get_child(word[index+1]) and node.get_child(word[index+1]).is_end > 0)):

                if word[index+1] == '.':
                    for i in range(len(node.children)):
                        if node.children[i] and node.children[i].is_end > 0:
                            b = prefix + chr(i + ord('a'))
                            vocab[b] = node.children[i].is_end

                else:
                    if node.get_child(word[index+1]).is_end > 0:
                        b = prefix + word[index+1]
                        vocab[b] = node.get_child(word[index+1]).is_end

            if node and index + 1 != len(word) and word[index+1] == '.':
                for i in range(len(node.children)):
                    if node.children[i]:
                        child_char = chr(i + ord('a'))
                        prefix += child_char
                        autocomplete_inner(node.get_child(str(child_char)), prefix, index + 1)
                        prefix = prefix[0:len(prefix)-1]

            if node and index + 1 != len(word) and word[index+1] != '.':
                if node.get_child(word[index+1]):
                    prefix += word[index+1]
                    autocomplete_inner(node.get_child(word[index+1]), prefix, index + 1)

        if self.empty() or len(word) == 0:
            return vocab

        if word[0] != '.':
            second_node = self.root.get_child(word[0])
            if second_node:
                autocomplete_inner(second_node, str(word[0]), 0)
        else:
            for i in range(len(self.root.children)):
                if self.root.children[i]:
                    child_char = chr(i + ord('a'))
                    autocomplete_inner(self.root.get_child(str(child_char)), str(child_char), 0)

        return vocab


class TrieClassifier:
    """
    Implementation of a trie-based text classifier.
    """

    __slots__ = "tries"

    def __init__(self, classes: List[str]) -> None:
        """
        Constructs a TrieClassifier with specified classes.
        :param classes: List of possible class labels of training and testing data.
        :return: None.
        """
        self.tries = {}
        for cls in classes:
            self.tries[cls] = Trie()

    @staticmethod
    def accuracy(labels: List[str], predictions: List[str]) -> float:
        """
        Computes the proportion of predictions that match labels.
        :param labels: List of strings corresponding to correct class labels.
        :param predictions: List of strings corresponding to predicted class labels.
        :return: Float proportion of correct labels.
        """
        correct = sum([1 if label == prediction else 0 for label, prediction in zip(labels, predictions)])
        return correct / len(labels)

    def fit(self, class_strings: Dict[str, List[str]]) -> None:
        """
        Fill TrieClassifier with the words from the "Tweets" in class_strings
        :param class_strings: Param description here.
        :return: Return value here.
        """
        for key in class_strings:
            for string in class_strings[key]:
                for word in string.split():
                    self.tries[key].add(word)

    def predict(self, strings: List[str]) -> List[str]:
        """
        Predict the sentiment of the "Tweets" passed in
        :param strings: Param description here.
        :return: Return value here.
        """
        predictions = []
        predict_dict = {key: 0 for key in self.tries.keys()}
        for string in strings:
            split_string = string.split()
            for key in self.tries.keys():
                for word in split_string:
                    predict_dict[key] += (self.tries[key].search(word) / len(self.tries[key]))
            predictions.append(max(predict_dict, key=predict_dict.get))
            predict_dict = {key: 0 for key in self.tries.keys()}
        return predictions
