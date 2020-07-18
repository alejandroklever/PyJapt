class ContainerSet:
    """
    Special class used for handling the existence of epsilon in a set of Symbols. It has all properties of a set
    """

    def __init__(self, *values, contains_epsilon: bool = False):
        self.set: set = set(values)
        self.contains_epsilon: bool = contains_epsilon

    def add(self, value) -> bool:
        """
        Add a new value to the ContainerSet and return true if size was changed
        :param value: value to be added
        :return: True if the item was added
        """
        n = len(self.set)
        self.set.add(value)
        return n != len(self.set)

    def extend(self, values) -> bool:
        n = len(self.set)
        self.set.update(values)
        return n != len(self.set)

    def set_epsilon(self, value=True) -> bool:
        last = self.contains_epsilon
        self.contains_epsilon = value
        return last != self.contains_epsilon

    def update(self, other) -> bool:
        n = len(self.set)
        self.set.update(other.set)
        return n != len(self.set)

    def epsilon_update(self, other) -> bool:
        return self.set_epsilon(self.contains_epsilon | other.contains_epsilon)

    def hard_update(self, other):
        return self.update(other) | self.epsilon_update(other)

    def __contains__(self, item):
        return item in self.set

    def __len__(self):
        return len(self.set) + int(self.contains_epsilon)

    def __str__(self):
        return '%s-%s' % (str(self.set), self.contains_epsilon)

    def __repr__(self):
        return str(self)

    def __iter__(self):
        return iter(self.set)

    def __nonzero__(self):
        return len(self) > 0

    def __eq__(self, other):
        if isinstance(other, set):
            return self.set == other
        return isinstance(other,
                          ContainerSet) and self.set == other.set and self.contains_epsilon == other.contains_epsilon


class TrieNode:
    def __init__(self, symbol, parent=None, final=False):
        self.symbol = symbol
        self.parent = parent
        self.children = {}
        self.count = 1
        self.final = final

    def add(self, symbol):
        try:
            self.children[symbol]
        except KeyError:
            self.children[symbol] = TrieNode(symbol, parent=self)

    def __getitem__(self, item):
        return self.children[item]

    def __setitem__(self, key, value):
        self.children[key] = value

    def __contains__(self, item):
        return item in self.children

    def __iter__(self):
        yield from self.children

    def __eq__(self, other):
        return self.symbol == other.symbol


class Trie:
    def __init__(self):
        self.root: TrieNode = TrieNode('^')
        self.root.count = 0

    def insert(self, sentence):
        index, node = self.__maximum_common_prefix(sentence)
        for symbol in sentence[index:]:
            node.add(symbol)
            node = node[symbol]
        node.final = True
        self.root.count += 1

    def extend(self, *sentences):
        for s in sentences:
            self.insert(s)

    def __maximum_common_prefix(self, sentence):
        current: TrieNode = self.root
        for i, symbol in enumerate(sentence):
            try:
                current = current[symbol]
                current.count += 1
            except KeyError:
                return i, current
        return len(sentence), current

    def __from_prefix(self, prefix):
        node: TrieNode = self.root
        for symbol in prefix:
            try:
                node = node[symbol]
            except KeyError:
                return []

        yield from Trie.__search_from_node(node, prefix)

    @staticmethod
    def __search_from_node(node, sentence):
        if node.final:
            yield sentence

        for child in node:
            yield from Trie.__search_from_node(node[child], sentence + child)

    def __len__(self):
        return self.root.count

    def __iter__(self):
        yield from self.__search_from_node(self.root, "")

    def __call__(self, prefix):
        yield from self.__from_prefix(prefix)

    def __contains__(self, item):
        i, node = self.__maximum_common_prefix(item)
        return i == len(item) and node.final
