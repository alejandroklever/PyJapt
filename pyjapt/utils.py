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
