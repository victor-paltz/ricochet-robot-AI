from collections import defaultdict


class UnionFind:
    def __init__(self, data):
        self.parent = {elem: elem for elem in data}

    def find(self, elem):
        if self.parent[elem] == elem:
            return elem

        self.parent[elem] = self.find(self.parent[elem])
        return self.parent[elem]

    def find_alt(self, elem):
        # non-recursive find
        # path-splitting: make each node point to its grandparent
        while self.parent[elem] != elem:
            elem, self.parent[elem] = (
                self.parent[elem],
                self.parent[self.parent[elem]],
            )
        return elem

    def union(self, elem1, elem2):
        self.parent1 = self.find(elem1)
        self.parent2 = self.find(elem2)
        if self.parent1 == self.parent2:
            return False
        self.parent[self.parent1] = self.parent2
        return True

    def groups(self):
        groups = defaultdict(list)
        for i in self.parent:
            groups[self.find(i)].append(i)
        return list(groups.values())
