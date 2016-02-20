import copy


class Vector:
    def __init__(self, a):
        self.a = a

    def __getitem__(self, item):
        return self.a[item]

    def __len__(self):
        return len(self.a)

    def multiply_vector(self, vector):
        return sum(self[i] * vector[i] for i in range(len(self)))

    def multiply_matrix(self, matrix):
        return Vector([sum(self[j] * matrix[j][i] for j in range(len(self))) for i in range(len(matrix[0]))])

    def __mul__(self, other):
        return Vector([x * other for x in self])

    def __str__(self):
        return '(' + " ".join("%.6lf" % x for x in self) + ')\n'
