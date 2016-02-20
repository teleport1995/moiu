import copy
from vector import Vector


class Matrix:
    def __init__(self, a):
        self.a = a

    def __getitem__(self, item):
        return self.a[item]

    def __len__(self):
        return len(self.a)

    def multiply_matrix(self, matrix):
        return Matrix([[sum(self[i][k] * matrix[k][j] for k in range(len(matrix))) for j in range(len(matrix[0]))]
                       for i in range(len(self))])

    def multiply_vector(self, vector):
        return Vector([sum(self[i][j] * vector[j] for j in range(len(self[0]))) for i in range(len(self))])

    def __mul__(self, other):
        return Matrix([[x * other for x in row] for row in self])

    def __str__(self):
        return '[\n' + "\n".join([" ".join("%.6lf" % x for x in row) for row in self]) + ']\n'
