from simplex_method import first_phase, EMPTY_SET, INFINITE_SET, eps
from inverse_matrix import get_inverse_matrix
import numpy as np
import copy

INF = 1e100


def get_h_ast(j_ast, d, a):
    k, m = len(j_ast), len(a)
    h_ast = np.array([[0.] * (k + m) for _ in range(k + m)])
    for i in range(k):
        for j in range(k):
            h_ast[i][j] = d[j_ast[i]][j_ast[j]]

    for i in range(m):
        for j in range(k):
            h_ast[k+i][j] = a[i][j_ast[j]]

    for i in range(m):
        for j in range(k):
            h_ast[j][k+i] = a[i][j_ast[j]]

    return h_ast


def second_phase(a, a_op_inv, b, c, d, j_op, j_ast, debug=False):

    m, n = len(b), len(c)

    xx = a_op_inv.dot(b)
    x = np.array([0.] * n)
    for i in range(m):
        x[j_op[i]] = xx[i]

    cnt = 0
    while True:

        if debug:
            print("\n\nИтерация номер ", cnt)
            cnt += 1
            print("Текущий x = ", str(x))
            print("Текущая целевая функция равна", c.dot(x) + x.dot(d).dot(x) / 2)
            print("Текущий опорный план = ", str(j_op))
            print("Текущий правильный опорный план = ", str(j_ast))

        cx = c + d.dot(x)
        u = -np.array([cx[i] for i in j_op]).dot(a_op_inv)
        delta = np.array([u.dot(a[:, j]) + cx[j] for j in range(n)])

        if debug:
            print("Текущий cx = ", str(cx))
            print("Текущий вектор потенциалов = ", str(u))
            print("Текущие оценки = ", str(delta))

        if delta.min() > -eps:
            return x

        j0 = max([x for x in range(n) if delta[x] < -eps])

        if debug:
            print("Текущий j0 = ", str(j0))

        l = np.array([0.] * n)
        l[j0] = 1

        h_ast = get_h_ast(j_ast, d, a)
        h_j0 = np.array([d[j_ast[x]][j0] for x in range(len(j_ast))] + [x for x in a[:, j0]])
        l_ast = -get_inverse_matrix(h_ast).dot(h_j0)[:len(j_ast)]

        if debug:
            print("Текущая матрица H = \n", str(h_ast))
            print("Текущая обратная матрица H^(-1) = \n", str(get_inverse_matrix(h_ast)))
            print("Текущий вектор h_j0 = ", str(h_j0))
            print("Текущий вектор l_ast = ", str(l_ast))

        for i in range(len(j_ast)):
            l[j_ast[i]] = l_ast[i]

        if debug:
            print("Текущий l = ", str(l))

        theta_j = np.array([INF if l[j] > -eps else -x[j] / l[j] for j in j_ast])
        delta_d = l.dot(d).dot(l)
        theta_j0 = INF if abs(delta_d) < eps else abs(delta[j0]) / delta_d

        if debug:
            print("Текущий theta_j = ", str(theta_j))
            print("Текущий delta_d = ", str(delta_d))
            print("Текущий theta_j0 = ", str(theta_j0))

        theta_0 = min([theta_j.min(), theta_j0])
        if theta_0 == INF:
            return INFINITE_SET

        s = -1
        for i in range(n):
            if i == j0:
                if abs(theta_0 - theta_j0) < eps:
                    s = j0
                    break
            elif i in j_ast:
                if abs(theta_0 - (-x[i] / l[i])) < eps:
                    s = i
                    break

        x += theta_0 * l

        if debug:
            print("Текущий s = ", str(s))

        if s == j0:
            j_ast.append(j0)
        elif s in j_ast and s not in j_op:
            j_ast.remove(s)
        else:
            for i in range(n):
                if i not in j_op and i in j_ast and abs(a_op_inv.dot(a[:, i])[j_op.index(s)]) > eps:
                    j_op.remove(s)
                    j_op.append(i)
                    j_ast.remove(s)
                    break
            else:
                j_op.remove(s)
                j_op.append(j0)
                j_ast.remove(s)
                j_ast.append(j0)
        a_op_inv = get_inverse_matrix(np.array([a[:, x] for x in j_op]).transpose())


def all_phases(a, b, c, d, debug=False):
    first = first_phase(a, b, c, debug)
    if first == EMPTY_SET:
        return first
    a, b, jb, ab_inv = first
    return second_phase(a, ab_inv, b, c, d, jb, copy.deepcopy(jb), debug)


def main():
    a = []
    d = []
    b1 = []
    with open('square.txt') as file:
        m, n = map(int, file.readline().split())
        for i in range(m):
            a.append(list(map(float, file.readline().split())))
        file.readline()
        b = list(map(float, file.readline().split()))
        file.readline()
        c = list(map(float, file.readline().split()))
        file.readline()
        d1 = list(map(float, file.readline().split()))
        file.readline()
        for i in range(2):
            b1.append(list(map(float, file.readline().split())))

    """ d = np.array([[0.] * n for _ in range(n)])
    for i in range(n):
        d[i][i] = 1.
    d *= 100
"""

    a, b, c, b1, d1 = np.array(a), np.array(b), np.array(c), np.array(b1), np.array(d1)

    c = -d1.transpose().dot(b1)
    d = b1.transpose().dot(b1)


    #a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)

    ans = all_phases(a, b, c, d, debug=True)
    if ans == EMPTY_SET:
        print('Множество допустимых планов пусто')
    elif ans == INFINITE_SET:
        print('Целевая функция не ограничена на множестве допустимых планов')
    else:
        x = ans
        print('xo = ' + str(x))
        print('Целевая функция при x0 равна ', c.dot(x) + x.transpose().dot(d).dot(x) / 2)
        print('Разность - ', str(a.dot(x) - b))

if __name__ == "__main__":
    main()
