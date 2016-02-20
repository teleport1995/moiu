import random
import numpy as np
import pickle
from simplex_method import eps, all_phases


m, n = 5, 9
zero_conditions = 3
zero_elements = 4


def generate_input():
    d, c = [], []
    for i in range(m + 1):
        b = np.array([[float(random.randint(-9, 10)) for _ in range(n)] for __ in range(i + 1)])
        d.append(b.transpose().dot(b) * 0)
        c.append(np.array([float(random.randint(-9, 10)) for _ in range(n)]))
    x0 = [0.] * zero_elements + [float(random.randint(1, 10)) for _ in range(n - zero_elements)]
    random.shuffle(x0)
    x0 = np.array(x0)
    alpha = []
    for i in range(m + 1):
        if i == 0:
            alpha.append(0.)
        else:
            beta_i = x0.dot(d[i]).dot(x0) + c[i].dot(x0)
            if i <= zero_conditions:
                alpha.append(-beta_i)
            else:
                alpha.append(-beta_i - 1)
    data = [d, c, x0, alpha]
    with open("data", "wb") as f:
        pickle.dump(data, f)


def solve_problem(data):
    d, c, x0, alpha = data
    for i in range(m+1):
        d[i] *= 0
    cur_value = x0.dot(d[0]).dot(x0) + c[0].dot(x0)
    print("Текущее значение целевой функции - ", cur_value)
    print("Текущий план - ", x0)
    new_n = n * 2 + zero_conditions
    new_c = np.array([-(2 * d[0][i].dot(x0) + c[0][i]) for i in range(n)] + [0.] * (new_n - n))
    new_m = n + zero_conditions
    new_a = np.array([[0.] * new_n for _ in range(new_m)])
    new_b = np.array([0.] * new_m)

    for i in range(n):
        new_a[i][i], new_a[i][n+i], new_b[i] = 1., 1., 1. if x0[i] < eps else 2.

    for i in range(zero_conditions):
        for j in range(n):
            new_a[n+i][j] = 2 * d[i+1][j].dot(x0) + c[i+1][j]
            if x0[j] > eps:
                new_b[n+i] += 2 * d[i+1][j].dot(x0) + c[i+1][j]
        new_a[n+i][2*n+i] = 1.
        new_b[n+i] += 0#-alpha[i+1]

    l, jb = all_phases(new_a, new_b, new_c)
    for i in range(n):
        if x0[i] > eps:
            l[i] -= 1.
    if l.dot(-new_c) > -eps:
        print("Текущий план оптимален")
    else:
        print("Текущий план неоптимален")
        l = l[:n]
        print("l = ", l)
        for i in range(1, m + 1):
            print(l.dot(2 * d[i].dot(x0) + c[i]))
        pr = 1.
        while True:
            x = x0 + pr * l
            """flag = True
            for i in range(1, m + 1):
                x0_value = x0.dot(d[i]).dot(x0) + c[i].dot(x0) + alpha[i]
                x_value = x.dot(d[i]).dot(x) + c[i].dot(x) + alpha[i]
                print(x0_value)
                print(x_value)
                if x_value > eps:
                    break
            if x.dot(d[0]).dot(x) + c[0].dot(x) < cur_value + eps:
                break"""
            if all([x.dot(d[i]).dot(x) + c[i].dot(x) + alpha[i] < eps for i in range(1, m + 1)]):
                if x.dot(d[0]).dot(x) + c[0].dot(x) < cur_value:
                    break
            pr /= 2
        print("Новый план", ["%.3f" % (_,) for _ in x])
        print("Новое значение целевой функции - ", x.dot(d[0]).dot(x) + c[0].dot(x))
        print("t = ", pr)


def main():
    generate_input()
    with open("data", "rb") as f:
        solve_problem(pickle.load(f))


if __name__ == "__main__":
    main()
