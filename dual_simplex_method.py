import inverse_matrix as inv
import numpy as np

eps = 1e-9
INF = 999
EMPTY_SET = "Empty"
INFINITE_SET = "Infinity"


def dual_second_phase(a, b, c, jb, ab_inv, debug=False):
    if debug:
        print('Вторая фаза')
    n = len(c)
    ab = np.array([a[:, j].copy() for j in jb]).transpose()

    cb = np.array([c[x] for x in jb])
    print(ab_inv)
    print(cb)
    u = cb.dot(ab_inv)
    delta = np.array([u.dot(av) - cc for av, cc in zip(a.transpose(), c)])

    iteration = 0
    while True:
        xb = ab_inv.dot(b)

        x = np.array([0.] * n)
        for xx, j in zip(xb, jb):
            x[j] = xx

        if debug:
            iteration += 1
            print('Итерация ' + str(iteration))
            print('Delta ' + str(delta))
            print('Ab_inv' + str(ab_inv))
            print('Текущий план x = ' + str(x))
            print('Базис плана = ' + str(np.array([j + 1 for j in jb])))
            print('Текущее значение целевой функции - ' + str(c.dot(x)))
            print()

        if x.min() >= -eps:
            return x, jb

        for jk in [i for i in range(n) if i in jb]:
            if x[jk] < -eps:
                k = jb.index(jk)
                mu = np.array([ab_inv[k].dot(a[:, j]) for j in range(n)])
                if min(x for j, x in enumerate(mu) if j not in jb) > -eps:
                    return EMPTY_SET

                sigma_0 = min(-delta[j] / mu[j] for j in range(n) if j not in jb and mu[j] < -eps)
                j_0 = min(j for j in range(n) if j not in jb and mu[j] < -eps and abs(sigma_0 + delta[j] / mu[j]) < eps)
                jb[k] = j_0
                for j in [j for j in range(n) if j != jk and j not in jb]:
                    delta[j] += sigma_0 * mu[j]
                delta[jk] = sigma_0
                ab, ab_inv = inv.replace_columns(ab, k, a[:, j_0].copy(), ab_inv)
                break


def main():
    a = []
    with open('dual-simplex-method.txt') as file:
        m, n = map(int, file.readline().split())
        for i in range(m):
            a.append(list(map(float, file.readline().split())))
        file.readline()
        b = list(map(float, file.readline().split()))
        file.readline()
        c = list(map(float, file.readline().split()))
        file.readline()
        jb = list(map(float, file.readline().split()))
    a, b, c = np.array(a), np.array(b), np.array(c)
    ab_inv = inv.get_inverse_matrix(np.array([a[:, k] for k in jb]).transpose())
    if ab_inv is None:
        print('Базисная матрица вырождена')
        return
    y = ab_inv.transpose().dot(np.array([c[x] for x in jb]))
    if not (a.transpose().dot(y) > c - eps).all():
        print('Неподходящий базис')
        return

    ans = dual_second_phase(a, b, c, jb, ab_inv, debug=True)
    if ans == EMPTY_SET:
        print('Множество допустимых планов пусто')
    elif ans == INFINITE_SET:
        print('Целевая функция не ограничена на множестве допустимых планов')
    else:
        x, jb = ans
        print('xo = ' + str(x))
        print('Jb = ' + str(np.array([j + 1 for j in jb])))
        print('Целевая функция при x0 равна ' + str(c.dot(x)))

if __name__ == "__main__":
    main()
