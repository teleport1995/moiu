import inverse_matrix as inv
import numpy as np

eps = 1e-9
INF = 999
EMPTY_SET = "Empty"
INFINITE_SET = "Infinity"


def second_phase(a, b, c, jb, ab_inv, debug=False):
    if debug:
        print('Вторая фаза')
    n = len(c)
    ab = np.array([a[:, j].copy() for j in jb]).transpose()
    xb = ab_inv.dot(b)
    iteration = 0
    while True:
        x = np.array([0.] * n)
        for xx, j in zip(xb, jb):
            x[j] = xx

        if debug:
            iteration += 1
            print('Итерация ' + str(iteration))
            print('Текущий план x = ' + str(x))
            print('Базис плана = ' + str(np.array([j + 1 for j in jb])))
            print('Текущее значение целевой функции - ' + str(c.dot(x)))
            print()

        cb = np.array([c[x] for x in jb])
        u = cb.dot(ab_inv)
        delta = np.array([u.dot(av) - cc for av, cc in zip(a.transpose(), c)])
        if delta.min() > -eps:
            return x, jb
        for j0 in [i for i in range(n) if i not in jb]:
            if delta[j0] < -eps:
                z = ab_inv.dot(a[:, j0])
                if z.max() < eps:
                    return INFINITE_SET
                theta_0 = min(xx / zz for xx, zz in zip(xb, z) if zz > eps)
                j_s = min(j for j, zz in zip(jb, z) if zz > eps > abs(theta_0 - x[j] / zz))
                s = jb.index(j_s)
                jb[s] = j0
                xb -= theta_0 * z
                xb[s] = theta_0
                ab, ab_inv = inv.replace_columns(ab, s, a[:, j0].copy(), ab_inv)
                break


def first_phase(a, b, c, debug=False):
    if debug:
        print('Первая фаза')
    m, n = len(a), len(c)
    for i in range(m):
        if b[i] < -eps:
            a[i] *= -1
            b[i] *= -1
    e = np.array([[1. if i == j else 0. for i in range(m)] for j in range(m)])
    na = np.hstack((a.copy(), e))
    nc = np.array([0. if i < n else -1. for i in range(n + m)])
    jb = list(range(n, n + m))

    if debug:
        print('Решаем вспомогательную задачу ЛП')
        print('a = \n' + str(na))
        print('b = ' + str(b))
        print('c = ' + str(c))

    x, jb = second_phase(na, b, nc, jb, e.copy())
    if x[n:].max() > eps:
        return EMPTY_SET
    while not (max(jb) < n):
        m = len(jb)
        nab = np.array([na[:, j] for j in jb]).transpose()
        nab_inv = inv.get_inverse_matrix(nab)
        for k in range(m):
            if jb[k] >= n:
                e_k = np.array([1. if i == k else 0. for i in range(m)])
                for i in range(n):
                    if i not in jb:
                        if abs(e_k.dot(nab_inv).dot(a[:, i])) > eps:
                            jb[k] = i
                            break
                else:
                    if debug:
                        print('Условие ' + str(jb[k] - n + 1) + ' линейно зависит от других')

                    jb_k = jb[k]
                    b = np.array([b[i] for i in range(m) if i != jb_k - n])
                    jb.remove(jb_k)
                    for i in range(len(jb)):
                        if jb[i] > jb_k:
                            jb[i] -= 1
                    na = np.array([na[i] for i in range(m) if i != jb_k - n])
                    na = np.array([na[:, i] for i in range(n + m) if i != jb_k]).transpose()
                break
    return na[:, :n], b, jb, inv.get_inverse_matrix(np.array([na[:, i] for i in jb]).transpose())


def all_phases(a, b, c, debug=False):
    first = first_phase(a, b, c, debug)
    if first == EMPTY_SET:
        return first
    a, b, jb, ab_inv = first
    return second_phase(a, b, c, jb, ab_inv, debug)


def main():
    a = []
    with open('simplex-method.txt') as file:
        m, n = map(int, file.readline().split())
        for i in range(m):
            a.append(list(map(float, file.readline().split())))
        file.readline()
        b = list(map(float, file.readline().split()))
        file.readline()
        c = list(map(float, file.readline().split()))
        file.readline()
        jb = list(map(float, file.readline().split()))
    a, b, c, jb = np.array(a), np.array(b), np.array(c), np.array(jb)
    ans = all_phases(a, b, c, debug=True)
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
