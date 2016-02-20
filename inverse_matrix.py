import numpy as np

eps = 1e-8


def get_d(k, z):
    n = len(z)
    ans = np.array([[1. if i == j else .0 for i in range(n)] for j in range(n)])
    ans[:, k] = np.array([1 / z[k] if i == k else -z[i] / z[k] for i in range(n)])
    return ans


def replace_columns(c, j, v, b):
        n = len(c)
        e_iter = np.array([1. if i == j else .0 for i in range(n)])
        alpha = e_iter.dot(b).dot(v)
        if abs(alpha) > eps:
            c[:, j] = v
            b = get_d(j, b.dot(v)).dot(b)
            return c, b
        return None


def get_inverse_matrix(a, debug=False):
    n = len(a)
    e = np.array([[1. if i == j else .0 for i in range(n)] for j in range(n)])
    c, b = e.copy(), e.copy()
    j = [i for i in range(n)]
    s = [0] * n
    for it in range(n):
        if debug:
            print('Итерация номер ' + str(it))
        for x in j:
            replace = replace_columns(c, it, a[:, x].copy(), b)
            if replace is not None:
                c, b = replace
                j.remove(x)
                s[x] = it
                if debug:
                    print('На ' + str(it) + '-ое место подходит вектор под номером ' + str(x))
                    print('Текущая матрица - \n' + str(c))
                    print('Текущая обратная матрица - \n' + str(b))
                break
        else:
            return None
    return np.array([b[s[i]] for i in range(n)])


def main():
    a = []
    with open('input.txt') as file:
        n, = map(int, file.readline().split())
        for i in range(n):
            a.append(list(map(float, file.readline().split())))
    a = np.array(a)
    ans = get_inverse_matrix(a, debug=True)
    if ans is not None:
        print()
        print(ans)
        print('Перемножение исходной и полученной обратной - \n' + str(a.dot(ans)))
    else:
        print('Исходная матрица является вырожденной')


if __name__ == "__main__":
    main()
