import numpy as np
import copy

eps = 1e-9


def get_cycle(m, n, s):
    s = copy.deepcopy(s)
    rows, cols = [0] * m, [0] * n
    for x, y in s:
        rows[x] += 1
        cols[y] += 1

    while True:
        if sum(rows) == 0:
            return set()

        iteration_count = 0

        for i in range(m):
            if rows[i] == 1:
                for x, y in s:
                    if i == x:
                        rows[x] -= 1
                        cols[y] -= 1
                        iteration_count += 1
                        s.remove((x, y))
                        break

        for j in range(n):
            if cols[j] == 1:
                for x, y in s:
                    if j == y:
                        rows[x] -= 1
                        cols[y] -= 1
                        iteration_count += 1
                        s.remove((x, y))
                        break

        if iteration_count == 0:
            return s


def get_initial_plan(a, b):
    a, b = copy.deepcopy(a), copy.deepcopy(b)
    m, n = len(a), len(b)
    plan = np.array([[0] * n for _ in range(m)])
    a_p, b_p = 0, 0
    basis = set()
    while a_p < m and b_p < n:
        basis.add((a_p, b_p))
        if a[a_p] < b[b_p]:
            b[b_p] -= a[a_p]
            plan[a_p][b_p] = a[a_p]
            a_p += 1
        else:
            a[a_p] -= b[b_p]
            plan[a_p][b_p] = b[b_p]
            b_p += 1
    for x in range(m):
        for y in range(n):
            if (x, y) not in basis and len(basis) < m + n - 1:
                basis.add((x, y))
                if len(get_cycle(m, n, basis)) > 0:
                    basis.remove((x, y))
    return plan, basis


def get_potentials(m, n, c, jb):
    u, v = [None] * m, [None] * n
    u[0] = 0
    for _ in range(m + n - 1):
        for x, y in jb:
            if u[x] is not None and v[y] is None:
                v[y] = c[x][y] - u[x]
                break
            elif u[x] is None and v[y] is not None:
                u[x] = c[x][y] - v[y]
                break
    return u, v


def split_cycle(cur_i, cur_j, cycle):
    hor, ver = set(), set()
    for _ in range(len(cycle) // 2):
        for i, j in cycle:
            if i == cur_i and j != cur_j:
                n_i, n_j = i, j
        hor.add((n_i, n_j))
        ver.add((cur_i, cur_j))
        for i, j in cycle:
            if j == n_j and i != n_i:
                cur_i, cur_j = i, j
    return hor, ver

cnt0 = 0


def solve_transport_task(a, b, c, debug=False):
    if abs(a.sum() - b.sum()) > eps or a.min() < -eps or b.min() < -eps:
        return None

    plan, basis = get_initial_plan(a, b)
    m, n = len(a), len(b)

    cnt = 0
    while True:
        cnt += 1
        if debug:
            print("Итерация номер " + str(cnt))
            print("Текущий план X = \n" + str(plan))
        u, v = get_potentials(m, n, c, basis)
        delta = np.array([[u[i] + v[j] - c[i][j] for j in range(n)] for i in range(m)])
        if delta.max() < eps:
            return plan, basis

        i0, j0 = min([(i, j) for i in range(m) for j in range(n) if delta[i][j] > eps and (i, j) not in basis])
        basis.add((i0, j0))
        cycle = get_cycle(m, n, basis)
        hor, ver = split_cycle(i0, j0, cycle)
        theta0 = min([plan[i][j] for i, j in hor])
        ia, ja = min([(i, j) for i, j in hor if abs(theta0 - plan[i][j]) < eps])
        for i, j in hor:
            plan[i][j] -= theta0
        for i, j in ver:
            plan[i][j] += theta0
        basis.remove((ia, ja))

        if debug:
            print("Целевая функция равна " + str(sum([c[i][j] * plan[i][j] for i in range(m) for j in range(n)])))


        if debug:
            if theta0 == 0:
                global cnt0
                cnt0 += 1
            print("basis - " + str(basis))
            print("potentials u - " + str(u) + " v - " + str(v))
            print("i0 = " + str(i0) + " j0 = " + str(j0))
            print("cycle - " + str(cycle))
            print("hor - " + str(hor) + " ver - " + str(ver))



def main():
    with open("transport.txt") as file:
        m, n = map(int, file.readline().split())
        file.readline()
        a = list(map(int, file.readline().split()))
        b = list(map(int, file.readline().split()))
        file.readline()
        c = [list(map(int, file.readline().split())) for _ in range(m)]
    a, b, c = np.array(a), np.array(b), np.array(c)
    ans = solve_transport_task(a, b, c, debug=True)
    if ans is None:
        print("Условия несовместны")
    else:
        x, jb = ans
        print("Искомая матрица X = ")
        print(x)
        print("Искомый базис = " + str(jb))
        print("Целевая функция равна " + str(sum([c[i][j] * x[i][j] for i in range(m) for j in range(n)])))
        print(cnt0)

if __name__ == "__main__":
    main()