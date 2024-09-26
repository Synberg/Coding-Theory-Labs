import itertools

import numpy as np
import random


def REF(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.copy()  # Создаем копию матрицы, чтобы не изменять оригинал
    m, n = matrix.shape
    current_row = 0

    for col in range(n):
        # Поиск строки с ненулевым элементом в текущем столбце
        row_with_leading_one = current_row
        while row_with_leading_one < m and matrix[row_with_leading_one, col] == 0:
            row_with_leading_one += 1

        if row_with_leading_one == m:
            continue  # Нет ненулевых элементов в этом столбце, пропускаем

        # Меняем строки местами, если ведущий элемент не в текущей строке
        if row_with_leading_one != current_row:
            matrix[[row_with_leading_one, current_row]] = matrix[[current_row, row_with_leading_one]]

        # Обнуляем все элементы ниже ведущего
        for row_below in range(current_row + 1, m):
            if matrix[row_below, col] == 1:
                matrix[row_below] ^= matrix[current_row]

        current_row += 1
        if current_row == m:
            break

    # Удаляем строки, состоящие только из нулей
    non_zero_rows = np.any(matrix, axis=1)
    return matrix[non_zero_rows]


def RREF(matrix: np.ndarray) -> np.ndarray:
    matrix_copy = matrix.copy()  # Создаем глубокую копию матрицы, чтобы не изменять оригинал
    m, n = matrix_copy.shape

    # Идем снизу вверх по строкам
    for current_row in range(m - 1, -1, -1):
        leading_col = np.argmax(matrix_copy[current_row] != 0)
        if matrix_copy[current_row, leading_col] == 0:
            continue  # В строке только нули, пропускаем

        # Обнуляем все элементы выше ведущего
        for row_above in range(current_row):
            if matrix_copy[row_above, leading_col] == 1:
                matrix_copy[row_above] ^= matrix_copy[current_row]

    return matrix_copy


def standard_view(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.copy()  # Копируем матрицу, чтобы не изменять оригинал
    m, n = matrix.shape
    identity_columns = []

    # Находим позиции ведущих единичных элементов
    for i in range(m):
        for j in range(n):
            if matrix[i, j] == 1 and np.all(matrix[:, j] == (np.eye(m)[i])):
                identity_columns.append(j)
                break

    # Проверяем, можно ли создать единичную матрицу из выбранных столбцов
    if len(identity_columns) != m:
        raise ValueError("Невозможно привести матрицу к стандартному виду")

    # Создаем перестановку столбцов: сначала identity_columns, затем остальные
    remaining_columns = [j for j in range(n) if j not in identity_columns]
    permuted_columns = identity_columns + remaining_columns

    # Применяем перестановку к матрице
    return matrix[:, permuted_columns]


def h_matrix(matrix: np.ndarray) -> np.ndarray:
    # Определяем размеры порождающей матрицы
    m, n = matrix.shape
    # Извлекаем дополнительную часть X (столбцы справа от единичной матрицы)
    X = matrix[:, m:n]
    identity_matrix = np.eye(n - m, dtype=int)
    return np.vstack((X, identity_matrix))


def generate_syndrome_table(matrix: np.ndarray, error_weight: int) -> dict:
    n = matrix.shape[0]
    syndrome_table = {}
    for error in range(1, error_weight + 1):
        for error_indices in itertools.combinations(range(n), error):
            error_vector = np.zeros(n, dtype=int)
            for index in error_indices:
                error_vector[index] = 1
            syndrome = error_vector @ matrix % 2
            syndrome_table[tuple(map(int, syndrome))] = tuple(error_indices)

    return syndrome_table


if __name__ == "__main__":
    print("Часть 1")
    s_matrix = np.array([[1, 0, 0, 1, 0, 1, 1],
                         [1, 1, 0, 0, 0, 0, 1],
                         [0, 0, 1, 1, 0, 0, 1],
                         [1, 0, 1, 0, 1, 0, 1],
                         [0, 0, 1, 1, 1, 1, 0]])
    G = RREF(REF(s_matrix))
    print("Порождающая матрица G:", G, sep="\n")
    G_standard = standard_view(G)
    print("Порождающая матрица G в стандартном виде:", G_standard, sep="\n")
    print()

    H = h_matrix(G_standard)
    print("Проверочная матрица H:", H, sep="\n")
    print()

    syndrome_table = generate_syndrome_table(H, 1)
    print("Таблица синдромов:", syndrome_table, sep="\n")
    print()

    u = np.array([1, 0, 0, 1])
    print("Кодовое слово длины k = 4:", u, sep="\n")
    v = u @ G_standard % 2
    print("Отправленное кодовое слово длины n = 7:", v, sep="\n")
    error = np.array([0] * 7)
    error[random.randint(0, 6)] = 1
    print("Возникшая ошибка:", error, sep="\n")
    v = (v + error) % 2
    print("Принятое с ошибкой слово:", v, sep="\n")
    syndrome = v @ H % 2
    print("Синдром принятого сообщения:", syndrome, sep="\n")
    error = np.array([0] * 7)
    error[syndrome_table[tuple(syndrome)][0]] = 1
    v = (v + error) % 2
    print("Исправленное сообщение:", v, sep="\n")
    print("Отправленное и исправленное сообщение совпадают")
    print()

    print("Кодовое слово длины k = 4:", u, sep="\n")
    print("Отправленное кодовое слово длины n = 7:", v, sep="\n")
    error = np.zeros(7, dtype=int)
    a, b = random.sample(range(7), 2)
    error[a], error[b] = 1, 1
    print("Возникшая ошибка:", error, sep="\n")
    v = (v + error) % 2
    print("Принятое с ошибкой слово:", v, sep="\n")
    syndrome = v @ H % 2
    print("Синдром принятого сообщения:", syndrome, sep="\n")
    error = np.array([0] * 7)
    error[syndrome_table[tuple(syndrome)][0]] = 1
    v = (v + error) % 2
    print("Исправленное сообщение:", v, sep="\n")
    print("Отправленное и исправленное сообщение не совпадают")
    print()


    print("2 часть")
    G_standard = np.array([[1,0,0,0,1,1,1,1,0,0,0,0],
                            [0,1,0,0,0,1,1,1,1,1,0,0],
                            [0,0,1,0,1,0,0,1,1,1,1,0],
                            [0,0,0,1,0,0,1,1,0,0,1,1]])
    print("Порождающая матрица G в стандартном виде:", G_standard, sep="\n")
    print()

    H = h_matrix(G_standard)
    print("Проверочная матрица H:", H, sep="\n")
    print()

    syndrome_table = generate_syndrome_table(H, 2)
    print("Таблица синдромов:", syndrome_table, sep="\n")
    print()

    u = np.array([0, 0, 1, 0])
    print("Кодовое слово длины k = 4:", u, sep="\n")
    v = u @ G_standard % 2
    print("Отправленное кодовое слово длины n = 12:", v, sep="\n")
    error = np.array([0] * 12)
    error[random.randint(0, 11)] = 1
    print("Возникшая ошибка:", error, sep="\n")
    v = (v + error) % 2
    print("Принятое с ошибкой слово:", v, sep="\n")
    syndrome = v @ H % 2
    print("Синдром принятого сообщения:", syndrome, sep="\n")
    error = np.array([0] * 12)
    for index in syndrome_table[tuple(syndrome)]:
        error[index] = 1
    v = (v + error) % 2
    print("Исправленное сообщение:", v, sep="\n")
    print("Отправленное и исправленное сообщение совпадают")
    print()

    print("Кодовое слово длины k = 4:", u, sep="\n")
    print("Отправленное кодовое слово длины n = 12:", v, sep="\n")
    error = np.array([0] * 12)
    a, b = random.sample(range(12), 2)
    error[a], error[b] = 1, 1
    print("Возникшая ошибка:", error, sep="\n")
    v = (v + error) % 2
    print("Принятое с ошибкой слово:", v, sep="\n")
    syndrome = v @ H % 2
    print("Синдром принятого сообщения:", syndrome, sep="\n")
    error = np.array([0] * 12)
    for index in syndrome_table[tuple(syndrome)]:
        error[index] = 1
    v = (v + error) % 2
    print("Исправленное сообщение:", v, sep="\n")
    print("Отправленное и исправленное сообщение совпадают")
    print()

    print("Кодовое слово длины k = 4:", u, sep="\n")
    print("Отправленное кодовое слово длины n = 12:", v, sep="\n")
    error = np.array([0] * 12)
    a, b, c = random.sample(range(12), 3)
    error[a], error[b], error[c] = 1, 1, 1
    print("Возникшая ошибка:", error, sep="\n")
    v = (v + error) % 2
    print("Принятое с ошибкой слово:", v, sep="\n")
    syndrome = v @ H % 2
    print("Синдром принятого сообщения:", syndrome, sep="\n")
    error = np.array([0] * 12)
    if tuple(syndrome) in syndrome_table:
        for index in syndrome_table[tuple(syndrome)]:
            error[index] = 1
        v = (v + error) % 2
        print("Исправленное сообщение:", v, sep="\n")
        print("Отправленное и исправленное сообщение не совпадают")
        print()
    else:
        print("Синдрома, соответствующего данной ошибке, не найдено в таблице синдромов. Сообщение исправить невозможно.")
