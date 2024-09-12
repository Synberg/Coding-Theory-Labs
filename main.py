import random
from itertools import combinations
import numpy as np


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


def get_leading_columns(matrix: np.ndarray) -> np.ndarray:
    leading_cols = []
    col = 0
    m, n = matrix.shape

    for i in range(m):
        while col < n and matrix[i, col] == 0:
            col += 1
        if col == n:
            return np.array([])
        leading_cols.append(col)

    return np.array(leading_cols)


def get_shortened_matrix(matrix: np.ndarray, cols_to_delete: np.ndarray) -> np.ndarray:
    cols_to_delete_set = set(cols_to_delete)
    transposed_matrix = matrix.T
    transposed_shortened_matrix = [row for idx, row in enumerate(transposed_matrix) if idx not in cols_to_delete_set]
    return np.array(transposed_shortened_matrix).T


def get_h_matrix(matrix: np.ndarray, leading: np.ndarray, m: int) -> np.ndarray:
    n = matrix.shape[1]
    result = np.zeros((m, n), dtype=int)
    identity_matrix = np.eye(n, dtype=int)
    short_index, unit_index, leading_index = 0, 0, 0
    cols_amount = len(leading)

    for i in range(m):
        if leading_index < cols_amount and i == leading[leading_index]:
            result[i] = matrix[short_index]
            short_index += 1
            leading_index += 1
        else:
            result[i] = identity_matrix[unit_index]
            unit_index += 1

    return result


def generate_linear_combinations(matrix: np.ndarray) -> np.ndarray:
    m = matrix.shape[0]
    all_combinations = set()
    # Генерация всех комбинаций кодовых слов
    for r in range(2, m + 1):
        for comb in combinations(range(m), r):
            # Создание линейной комбинации для текущей комбинации строк
            combination = np.sum(matrix[list(comb)], axis=0) % 2
            all_combinations.add(tuple(combination))
    result_array = np.array(list(all_combinations), dtype=int)
    return result_array


def get_all_code_words_iterative(num: int) -> np.ndarray:
    combinations = []
    for i in range(2 ** num):
        combination = []
        for j in range(num):
            combination.append((i >> j) & 1)
        combinations.append(combination)
    return np.array(combinations)


def multiply_words_with_G(G: np.ndarray, all_words: np.ndarray) -> np.ndarray:
    code_words = np.dot(all_words, G) % 2
    return code_words


def Hamming_weight(word: np.ndarray) -> int:
    return sum(word)


def Hamming_distance(words: np.ndarray) -> int:
    result = 2147000000
    for word in words:
        result = min(result, Hamming_weight(word))
    return result


if __name__ == "__main__":
    default_matrix = np.array([
        [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    ])

    step_matrix = REF(default_matrix)
    print("Ступенчатая матрица G", step_matrix, sep='\n')

    reduced_step_matrix = RREF(step_matrix)
    k, n = len(reduced_step_matrix), len(reduced_step_matrix[0])
    print("Приведенная ступенчатая матрица G*", reduced_step_matrix, sep='\n')

    leading_cols = get_leading_columns(reduced_step_matrix)
    print("Индексы ведущих столбцов", leading_cols, sep='\n')

    shortened_matrix = get_shortened_matrix(reduced_step_matrix, leading_cols)
    print("Сокращенная матрица X", shortened_matrix, sep='\n')

    h_matrix = get_h_matrix(shortened_matrix, leading_cols, n)
    print("Проверочная матрица H", h_matrix, sep='\n')

    words_length_n_1 = generate_linear_combinations(default_matrix)
    print("Все кодовые слова длины n (на основе порождающего множества):", words_length_n_1, sep='\n')
    words_length_k = get_all_code_words_iterative(k)
    words_length_n_2 = multiply_words_with_G(step_matrix, words_length_k)
    print("Все кодовые слова длины n (на основе всех двоичных слов длины k):", words_length_n_2, sep='\n')

    u = words_length_k[random.randint(0, len(words_length_k)) - 1]
    print("Случайное кодовое слово u:", u, sep='\n')
    v = u @ step_matrix % 2
    print("Кодовое слово v = u @ G", v, sep='\n')
    print("Умножение кодового слова на проверочную матрицу H:", v @ h_matrix % 2, sep='\n')

    distance = Hamming_distance(step_matrix)
    print("Длина Хэмминга:", distance, sep='\n')
    print("Кратность ошибки t:", distance - 1, sep='\n')

    v = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1])
    v += np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    print("Ошибка обнаружена:", v @ h_matrix % 2, sep='\n')
    v -= np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    v += np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0])
    print("Ошибка не обнаружена:", v @ h_matrix % 2, sep='\n')
