"""
Формирование изображения дифракционной линзой
"""
import math
from timeit import timeit
from collections import namedtuple

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from numba import njit
from scipy import integrate

# Количество лучей
AMOUNT_RAYS = 1_000

# количество точек на регистраторе
AMOUNT_POINTS = 100

# Радиус пучка
BEAM_RADIUS = 3

# Определение области входного пучка
U0 = V0 = -BEAM_RADIUS
UN = VN = BEAM_RADIUS

# Радиус фокального кольца
FOCAL_RING_RADIUS = 0.5 * BEAM_RADIUS

# Определение выходной области на фокусировке
X0 = Y0 = -1.5 * FOCAL_RING_RADIUS
XN = YN = 1.5 * FOCAL_RING_RADIUS

# Расстояние от объекта до линзы
DISTANCE_OBJECT_TO_LENS = 1

# Расстояние от линзы до матрицы
DISTANCE_LENS_TO_MATRIX = 1

# F - фокусное расстояние линзы
F = 1

# TODO: разобраться за что отвечает данный параметр
M = 10 ** 10

# лямбда нулевое
L0 = 15

SIGMA = 0.2 / 6

# Массив (r, g, b) пикселей для соответствующей длины волны
IMAGE_ARR = np.array(Image.open('../res/img/Sample.png'))

# rgb: r-640 нм, g-530 нм, b-465 нм.
RGB_LAMBDAS = namedtuple('RGB', 'r g b')(*(lam * 10 ** -9 for lam in (640, 530, 465)))

# Базовая длина волны
lamda = RGB_LAMBDAS.r


# print(image_arr)

# ====================================================================================================
@njit(cache=True)
def s_x0(u, v, u0, v0) -> float:
    return (u - u0) / ((u - u0) ** 2 + (v - v0) ** 2 + DISTANCE_OBJECT_TO_LENS ** 2)


@njit(cache=True)
def s_y0(u, v, u0, v0) -> float:
    return (v - v0) / ((u - u0) ** 2 + (v - v0) ** 2 + DISTANCE_OBJECT_TO_LENS ** 2)


@njit(cache=True)
def s_z0(u, v, u0, v0) -> float:
    return DISTANCE_OBJECT_TO_LENS / ((u - u0) ** 2 + (v - v0) ** 2 + DISTANCE_OBJECT_TO_LENS ** 2)


# ====================================================================================================
@njit(cache=True)
def d_psi_du(u, v) -> float:
    return -u / (F ** 2 + u ** 2 + v ** 2) ** 0.5


@njit(cache=True)
def d_psi_dv(u, v) -> float:
    return -v / (F ** 2 + u ** 2 + v ** 2) ** 0.5


# ====================================================================================================
@njit(cache=True)
def s_xm(u, v, u0, v0, l, m) -> float:
    return s_x0(u, v, u0, v0) + (l * m) / (L0 * AMOUNT_RAYS) * d_psi_du(u, v)


@njit(cache=True)
def s_ym(u, v, u0, v0, l, m) -> float:
    return s_y0(u, v, u0, v0) + (l * m) / (L0 * AMOUNT_RAYS) * d_psi_dv(u, v)


@njit(cache=True)
def s_zm(u, v, u0, v0, l, m, min_val=10 ** -4) -> float:
    val = (1 - s_xm(u, v, u0, v0, l, m) - s_ym(u, v, u0, v0, l, m))
    if val >= min_val:
        return val ** 0.5
    return min_val


# ====================================================================================================

@njit(cache=True)
def xy_m(u, v, u0=U0, v0=V0, l=lamda, m=M) -> tuple[float, float]:
    sxm = s_xm(u, v, u0, v0, l, m)
    sym = s_ym(u, v, u0, v0, l, m)
    szm = s_zm(u, v, u0, v0, l, m)
    return u + (sxm / szm) * DISTANCE_LENS_TO_MATRIX, \
           v + (sym / szm) * DISTANCE_LENS_TO_MATRIX


# ====================================================================================================

@njit
def B(u0, v0, l) -> float:
    """Вернёт яркость пикселя на объекте для переданной длины волны"""
    return IMAGE_ARR[u0, v0, RGB_LAMBDAS.index(l)]


@njit(cache=True)
def R(u, v, u0, v0) -> float:
    """Функция, зависящая от диаграммы рассеяния точки в предметной области"""
    return DISTANCE_OBJECT_TO_LENS / (math.sqrt((u - u0) ** 2 + (v - v0) ** 2 + DISTANCE_OBJECT_TO_LENS ** 2))


@njit(cache=True)
def i0_func(u, v, u0=U0, v0=V0, l=lamda) -> float:
    return (B(u0, v0, l) * R(u, v, u0, v0)) / \
           ((u - u0) ** 2 + (v - v0) ** 2 + DISTANCE_OBJECT_TO_LENS ** 2)


# ====================================================================================================

@njit(cache=True)
def delta(x, y) -> float:
    """Во всех формулах дельта функция Дирака заменяется ее аппроксимацией."""
    return math.exp(-((x ** 2 + y ** 2) / (2 * SIGMA ** 2))) / (2 * math.pi * SIGMA ** 2)


# ====================================================================================================

def f(u, v, u0, v0, x, y, l, m):
    xm, ym = xy_m(u, v, u0, v0, l, m)
    return i0_func(u, v, u0, v0, l) * delta(x - xm, y - ym)


def I(u0, v0, x, y, l, m):
    """Интенсивность на изображении"""
    return lambda a, b, g, h: integrate.dblquad(f, a, b, g, h, args=(u0, v0, x, y, l, m))


@njit
def ray_trace():
    # Вычисление шага на апертуре
    del_u = (UN - U0) / AMOUNT_RAYS
    del_v = (VN - V0) / AMOUNT_RAYS
    del_uv = del_u * del_v

    # Вычисление шага в фокальной области
    del_x = (XN - X0) / AMOUNT_POINTS
    del_y = (YN - Y0) / AMOUNT_POINTS

    # Определить номера точек в которые вносится вклад
    di = math.floor(3 * SIGMA / del_x)
    dj = math.floor(3 * SIGMA / del_y)

    # инициализация массива
    result_array = np.zeros((AMOUNT_POINTS, AMOUNT_POINTS), np.float64)

    # Начало трассировки лучей
    for i in range(1, AMOUNT_RAYS):
        ui = U0 + del_u * i
        for j in range(1, AMOUNT_RAYS):
            vj = V0 + del_v * j
            xij, yij = xy_m(ui, vj)
            # Определение ближайшей точки на регистраторе от точки прихода луча
            i_temp = math.floor((xij - X0) / del_x)
            j_temp = math.floor((yij - Y0) / del_y)

            i_left = i_temp - di - 1
            i_right = i_temp + di + 1
            j_down = j_temp - dj - 1
            j_up = j_temp + dj + 1

            if i_left < 1:
                i_left = 1
            if i_right < 1:
                i_right = 1
            if j_down < 1:
                j_down = 1
            if j_up < 1:
                j_up = 1

            if i_left > AMOUNT_POINTS:
                i_left = AMOUNT_POINTS
            if i_right > AMOUNT_POINTS:
                i_right = AMOUNT_POINTS
            if j_down > AMOUNT_POINTS:
                j_down = AMOUNT_POINTS
            if j_up > AMOUNT_POINTS:
                j_up = AMOUNT_POINTS

            # Нахождение вклада от лучей в релевантные точки в фокальной области
            v = i0_func(ui, vj) * del_uv
            for ii in range(i_left, i_right):
                for jj in range(j_down, j_up):
                    result_array[ii][jj] += v * delta(X0 + (ii - 1) * del_x - xij,
                                                      Y0 + (jj - 1) * del_y - yij)

    return result_array


def main(calc: bool = True, save: bool = True, draw: bool = True):
    delimiter = ' '
    filename = 'save.txt'

    array = ray_trace() if calc else np.loadtxt(filename, delimiter=delimiter)

    # Сохранение результата
    if calc and save:
        np.savetxt(filename, array, delimiter=delimiter)

    # Отображение результата
    if draw:
        plt.imshow(array, interpolation="nearest", origin="upper")
        plt.savefig("ring_spiral.png", bbox_inches='tight', dpi=100)
        plt.show()


if __name__ == '__main__':
    # Основной функционал
    main()

    # Тест скорости
    TEST_SPEED = False
    print(f'Всего запусков: {(number := 10)}\n'
          f'Среднее: {(time := timeit(lambda: main(True, False, False), number=number)) / number:.2f} секунд\n'
          f'Тестирование заняло: {time:.2f} секунд' if TEST_SPEED else '')
