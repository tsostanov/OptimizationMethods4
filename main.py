from dataclasses import dataclass
from typing import Callable

import sympy as sp


@dataclass
class Point:
    x: float
    y: float


# Определяем функции, которые будут использоваться для методов
def f(x, y):
    return (x ** 2) - 2 * (y ** 2) - 2 * (x * y) + x


# частная производная по x
def fdx(x, y):
    return 2 * x - 2 * y + 1


# частная производная по y
def fdy(x, y):
    return -4 * y - 2 * x


def gradient_descent(
        f: Callable[[float, float], float],
        fdx: Callable[[float, float], float],
        fdy: Callable[[float, float], float],
        start: Point,
        step: float = -0.25,
        eps: float = 0.01,
        max_iterations: int = 3,
) -> Point:
    curr = start
    curr_f = f(curr.x, curr.y)
    print(f'Итерация 0: X = ({float(curr.x)}, {float(curr.y)}), f(X) = {float(curr_f)}')
    iterations = 1
    while True:
        new = Point(
            curr.x + step * fdx(curr.x, curr.y),
            curr.y + step * fdy(curr.x, curr.y)
        )
        new_f = f(new.x, new.y)
        print(f'Итерация {iterations}: X = ({float(new.x)}, {float(new.y)}), f(X) = {float(new_f)}')
        if abs(new_f - curr_f) < eps or iterations >= max_iterations:
            return new
        curr = new
        curr_f = new_f
        iterations += 1

def fastest_descent(
        f: Callable[[float, float], float],
        fdx: Callable[[float, float], float],
        fdy: Callable[[float, float], float],
        fast: Point,
        eps: float = 0.01,
        max_iterations: int = 3,
) -> Point:
    curr = fast
    curr_f = f(curr.x, curr.y)
    print(f'Итерация 0: X = ({float(curr.x)}, {float(curr.y)}), f(X) = {float(curr_f)}')
    iterations = 1
    while True:
        h = sp.symbols('h')
        new_x_with_h = curr.x - h * fdx(curr.x, curr.y)
        new_y_with_h = curr.y - h * fdy(curr.x, curr.y)
        print("x =", new_x_with_h)
        print("y =", new_y_with_h)
        new_f_with_h = (new_x_with_h ** 2) - 2 * (new_y_with_h ** 2) - 2 * (new_y_with_h * new_x_with_h) + new_x_with_h
        print("f =", new_f_with_h)
        dfdh = sp.diff(new_f_with_h, h)
        h_value = sp.solve(dfdh, h)
        print("h =", h_value)
        new = Point(
            new_x_with_h.subs('h', h_value[0]),
            new_y_with_h.subs('h', h_value[0])
        )
        new_f = f(new.x, new.y)
        print(f'Итерация {iterations}: X = ({float(new.x)}, {float(new.y)}), f(X) = {float(new_f)}')
        if abs(new_f - curr_f) < eps or iterations >= max_iterations:
            return new
        curr = new
        curr_f = new_f
        iterations += 1


if __name__ == '__main__':
    start = Point(0, 0)
    fast = Point(0, 0)
    extr = gradient_descent(f, fdx, fdy, start)
    print(f'Градиентный спуск: X = ({float(extr.x)}, {float(extr.y)}), f(X) = {float(f(extr.x, extr.y))}')
    extrm = fastest_descent(f, fdx, fdy, fast)
    print(f'Наискорейший спуск: X = ({float(extrm.x)}, {float(extrm.y)}), f(X) = {float(f(extrm.x, extrm.y))}')
