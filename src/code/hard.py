import numpy as np
import matplotlib.pyplot as plt
from easy import logistic_map
from normal import generate_sequence

'Реализует графическое построение для задач секции Hard'


def build_staircase(x_0: float, r: float,
                    n_steps: np.int64) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит точки для лестницы Ламерея при логистическом отображении.

    Args:
        x_0 (float): начальная точка последовательности
        r (float): управляющий параметр отображения
        n_steps (np.int64): количество итераций для достижения x_n
         члена последовательности

    Returns:
        (tuple[np.ndarray, np.ndarray]): пара последовательностей,
         задающих координаты графика
    """
    x_array = np.empty(2 * int(n_steps) + 1, dtype=float)
    y_array = np.empty(2 * int(n_steps) + 1, dtype=float)

    x_n: float = x_0
    x_array[0] = x_n
    y_array[0] = x_n

    for k in range(int(n_steps)):
        x_next: float = float(logistic_map(x_n, r))

        x_array[2 * k + 1] = x_n
        y_array[2 * k + 1] = x_next

        x_array[2 * k + 2] = x_next
        y_array[2 * k + 2] = x_next

        x_n = x_next

    return x_array, y_array


def plot_staircase(x_0: float, r: float, n_steps: np.int64,
                   save_path: str = "src/images/lamere.png") -> None:
    """
    Строит график y=f(x), прямую y=x и лестницу Ламерея для заданного r.

    Args:
        x_0 (float): начальная точка графика
        r (float): управляющий параметр отображения
        n_steps (np.int64): количество итераций для достижения x_n
         точки графика
        save_path (str): путь, куда сохраняется изображение в проекте
    """
    x_grid = np.linspace(0.0, 1.0, 1000)
    y_grid = logistic_map(x_grid, r)

    x_array, y_array = build_staircase(x_0, r, n_steps)

    plt.figure()
    plt.plot(x_grid, y_grid, label=r"$f(x)=rx(1-x)$")
    plt.plot(x_grid, x_grid, label=r"$f(x)=x$")
    plt.plot(x_array, y_array, marker="o", linewidth=1.2,
             label="Лестница Ламерея")

    plt.title(f"Лестница Ламерея ($x_0$={x_0:.3f}, $r$={r:.3f})")
    plt.xlabel("$x_n$")
    plt.ylabel("$x_{n+1}$")
    plt.grid(True)
    plt.legend()

    plt.savefig(save_path)
    plt.show()


def estimate_period(seq: np.ndarray, skip: int = 500, tail_len: int = 1000,
                    max_m: int = 512, eps: float = 1e-6) -> int:
    """
    Оценивает период m по хвосту последовательности,
    проверяя условие близости:
        |x_{t+i} - x_{t+i-m}| < eps  для всех i на хвосте.

    Args:
        seq (np.ndarray): последовательность {x_n}
        skip (int): сколько первых значений отбрасываем
        tail_len (int): сколько значений рассматриваем
        max_m (int): максимальный проверяемый период
        eps (float): допуск для сравнения

    Returns:
        (int): найденный период m; 0 если валидный период не найден
    """
    n = len(seq)
    end = min(n, skip + tail_len)
    tail = seq[skip:end]

    max_m = min(max_m, len(tail) // 2)

    for m in range(1, max_m + 1):
        a = tail[m:]
        b = tail[:-m]
        if np.all(np.abs(a - b) < eps):
            return m

    return 0


def plot_periods(r_values: np.ndarray, periods: np.ndarray) -> None:
    """Строит график зависимости длины цикла m от параметра r."""
    plt.figure()
    plt.plot(r_values, periods, marker="o", linestyle="")

    plt.xlabel("$r$")
    plt.ylabel("$m$")
    plt.grid(True)
    plt.title("Длина цикла $m$ для логистического отображения")

    plt.savefig("src/images/experimental.png")
    plt.show()


def main() -> None:
    """Строит графики лестницы Ламерея для логистического отображения при
    разных заданных r, а также графики для эмпирического исследования
    ограничений циклов порядка m"""
    rng = np.random.default_rng()

    x_0: float = rng.uniform(0.01, 0.99)
    n_steps: np.int64 = np.int64(rng.integers(40, 80))

    r_values: np.ndarray = np.array([3.2, 3.5, 3.568], dtype=float)

    for r in r_values:
        plot_staircase(
            x_0=x_0,
            r=float(r),
            n_steps=n_steps,
            save_path=f"src/images/lamere_r_{r:.3f}.png")

    r_infty: float = 3.5699456
    x_0 = float(rng.uniform(0.01, 0.99))
    r_values = np.linspace(3.001, r_infty, 150, dtype=float)
    n_steps = np.int64(1500)

    periods = np.empty(len(r_values), dtype=int)
    for i, r in enumerate(r_values):
        seq = generate_sequence(x_0, float(r), n_steps, "log")
        periods[i] = estimate_period(seq=seq)
    plot_periods(r_values, periods)


if __name__ == "__main__":
    main()