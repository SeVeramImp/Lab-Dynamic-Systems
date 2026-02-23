import numpy as np
import matplotlib.pyplot as plt
from easy import logistic_map, N0_map

' Реализует графическое построение для задач секции Normal'


def generate_sequence(x_0: float, r: float, n_steps: np.int64,
                      map: str) -> np.ndarray:
    """
    Создаёт последовательность {x_n}

    Args:
        x_0 (float): начальная точка последовательности
        r (float): управляющий параметр отображения
        n_steps (np.int64): количество итераций для достижения x_n члена
         последовательности
        map (str): выбранное отображение, задающее последовательность

    Returns:
        (np.ndarray): последовательность {x_n}
    """
    seq = np.empty(n_steps + 1, dtype=float)
    seq[0] = x_0
    for n in range(n_steps):
        match (map):
            case ("log"): seq[n + 1] = logistic_map(seq[n], r)
            case ("N0"): seq[n + 1] = N0_map(seq[n], r)
    return seq


def plot_logistic_sequence(x_0: float, r: float, n_steps: np.int64) -> None:
    """
    Строит график зависимости x_n от n для последовательности {x_n},
    заданной логистическим отображением, для заданного r.
    """
    seq: np.ndarray = generate_sequence(x_0, r, n_steps, "log")

    n: np.ndarray = np.arange(int(n_steps) + 1)
    plt.figure()
    plt.plot(n, seq, marker="o")

    plt.title(f"Последовательность $x_n$ ($x_0$={x_0:.3f}, $r$={r:.3f})")
    plt.xlabel("$n$")
    plt.ylabel("$x_n$")
    plt.grid(True)

    plt.savefig("src/images/seq.png", bbox_inches="tight")
    plt.show()


def plot_N0_sequence(x_0: float, r_values: np.ndarray,
                     n_steps: np.int64) -> None:
    """
    Строит графики зависимости x_n от n для последовательности {x_n},
    заданной отображением N0_map, при нескольких значениях r.
    """
    n: np.ndarray = np.arange(int(n_steps) + 1)
    plt.figure()

    for r in r_values:
        seq: np.ndarray = generate_sequence(x_0, float(r), n_steps, "N0")
        plt.plot(n, seq, marker="o", label=f"$r$={float(r):.3f}")

    plt.title(f"$N0\\_map$: зависимость $x_n$ от $n$ ($x_0$={x_0:.3f})"
              "при разных $r$")
    plt.xlabel("$n$")
    plt.ylabel("$x_n$")
    plt.grid(True)
    plt.legend()

    plt.savefig("src/images/N0_seq.png")
    plt.show()


def plot_subsequences(r: float, n_steps: np.int64) -> None:
    """
    Строит графики подпоследовательностей {x_2n} и {x_2n+1} для логистического
    отображения и отмечает неподвижную точку x^*=(r-1)/r.

    Подбирает x_0 так, чтобы выполнялось условие x_{2n} > x^* для всех чётных n>=1.
    """
    rng = np.random.default_rng()

    x_still: float = (r - 1.0) / r

    n: np.ndarray = np.arange(int(n_steps) + 1)
    plt.figure()

    is_even: bool = False
    while is_even is False:
        x_0: float = float(rng.uniform(0.01, 0.99))
        seq: np.ndarray = generate_sequence(x_0, r, n_steps, "log")
        is_even = bool(np.all(seq[2::2] > x_still))

    plt.plot(n, seq, linestyle="-")
    plt.plot(n[0::2], seq[0::2], "o", label=r"$x_{2n}$")
    plt.plot(n[1::2], seq[1::2], "o", label=r"$x_{2n+1}$")
    plt.axhline(y=x_still, label=r"$x^*$")

    plt.title(f"$r$={r:.3f}, $x^*$={x_still:.3f}")
    plt.xlabel("$n$")
    plt.ylabel("$x_n$")
    plt.grid(True)
    plt.legend()

    plt.savefig("src/images/stilldot.png", bbox_inches="tight")
    plt.show()


def main() -> None:
    "Вызывает три функции визуализации для секции Normal."
    rng = np.random.default_rng()

    x_0: float = rng.uniform(0.01, 0.99)
    r: float = rng.uniform(0.01, 1.0)
    n_steps: np.int64 = rng.integers(15, 30)
    plot_logistic_sequence(x_0, r, n_steps)

    x_0 = rng.uniform(0.01, 0.99)
    r_values: np.ndarray = rng.uniform(0.01, 1.0, size=5)
    n_steps = rng.integers(15, 30)
    plot_N0_sequence(x_0, r_values, n_steps)

    r = rng.uniform(2.01, 2.99)
    n_steps = rng.integers(15, 30)
    plot_subsequences(r, n_steps)


if __name__ == "__main__":
    main()
