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


def main() -> None:
    """
    Строит графики зависимости x_n от n для последовательности {x_n}, заданной
    логистическим отображением, её подпоследовательностей {x_2n}, {x_2n+1},
    а также для последовательности {x_n}, заданной произвольным точечным
    отображением, сохраняет результат в src/images
    """
    rng = np.random.default_rng()

    x_0: float = rng.uniform(0.01, 0.99)
    r: float = rng.uniform(0.01, 1.0)
    n_steps: np.int64 = rng.integers(15, 30)

    seq1: np.ndarray = generate_sequence(x_0, r, n_steps, "log")

    n: np.ndarray = np.arange(n_steps + 1)
    plt.figure()
    plt.plot(n, seq1, marker="o")

    plt.title(f"Последовательность $x_n$ ($x_0$={x_0:.3f}, $r$={r:.3f})")
    plt.xlabel("$n$")
    plt.ylabel("$x_n$")
    plt.grid(True)

    plt.savefig("src/images/seq.png", bbox_inches="tight")
    plt.show()

    ########################################################################

    x_0 = rng.uniform(0.01, 0.99)
    n_steps = rng.integers(15, 30)

    r_values: np.ndarray = rng.uniform(0.01, 1.0, size=5)

    n = np.arange(n_steps + 1)
    plt.figure()

    for r in r_values:
        seq3: np.ndarray = generate_sequence(x_0, r, n_steps, "N0")
        plt.plot(n, seq3, marker="o", label=f"$r$={r:.3f}")

    plt.title(f"$N0\\_map$: зависимость $x_n$ от $n$ ($x_0$={x_0:.3f})"
              "при разных r")
    plt.xlabel("$n$")
    plt.ylabel("$x_n$")
    plt.grid(True)
    plt.legend()

    plt.savefig("src/images/N0_seq.png", bbox_inches="tight")
    plt.show()

    ########################################################################

    r = rng.uniform(2.01, 2.99)
    n_steps = rng.integers(15, 30)

    x_still: float = (r - 1) / r

    n = np.arange(n_steps + 1)
    plt.figure()

    even_ok: bool = False
    while even_ok is False:
        x_0 = rng.uniform(0.01, 0.99)
        seq2: np.ndarray = generate_sequence(x_0, r, n_steps, "log")
        even_ok = np.all(seq2[2::2] > x_still)
    plt.plot(n, seq2, linestyle="-")

    plt.plot(n[0::2], seq2[0::2], "o", label=r"$x_{2n}$")
    plt.plot(n[1::2], seq2[1::2], "o", label=r"$x_{2n+1}$")

    plt.axhline(y=x_still, label=r"$x^*$")

    plt.title(f"$r$={r:.3f}, $x^*$={x_still:.3f}")
    plt.xlabel("$n$")
    plt.ylabel("$x_n$")
    plt.grid(True)
    plt.legend()

    plt.savefig("src/images/stilldot.png")
    plt.show()


if __name__ == "__main__":
    main()
