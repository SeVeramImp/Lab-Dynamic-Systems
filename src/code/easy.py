import numpy as np
import matplotlib.pyplot as plt

' Реализует графическое построение для задач секции Easy'


def logistic_map(x: np.ndarray[np.float64], r: float) -> np.ndarray[np.float64]:
    """
    Реализует логистическое отображение: x_{n+1} = r * x_n * (1 - x_n)

    Args:
        x (np.ndarray[np.float64]): массив значений x_n
        r (float): параметр отображения

    Returns:
        np.ndarray[np.float64]: массив значений x_{n+1}
    """
    return r * x * (1 - x)


def N0_map(x: np.ndarray[np.float64], r: float) -> np.ndarray[np.float64]:
    """
    Реализует заданное вариантом N_0 точечное отображение:
    x_{n+1} = r * x_n * (1 - x_n) * (2 + x_n)

    Args:
        x (np.ndarray[np.float64]): массив значений x_n
        r (float): параметр отображения

    Returns:
        np.ndarray[np.float64]: массив значений x_{n+1}
    """
    return r * x * (1.0 - x) * (2.0 + x)


def main() -> None:
    """
    Строит графики логистического и произвольного точечного отображений
    при разных параметрах r, сохраняет результат в src/images
    """
    r_values: list[float] = [0.2, 0.5, 1.0, 2.0, 3.2, 4.0]
    x: np.ndarray[np.float64] = np.linspace(0.0, 1.0, 1000)

    plt.figure()

    for r in r_values:
        y: np.ndarray[np.float64] = logistic_map(x, r)
        plt.plot(x, y, label=f"r = {r}")

    plt.title("Логистическое отображение: $x_{n+1} = r x_n (1 - x_n)$")
    plt.xlabel("$x_n$")
    plt.ylabel("$x_{n+1}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("src/images/logistic_map.png")
    plt.show()

    ########################################################################

    r_max: float = 27.0 / (2.0 * (7.0 * np.sqrt(7.0) - 10.0))
    r_values = [0.2, 0.5, 1.0, r_max]

    plt.figure()
    for r in r_values:
        y = N0_map(x, r)
        plt.plot(x, y, label=f"r = {r:.4f}")

    plt.title("Точечное отображение: $x_{n+1}=r x_n (1-x_n) (2+x_n)$")
    plt.xlabel("$x_n$")
    plt.ylabel("$x_{n+1}$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("src/images/N0_map.png")
    plt.show()


if __name__ == "__main__":
    main()
