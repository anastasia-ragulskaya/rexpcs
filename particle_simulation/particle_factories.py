import logging

import numpy as np

from particle_simulation import MapSize

logger = logging.getLogger(__name__)


def _expand(x: np.array,
            n: int,
            axis: int = 1):
    return np.expand_dims(x, axis=axis).repeat(n, axis=axis)


def _add_distribution(x: np.array,
                      mu: float,
                      sigma: float = None):
    """
    Adds lognormal distribution along number_of_particles
    axis (axis=1).
    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    if not sigma:
        return x
    distr = np.random.lognormal(mu, sigma, (1, x.shape[1])).repeat(
        x.shape[0], axis=0
    )
    return x + distr


def _custom_shuffle(x: np.array):
    indices = np.arange(x.shape[1])
    np.random.shuffle(indices)
    return x[:, indices]


class Concentrations(object):
    MaxValue = 1

    def __init__(self, map_size: MapSize):
        self.number_of_particles = map_size.number_of_particles
        self.t_size = map_size.t_size

    def constant(self, value: float = 1, **kwargs):
        c = np.ones((self.t_size, self.number_of_particles)) * value
        if 'distribution' in kwargs.keys():
            c = _add_distribution(c, *kwargs['distribution'])
        return c

    def function(self, func, init_value, **kwargs):
        t = np.linspace(0, 1, self.t_size)
        c = init_value + func(t)
        c = _expand(c, self.number_of_particles)
        if 'distribution' in kwargs.keys():
            c = _add_distribution(c, *kwargs['distribution'])
        return c

    def two_functions(self, func1, func2, share,
                      init_value, shuffle: bool = True, **kwargs):
        t = np.linspace(0, 1, self.t_size)
        n1 = int(self.number_of_particles * share)
        n2 = self.number_of_particles - n1

        c1 = init_value + func1(t)
        c2 = init_value + func2(t)
        c1 = _expand(c1, n1)
        c2 = _expand(c2, n2)
        c = np.concatenate([c1, c2], axis=1)
        if shuffle:
            c = _custom_shuffle(c)
        if 'distribution' in kwargs.keys():
            c = _add_distribution(c, *kwargs['distribution'])
        return c

    def n_functions(self, funcs: tuple, shares: tuple,
                    init_value, shuffle: bool = True, **kwargs):
        t = np.linspace(0, 1, self.t_size)
        cs = list()
        assert len(funcs) == len(shares) + 1
        ns = [int(self.number_of_particles * share)
              for share in shares]
        ns.append(self.number_of_particles - sum(ns))

        for func, n in zip(funcs, ns):
            c = init_value + func(t)
            c = _expand(c, n)
            cs.append(c)
        c = np.concatenate(cs, axis=1)
        if shuffle:
            c = _custom_shuffle(c)
        if 'distribution' in kwargs.keys():
            c = _add_distribution(c, *kwargs['distribution'])
        return c

    def born_again(self,
                   share: float = 0.5,
                   speed_rate: float = 1,
                   value_rate: float = 1,
                   second_moment: float = 0.5,
                   init_value: float = 0,
                   **kwargs):
        speed = 1 / (1 + speed_rate)
        value = (1 - 1 / (1 + value_rate)) * self.MaxValue

        n = int(self.t_size * speed)
        m = self.t_size - n
        k = int(second_moment * m)

        def func1(x):
            y1 = np.linspace(init_value, value, n)
            y2 = np.linspace(y1[-1], self.MaxValue, m)
            return np.concatenate([y1, y2], axis=0)

        def func2(x):
            y1 = np.linspace(init_value, value, n)
            y2 = np.linspace(y1[-1], 0, m)[:k]
            y3 = np.linspace(y2[-1], self.MaxValue,
                             x.size - y1.size - y2.size)
            return np.concatenate([y1, y2, y3], axis=0)

        return self.two_functions(func1, func2, share, init_value, **kwargs)

    def linear(self, init_value: float = 0,
               **kwargs):
        def func(x):
            return x * (self.MaxValue - init_value)

        return self.function(func, init_value, **kwargs)

    def sqrt(self, init_value: float = 0.5, **kwargs):
        def func(x):
            return np.sqrt(x) * self.MaxValue

        return self.function(func, init_value, **kwargs)

    def destroy_twice(self, shares: tuple = (0.5, 0.3),
                      speed_rate: float = 1,
                      value_rate: float = 1,
                      second_destroy_moment: float = 0.5,
                      init_value: float = 0,
                      **kwargs):
        speed = 1 / (1 + speed_rate)
        value = (1 - 1 / (1 + value_rate)) * self.MaxValue

        n = int(self.t_size * speed)
        m = self.t_size - n
        k = int(second_destroy_moment * m)

        def func1(x):
            y1 = np.linspace(init_value, value, n)
            y2 = np.linspace(y1[-1], self.MaxValue, m)
            return np.concatenate([y1, y2], axis=0)

        def func2(x):
            y1 = np.linspace(init_value, value, n)
            y2 = np.linspace(y1[-1], 0, m)
            return np.concatenate([y1, y2], axis=0)

        def func3(x):
            y1 = np.linspace(init_value, value, n)
            y2 = np.linspace(y1[-1], self.MaxValue, m)[:k]
            y3 = np.linspace(y2[-1], 0,
                             x.size - y1.size - y2.size)
            return np.concatenate([y1, y2, y3], axis=0)

        return self.n_functions((func1, func2, func3),
                                shares, init_value=init_value, **kwargs)

    def destroy_diff_speeds(self, share: float = 0.5,
                            speed_rate: float = 1,
                            value_rate: float = 1,
                            init_value: float = 0,
                            **kwargs):
        speed = 1 / (1 + speed_rate)
        value = (1 - 1 / (1 + value_rate)) * self.MaxValue

        n = int(self.t_size * speed)
        m = self.t_size - n

        def func1(x):
            y1 = np.linspace(init_value, value, n)
            y2 = np.linspace(y1[-1], self.MaxValue, m)
            return np.concatenate([y1, y2], axis=0)

        def func2(x):
            y1 = np.linspace(init_value, value, n)
            y2 = np.linspace(y1[-1], 0, m)
            return np.concatenate([y1, y2], axis=0)

        return self.two_functions(func1, func2, share, init_value, **kwargs)

    def destroy_part(self, share: float = 0.5,
                     speed_rate: float = 1,
                     init_value: float = 0,
                     **kwargs):
        def func1(x):
            return x * (self.MaxValue - init_value)

        def func2(x):
            n = 1 / (1 + speed_rate)
            m = 1 - n
            y = (x < n) * m * x + (n - x * n) * (x >= n)
            return (self.MaxValue / m - init_value) * y

        return self.two_functions(func1, func2, share, init_value, **kwargs)

    def saturation_and_destroy(self,
                               share: float = 0.5,
                               saturation_time: float = 0.5,
                               destroy_time: float = 0.5,
                               init_value: float = 0,
                               **kwargs):
        growth_time_1 = int(self.t_size * saturation_time)
        growth_time_2 = int(self.t_size * destroy_time)
        time_of_saturation = self.t_size - growth_time_1
        time_of_destroy = self.t_size - growth_time_2

        def func1(x):
            y1 = np.linspace(init_value, self.MaxValue, growth_time_1)
            y2 = np.linspace(self.MaxValue, self.MaxValue, time_of_saturation)
            return np.concatenate([y1, y2], axis=0)

        def func2(x):
            value = func1(0)[growth_time_2]
            y1 = np.linspace(init_value, value, growth_time_2)
            y2 = np.linspace(value, 0, time_of_destroy)
            return np.concatenate([y1, y2], axis=0)

        return self.two_functions(func1, func2, share, init_value, **kwargs)


class Centers(object):
    MaxIteration = 100

    def __init__(self, map_size: MapSize):
        self.map_size = map_size
        self.number_of_particles = map_size.number_of_particles
        self.t_size, self.x_size = map_size.t_size, map_size.x_size
        self.dim = map_size.dim

    def constant_uniform(self):

        centers = np.stack([
            _expand(np.random.uniform(0, self.x_size, self.number_of_particles), self.t_size, 0)
            for _ in range(self.dim)
        ])

        return centers

    def no_touching(self, radius, min_distance: float = 0):
        max_radius = [radius[:, i].max() for i in range(self.number_of_particles)]
        center_x = list()
        center_y = list()

        def touches(r_self, x, y, center_x, center_y) -> bool:
            for i, (r_other, x_other, y_other) in enumerate(zip(
                    max_radius, center_x, center_y
            )):
                distance = (x_other - x) ** 2 + (y_other - y) ** 2
                if (distance < (r_self + r_other + min_distance) ** 2
                        and distance != 0):
                    return True
            return False

        for i in range(self.number_of_particles):
            iteration = 0
            r = max_radius[i]
            cx = np.random.uniform(0, self.x_size)
            cy = np.random.uniform(0, self.x_size)
            while touches(r, cx, cy, center_x, center_y):
                iteration += 1
                logger.debug(f'iteration = {iteration}')
                if iteration > self.MaxIteration:
                    raise ValueError("C'est impossible!")
                cx = np.random.uniform(0, self.x_size)
                cy = np.random.uniform(0, self.x_size)
            center_x.append(cx)
            center_y.append(cy)

        center_x = _expand(np.array(center_x), self.t_size, 0)
        center_y = _expand(np.array(center_y), self.t_size, 0)
        return center_x, center_y

    def brownian_motion(self, radii: np.ndarray, viscosity: float = 1.):
        init = np.random.uniform(0, self.x_size, (self.dim, self.number_of_particles))

        centers = np.stack([
            brownian_motion(radii, viscosity, i) for i in init
        ])

        return centers

    def random_walk(self, speed: float = 0.1,
                    no_init_touch: bool = False,
                    *args, **kwargs):
        if self.dim != 2:
            raise NotImplementedError

        random_choices = np.random.randint(
            1, 5, (self.t_size, self.number_of_particles))
        if no_init_touch:
            center_x, center_y = self.no_touching(*args, **kwargs)
        else:
            center_x, center_y = self.constant_uniform()

        if speed > 0:

            for p in range(self.number_of_particles):
                for t in range(1, self.t_size):
                    val = random_choices[t, p]
                    if val == 1:
                        center_x[t, p] = center_x[t - 1, p] + speed
                        center_y[t, p] = center_y[t - 1, p]
                    elif val == 2:
                        center_x[t, p] = center_x[t - 1, p] - speed
                        center_y[t, p] = center_y[t - 1, p]
                    elif val == 3:
                        center_x[t, p] = center_x[t - 1, p]
                        center_y[t, p] = center_y[t - 1, p] + speed
                    else:
                        center_x[t, p] = center_x[t - 1, p]
                        center_y[t, p] = center_y[t - 1, p] - speed
        return center_x, center_y

    def structure_factor(self, distance: float = None):
        # n = int(np.sqrt(self.number_of_particles) + 1)
        raise NotImplementedError


def brownian_motion(radii: np.ndarray, viscosity: float = 1,
                    init_positions: np.ndarray = None):
    stds = 1 / radii / viscosity
    increments = np.random.normal(size=radii.shape) * np.sqrt(stds)

    if init_positions is not None:
        increments[0] = init_positions

    coords = np.cumsum(increments, 0)

    return coords


class Radius(Concentrations):
    def __init__(self, map_size: MapSize,
                 max_radius: float):
        super(Radius, self).__init__(map_size)
        self.MaxValue = max_radius


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    logging.basicConfig(level=logging.DEBUG)
    map_size_ = MapSize(100, 400, number_of_particles=50)
    # radius = Radius(map_size, 50).destroy_diff_speeds(
    #     speed_rate=5, value_rate=20)
    # radius = Radius(map_size, 50).destroy_twice(
    #     shares=(1 / 3, 1 / 3),
    #     speed_rate=5, value_rate=20)
    radius_ = Radius(map_size_, max_radius=10).destroy_diff_speeds(
        init_value=0,
        share=1,
        speed_rate=4, value_rate=5, distribution=(0.2, 0.5))
    center_x_, center_y_ = Centers(map_size_).constant_uniform()
    for n in range(map_size_.number_of_particles):
        plt.plot(radius_[:, n])
    plt.show()
