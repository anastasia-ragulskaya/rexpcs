from typing import NamedTuple, Tuple

import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import LogNorm


def get_cmap(y):
    y = np.array(y)
    norm = mpl.colors.Normalize(vmin=y.min(), vmax=y.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap='jet')
    cmap.set_array([])
    return cmap


class MapSize(NamedTuple):
    t_size: int
    x_size: int
    number_of_particles: int
    dim: int = 2

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.t_size, *self.image_shape)

    @property
    def image_shape(self) -> Tuple[int, ...]:
        return tuple(self.x_size for _ in range(self.dim))

    def asdict(self) -> dict:
        return dict(t_size=self.t_size,
                    x_size=self.x_size,
                    number_of_particles=self.number_of_particles,
                    dim=self.dim,
                    )


def calc_polygons_new(startx, starty, endx, endy, radius):
    sl = (2 * radius) * np.tan(np.pi / 6)

    # calculate coordinates of the hexagon points
    p = sl * 0.5
    b = sl * np.cos(np.radians(30))
    w = b * 2
    h = 2 * sl

    # offsets for moving along and up rows
    xoffset = b
    yoffset = 3 * p

    row = 1

    shifted_xs = []
    straight_xs = []
    shifted_ys = []
    straight_ys = []

    while startx < endx:
        xs = [startx, startx, startx + b, startx + w, startx + w, startx + b, startx]
        straight_xs.append(xs)
        shifted_xs.append([xoffset + x for x in xs])
        startx += w

    while starty < endy:
        ys = [starty + p, starty + (3 * p), starty + h, starty + (3 * p), starty + p, starty, starty + p]
        (straight_ys if row % 2 else shifted_ys).append(ys)
        starty += yoffset
        row += 1

    polygons = [zip(xs, ys) for xs in shifted_xs for ys in shifted_ys] + [zip(xs, ys) for xs in straight_xs for ys in
                                                                          straight_ys]
    return polygons


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    for z in calc_polygons_new(startx=0,
                               starty=0,
                               endx=15,
                               endy=15,
                               radius=10):
        for xs, ys in z:
            for x, y in z:
                plt.scatter(x, y)
    plt.show()
