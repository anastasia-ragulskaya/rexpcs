import logging
from typing import Tuple, List

from tqdm import tqdm
import h5py

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np
from scipy.fftpack import fftshift, fftn

from .utils import get_cmap, MapSize

logger = logging.getLogger(__name__)


class Simulation(object):
    H5FilePath = 'simulations.h5'

    def __init__(self, *,
                 map_size: MapSize,
                 radius: np.array,
                 c: np.array,
                 centers: Tuple[np.array, np.array],
                 init_c: float = -1,
                 recalculate_background: bool = False):

        if recalculate_background:
            raise NotImplementedError('Cannot recalculate yet.')

        self.radius = radius
        self.c = c
        self.centers = np.array(centers)
        self.recalculate_background = recalculate_background
        self.map_size = map_size
        self.t_size = map_size.t_size
        self.x_size = map_size.x_size
        self.number_of_particles = map_size.number_of_particles
        self.dim = map_size.dim

        self._check_input()

        self.init_c = init_c
        self.real_space = np.zeros(map_size.shape)

        self._coords = np.array(np.meshgrid(*[np.arange(self.x_size) for _ in range(self.dim)]))

        self.recip_space = np.zeros_like(self.real_space)

    def __getitem__(self, t: int):
        return self.real_space[t]

    def _get_mean_sf(self, q, dq):
        mask, number_of_pixels = self.get_mask(q, dq)
        if number_of_pixels == 0:
            return
        kinetics = np.zeros(self.t_size)
        for t in range(self.t_size):
            kinetics[t] = self.recip_space[t, mask].mean()
        return kinetics

    def get_kinetics(self, dq: int = 3,
                     q_min: int = 2,
                     dt: int = 10):
        qs = np.arange(q_min, self.x_size // 2, dq)
        rs = self.x_size / (qs + dq / 2)
        ts = np.arange(0, self.t_size, dt)
        kinetics = np.zeros((self.t_size, qs.size))
        for i, q in enumerate(qs):
            y = self._get_mean_sf(q, dq)
            if y is not None:
                kinetics[:, i] = y

        return qs, rs, ts, kinetics

    def plot_kinetics(self, dq: int = 3,
                      q_min: int = 2,
                      dt: int = 10,
                      file_paths: Tuple[str, str] = None):

        qs, rs, ts, kinetics = self.get_kinetics(dq, q_min, dt)
        cmap = get_cmap(ts)

        for t in ts:
            plt.plot(rs, kinetics[t, :], color=cmap.to_rgba(t))

        ax = plt.gca()
        ax.set_xlim([0, rs.max()])
        ax.set_ylim([0, kinetics.max()])
        ax.set_xlabel('R')
        cbar = plt.colorbar(cmap)
        cbar.set_label('Time')
        ax.grid(linestyle='--', linewidth=0.5)
        if file_paths is not None:
            plt.savefig(file_paths[0])
        else:
            plt.show()

        plt.figure()
        for t in ts:
            plt.plot(qs, kinetics[t, :], color=cmap.to_rgba(t))

        ax = plt.gca()
        ax.set_xlim([0, qs.max()])
        ax.set_ylim([0, kinetics.max()])
        ax.set_xlabel('Q')
        cbar = plt.colorbar(cmap)
        cbar.set_label('Time')
        ax.grid(linestyle='--', linewidth=0.5)

        if file_paths is not None:
            plt.savefig(file_paths[1])
        else:
            plt.show()

    def _check_input(self):
        shape = (self.t_size, self.number_of_particles)

        def check(x_, name_):
            if x_.shape != shape:
                raise ValueError(f'{name_} shape is wrong:\n'
                                 f'{x_.shape} != {shape}')

        centers = [(center, f'Center {i + 1}') for i, center in enumerate(self.centers)]

        for x, name in (
                (self.radius, 'Radius'),
                (self.c, 'Concentration'),
                *centers
        ):
            check(x, name)

    def _expand_to_dim(self, arr):
        arr_dim = len(arr.shape)
        return np.expand_dims(arr, [d + arr_dim for d in range(self.dim)])

    def run(self, *,
            noise: Tuple[float, float] = None
            ):

        centers = self._expand_to_dim(self.centers)
        c = self.c
        radius = self.radius
        image_shape = self.map_size.image_shape

        for t in tqdm(np.arange(self.t_size)):
            image = np.ones(image_shape) * self.init_c
            for p in range(self.number_of_particles):
                mask = np.sum((self._coords - centers[:, t, p]) ** 2, 0) <= radius[t, p] ** 2
                image[mask] = c[t, p]

            if noise is not None:
                mu, sigma = noise
                image += np.random.lognormal(
                    mu, sigma, image_shape)

            self.real_space[t] = image
            self.recip_space[t] = np.abs(fftshift(fftn(image)))

    def save_animation(self, filename: str):
        if self.dim != 2:
            return

        print(f'Saving animation to {filename} ... ')

        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.x_size),
                      ylim=(0, self.x_size))

        img = ax.imshow(self[0], cmap='jet', animated=True,
                        vmin=-1, vmax=self.real_space.max())
        plt.colorbar(img, ax=ax)

        def init():
            return img,

        def animate(i):
            plt.title(f'Time = {i}')
            img.set_data(self[i])
            return img,

        anim = FuncAnimation(fig, animate,
                             init_func=init,
                             frames=self.t_size - 1,
                             interval=20, blit=True)

        anim.save(filename, writer='imagemagick', fps=30)
        print(f'Animation saved to {filename}.')

    def r_str_from_q(self, q, dq):
        if q == 0:
            r = self.x_size / dq
            return f'r >= {r:.2f}\n' \
                   f'q <= {dq:.2f}'
        r1 = self.x_size / q
        r2 = self.x_size / (q + dq)
        return f'{r2:.2f} <= r <= {r1:.2f}\n' \
               f'{q:.2f} <= q <= {(q + dq):.2f}'

    def get_mask(self, q: float, dq: float):

        center = self.x_size / 2
        rr = np.sum((self._coords - center) ** 2, 0)
        mask = (rr >= q ** 2) & (rr <= (q + dq) ** 2)
        number_of_pixels = self.recip_space[0, mask].size

        return mask, number_of_pixels

    def get_q_array(self,
                    q: float,
                    dq: float,
                    mode='std'):

        mask, number_of_pixels = self.get_mask(q, dq)
        logger.info(f'Number of pixels within {q:.1e} - {(q + dq):.1e} = '
                    f'{number_of_pixels}')
        if number_of_pixels == 0:
            return

        q_array = self.recip_space[:, mask].T

        if mode == 'mean':
            q_array /= q_array.mean(axis=0)[None]

        return q_array

    def get_two_time_corr_function(self,
                                   q: float,
                                   dq: float,
                                   mode='std'):
        if mode not in ['mean', 'std']:
            raise ValueError("Mode should be one of 'mean', 'std'.")
        int_a = self.get_q_array(q, dq, mode)
        if int_a is None:
            return
        if mode == 'mean':
            return int_a.T.dot(int_a) / int_a.shape[0]
        else:
            # package code:
            # intensity_array = np.swapaxes(int_a, 0, 1)
            # std_array = intensity_array.std(axis=1)[:, np.newaxis]
            # std_array[std_array == 0] = 1
            # intensity_array /= std_array
            # res = intensity_array.dot(intensity_array.transpose()) / intensity_array.shape[1]
            # i_mean = np.expand_dims(intensity_array.mean(axis=1), axis=1)
            # res -= i_mean.dot(i_mean.transpose())
            # return res

            # original code:
            std = int_a.std(0)[np.newaxis]
            std[std == 0] = 1
            std = std.T.dot(std)
            mean = int_a.mean(0)[np.newaxis]
            mean = mean.T.dot(mean)
            return (int_a.T.dot(int_a) / int_a.shape[0] - mean) / std

    def plot_xpcs(self, q, dq, mode: str = 'std', ttc: np.ndarray = None, show: bool = True):
        if ttc is None:
            ttc = self.get_two_time_corr_function(q, dq, mode)
        if ttc is None:
            return

        plt.imshow(ttc, cmap='jet', origin='lower', vmin=0, vmax=1)
        _, number_of_pixels = self.get_mask(q, dq)
        plt.title(f'{self.r_str_from_q(q, dq)},'
                  f' {number_of_pixels} pixels')
        plt.colorbar()
        if show:
            plt.show()
