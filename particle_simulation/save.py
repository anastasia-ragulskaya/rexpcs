import logging
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from h5py import File

import matplotlib.pyplot as plt

from .simulation import Simulation

logger = logging.getLogger(__name__)


class SaveSimulation(object):
    project_dir = Path(__file__).parents[1]
    data_dir = project_dir / 'saved_data'

    def __init__(self, name: str,
                 simulation: Simulation):
        self.name = name
        self.simulation = simulation
        self.current_dir = self.data_dir / name
        logger.debug(f'Current dir: {self.current_dir}')

    def save_config(self, config_dict: dict):
        with open(self.current_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f)

    def save_parameters(self):
        r = self.simulation.radius
        c = self.simulation.c

        centers = [(center, f'Center {i + 1}') for i, center in enumerate(self.simulation.centers)]

        for param, name in ((r, 'Radius'),
                            (c, 'Concentration'),
                            *centers):
            plt.figure()
            for p in range(self.simulation.number_of_particles):
                plt.plot(param[:, p])
            plt.title(name)
            ax = plt.gca()
            ax.set_xlabel('Time')
            ax.set_ylabel(name)
            ax.grid(linestyle='--', linewidth=0.5)
            plt.savefig(self.current_dir / f'parameter_{name}.png')
            plt.close()

    def save(self,
             number_of_xpcs: int = 30,
             save_animation: bool = True,
             *,
             mode: str = 'std',
             q_min: float = 0,
             q_max: float = None,
             dq: float = 3,
             save_kinetics: bool = True,
             save_parameters: bool = True,
             save_real_space: bool = False,
             save_recip_space: bool = False,
             ):

        Path.mkdir(self.current_dir)

        if q_max is None:
            q_max = self.simulation.x_size // 2
        else:
            q_max = min(q_max, self.simulation.x_size // 2)
        qs = np.linspace(q_min, q_max, number_of_xpcs)

        with File(self.current_dir / 'data.h5', 'w') as f:
            f.attrs.update(self.simulation.map_size.asdict())
            if save_real_space:
                f.create_dataset('u', data=self.simulation.real_space)
            if save_recip_space:
                f.create_dataset('ft', data=self.simulation.recip_space)
            if save_kinetics:
                qs, rs, ts, kinetics = self.simulation.get_kinetics()
                kinetics_group = f.create_group('kinetics')
                kinetics_group.create_dataset('kinetics', data=kinetics)
                kinetics_group.create_dataset('qs', data=qs)
                kinetics_group.create_dataset('rs', data=rs)
                kinetics_group.create_dataset('ts', data=ts)

                self.simulation.plot_kinetics(
                    file_paths=(self.current_dir / 'kinetics_R.png',
                                self.current_dir / 'kinetics_Q.png')
                )

            ttc_group = f.create_group('ttcs')
            for q in tqdm(qs):
                logger.debug(f'Plotting q = {q:.2f}')
                plt.figure()
                ttc = self.simulation.get_two_time_corr_function(q, dq, mode=mode)
                self.simulation.plot_xpcs(q, dq, mode=mode, ttc=ttc, show=False)
                plt.savefig(self.current_dir / f'xpcs_q_{q:.2f}.png')
                plt.close()
                dset = ttc_group.create_dataset(str(q), data=ttc)
                dset.attrs.update(dict(q=q, dq=dq, mode=mode))

            if save_parameters:
                param_group = f.create_group('parameters')
                param_group.create_dataset('c', data=self.simulation.c)
                param_group.create_dataset('centers', data=self.simulation.centers)
                param_group.create_dataset('radii', data=self.simulation.radius)
                self.save_parameters()
            if save_animation:
                self.simulation.save_animation(self.current_dir / f'animation_{self.name}.gif')


