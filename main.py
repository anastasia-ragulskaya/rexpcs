import logging

import numpy as np

from particle_simulation import *

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    map_size = MapSize(t_size=100,
                       x_size=512,
                       number_of_particles=512,
                       dim=2
                       )
    init_radius = 2
    max_radius = 9
    name = 'brownian_viscosity_60_20'
    number_of_xpcs_plots = 50
    viscosity =  np.linspace(60, 20, map_size.t_size)[:, None]
    save_animation = True
    no_init_touch = True
    min_distance = 5
    radius_distribution = (0.2, 0.7)
    config_dict = dict(max_radius=max_radius,
                       radius_distribution=radius_distribution,
                       no_init_touch=no_init_touch,
                       min_distance=min_distance,
                       init_radius=init_radius)

    config_dict.update(map_size.asdict())

    # c = Concentrations(map_size).destroy_diff_speeds(share=1,
    #                                                  speed_rate=10,
    #                                                  value_rate=10,
    #                                                  distribution=(0.001, 0.002))
    # c = Concentrations(map_size).born_again(share=0.5,
    #                                         speed_rate=10,
    #                                         value_rate=10,
    #                                         distribution=(0.001, 0.002))
    # c = Concentrations(map_size).linear(-1)
    c = Concentrations(map_size).constant()
    # radius = Radius(
    #     map_size, max_radius=max_radius).destroy_twice(
    #     speed_rate=10, value_rate=10, distribution=(0.2, 0.5))
    # radius = Radius(map_size, max_radius=max_radius).destroy_diff_speeds(
    #     init_value=0,
    #     share=1,
    #     speed_rate=4, value_rate=5, distribution=(0.2, 0.5))
    # radius = Radius(map_size, max_radius=max_radius).linear(
    #     0,
    #     distribution=(1, 0.5))
    # radius = Radius(map_size, max_radius=max_radius).saturation_and_destroy(
    #     init_value=init_radius, saturation_time=0.5,
    #     destroy_time=0.25, distribution=radius_distribution)
    radius = Radius(map_size, max_radius=max_radius).constant(init_radius)
    # radius = Radius(map_size, max_radius=max_radius).destroy_part(
    #     init_value=init_radius, distribution=radius_distribution)
    # radius = Radius(map_size, max_radius=max_radius).constant(max_radius)
    #centers = Centers(map_size).random_walk(
     #    randow_walk_speed, no_init_touch=no_init_touch,
      #   radius=radius, min_distance=min_distance)

    centers = Centers(map_size).brownian_motion(radius, viscosity=viscosity)
    # centers = Centers(map_size).no_touching(radius, min_distance)

    simulation = Simulation(radius=radius,
                            centers=centers,
                            c=c,
                            map_size=map_size)

    simulation.run()
    save = SaveSimulation(name, simulation)
    save.save(number_of_xpcs_plots, save_animation)
    save.save_config(config_dict)
