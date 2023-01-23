# RE_xpcs

Particle simulation with correlation maps for Reverse-engineering  of X-ray Photon Correlation Spectroscopy (XPCS) experiments. 

The package provides the source code used in the following scientific publication:

[1] "Reverse-engineering method for XPCS studies of non-equilibrium dynamics", Anastasia Ragulskaya, Vladimir Starostin, Nafisa Begam, Anita Girelli, Hendrik Rahmann, Mario Reiser, Fabian Westermeier, Michael Sprung, Fajun Zhang, Christian Gutt and Frank Schreiber, IUCrJ 9 (2022), 439.

## Main dependencies

To be able to run the code, a python 3.7 installation with the following dependencies is required:

* tqdm
* matplotlib
* numpy
* scipy
* h5py

# Install

Install package via pip: 

```python
pip install rexpcs
```

# Usage

The programm simulates the behaviour of particles with user-defined parameters in $N$-dimensional real space. It can be considered as the evolution of $N$-dimensional concentration maps. Fast Fourier transform of the fluctuations of the concentrations gives information about the speckle pattern (i.e. the image in reciprocal space $I(q,t_{age}=(t_1+t_2)/2)$ ). This information is then used to calculate two-time correlation maps (TTC):

$$ G(q,t_1,t_2)=\frac{\overline{I(t_1)I(t_2)}-\overline{I(t_1)}\cdot\overline{I(t_2)}}{[\overline{I^2(t_1)}-\overline{I(t_1)}^2]^{\frac{1}{2}}\cdot[\overline{I^2(t_2)}-\overline{I(t_2)}^2]^{\frac{1}{2}}},$$

where the average is performed over pixels within the same momentum transfer $q \pm \Delta q$. $t_1$ and $t_2$ are the times at which the intensity correlation is calculated. The example of the TTC is presented below:

![](/saved_data/brownian_viscosity_60_20/xpcs_q_101.00.png)

## Parameters of simulation
The user can define the following parameters (in main.py):

* time of the simulation (*t_size*)
* map  (*x_size*)
* number of particles (*number_of_particles*)
* dimension (*dim*)
* initial radius (*init_radius*)
* final radius (*max_radius*)
* position: minimal distance between the particles (*min_distance*)
* radius distribution (*radius_distribution*)
* concentration (*c*)

To manipulate the evolution of described parameters different **functions)** are available (in particle_factories).

Concentration:
* concentration is a constant value (*constant*)
* user-defined evolution with 1 function (*function*)
* user-defined evolution with n functions (*n_functions*)
* linear change (*linear*)
* square change (*sqrt*)
* part of the particles has increase in the value, part has a decrease (*destroy_part*)

Centers:
* uniform distribution (*constant_uniform*)
* define, if particles touch (*no_touching*)
* centers motion can be described as Brownian motion (*brownian_motion*), thus, the value depends on radii of the particles
* random walk motion (*random_walk*)

Radius:
similar to concentrations

The results are saved in .h5 file (save_h5.py), furthermore the configuration is saved in config.json. Here is the example of a config file:

```con{"max_radius": 9, "radius_distribution": [0.2, 0.7], "no_init_touch": true, "min_distance": 5, "init_radius": 2, "t_size": 100, "x_size": 512, "number_of_particles": 512, "dim": 2}```

The programm creates the folder in \saved_data\name folder, where name for saving is defined by user in the main.py. The folder consists of animation (gif) of the simulation in the real space, calculation of kinetics (averaged over all pixels $I(q,t_{age}$ ) as a function of $q$ and with recalculation to the $radii$. It also includes plots of the evolution of $center$ , $concentration$ , and $radii$ , as well as TTCs for all $q$-values.

Therefore, the relation between the parameters and TTC features can be identified as shown in [1]. 






