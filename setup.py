from setuptools import setup, find_packages
from pathlib import Path


def get_version(package_name: str) -> str:
    local_dict = {}
    with open(str(Path(__file__).parent / package_name / '__version.py'), 'r') as f:
        exec(f.read(), {}, local_dict)

    return local_dict['__version__']


PACKAGE_NAME = 'particle_simulation'

__version__ = get_version(PACKAGE_NAME)

with open('requirements.txt') as req_file:
    install_requires = req_file.read().splitlines()


classifiers = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GPLv3 License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
]

setup(
    name=PACKAGE_NAME,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    version=__version__,
    author='Vladimir Starostin',
    author_email='vladimir.starostin@uni-tuebingen.de',
    description='Particle simulation for Reverse Engineering',
    license='GPLv3',
    python_requires='>=3.6',
    install_requires=install_requires
)
