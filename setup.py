from setuptools import setup, find_packages
from codecs import open
from os import path

setup(
  name='PyCasper',
  version='0.1.1.dev3',
  url='https://github.com/chahuja/pycasper',
  author='Chaitanya Ahuja',
  author_email='ahujachaitanya@gmail.com',
  packages=['pycasper',],
  license='Creative Commons Attribution-Noncommercial-Share Alike license',
  long_description=open('README.md').read(),
  python_requires='>=3',
)
