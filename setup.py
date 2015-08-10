from setuptools import setup
from setuptools import find_packages

setup(name='lfw_fuel',
      version='0.1.0',
      description='Labeled Faces in the Wild fuel dataset',
      author='Tom White',
      author_email='tom@sixdozen.com',
      url='https://github.com/dribnet/lfw_fuel',
      license='MIT',
      install_requires=['kerosene', 'fuel'],
      packages=find_packages())
