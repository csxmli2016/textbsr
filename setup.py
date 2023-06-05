
from setuptools import setup, find_packages
import os
import codecs

def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires

def read(fname):
	return codecs.open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name='textbsr',
      version='0.1.12',
      description='a simple version for blind text image super-resolution (current version is only for English and Chinese)',
      author='Xiaoming Li',
      author_email='csxmli@gmail.com',
      long_description=read("README.md"),
      long_description_content_type="text/markdown",
      packages=find_packages(),  #
      #
      license="S-Lab License 1.0",
      keywords='blind text image super-resolution',
      url='https://github.com/csxmli2016/MARCONet',
      include_package_data=True,
      install_requires=get_requirements(),
      python_requires='>=3.6',
      entry_points = {
        'console_scripts': [
            'textbsr = textbsr.textbsr:textbsr',
        ],
      }
      )
