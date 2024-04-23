from setuptools import setup, find_packages
import os

def read_md(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return "Long description not available"

setup(
    name='WebSeeker',
    version='0.0.0.1',
    packages=find_packages(),
    description='A package to enable commonly used functions through 1 line of code.',
    long_description=read_md('readme.md'),
    long_description_content_type='text/markdown',
    author='Stefanie Eriksson',
    author_email='stefanie1@protonmail.com',
    url='https://github.com/stefanieeriksson/pyfixer',
    install_requires=[
        'pandas',
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)