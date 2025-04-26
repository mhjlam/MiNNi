from setuptools import setup, find_packages

setup (
    name='MiNNi',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    author='Maurits Lam',
    description='Mini Neural Network',
    python_requires='>=3.6',
)
