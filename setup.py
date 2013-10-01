try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='KMeans',
    version='0.1.0',
    author='Jakub Konka',
    author_email='kubkon@gmail.com',
    packages=['kmeans', 'kmeans.test'],
    url='https://github.com/kubkon/kmeans',
    license='LICENSE.txt',
    description='Simple K-Means.',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy>=1.7.1",
        "matplotlib>=1.3.0",
    ],
)