from distutils.core import setup

setup(
    name='KMeans',
    version='0.1.0',
    author='Jakub Konka',
    author_email='kubkon@gmail.com',
    packages=['kmeans', 'kmeans.test'],
    url='',
    license='LICENSE.md',
    description='Simple K-Means.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy>=1.7.1",
        "matplotlib>=1.3.0",
    ],
)