from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name='spie',
    version='0.1.0',
    author='Samuel Sokota',
    author_email='ssokota@gmail.com',
    license='LICENSE.txt',
    description='Sample-based methods for simultaneous prediction interval estimation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ssokota/spie',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=["numpy >=1.15.0", "matplotlib >= 2.0.0"])
