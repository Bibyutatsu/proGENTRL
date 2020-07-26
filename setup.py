import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='progentrl',
    version='1.0.0',
    description="Pro Generative Tensorial Reinforcement Learning (proGENTRL): Leverage the power using pytorch Lightning and generate molecules",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Bibyutatsu/proGENTRL",
    author="Bibyutatsu",
    author_email="bibhashm220896@gmail.com",
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6'
    ],
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.5.0',
    install_requires=[
        'numpy>=1.15',
        'pandas>=0.23',
        'scipy>=1.1.0',
        'molsets',
        'scikit-learn>=0.21.3',
        'pytorch-lightning>=0.7.6',
        'joblib>=0.13.2'
    ]
)
