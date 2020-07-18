from setuptools import setup, find_packages


setup(
    name='progentrl',
    version='1.0',
    python_requires='>=3.5.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.15',
        'pandas>=0.23',
        'scipy>=1.1.0',
        'molsets',
        'scikit-learn>=0.21.3',
        'pytorch-lightning>=0.7.6',
        'joblib>=0.13.2'
    ],
    description='Pro Generative Tensorial Reinforcement Learning (proGENTRL)',
)
