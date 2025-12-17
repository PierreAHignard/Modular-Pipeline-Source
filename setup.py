from setuptools import setup, find_packages

setup(
    name='Transfer Learning Pipeline',
    version='0.1.0',
    description='Transfer Learning Pipeline source code',
    author='PA Hignard',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'tqdm>=4.61.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scikit-learn>=0.24.0',
        'pandas>=1.3.0',
        'Pillow>=8.3.0',
    ],
)
