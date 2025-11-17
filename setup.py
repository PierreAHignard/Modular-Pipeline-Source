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
        # 'torch>=1.9.0',
        # 'torchvision>=0.10.0',
    ],
)
