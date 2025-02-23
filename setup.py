from setuptools import setup, find_packages

setup(
    name="gwbird",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",   
        "healpy",
        "astropy",
        "mpmath"
    ],
    author="Ilaria Caporali, Angelo Ricciardone",
    description="A package for gravitational wave background analysis",
    url="https://github.com/ilariacaporali/GWBird/tree/master",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
)