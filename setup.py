from setuptools import setup, find_packages

setup(
    name="gwbird",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={"gwbird": ["psd/*.txt", "NANOGrav/*.par", "EPTA/*.txt"]},
    install_requires=[
        "numpy>=2.0.0",
        "scipy",
        "matplotlib",   
        "healpy",
        "astropy",
        "mpmath", 
        "glob2",
        "pint-pulsar",

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