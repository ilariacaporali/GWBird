from pathlib import Path

from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="gwbird",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={"gwbird": ["psd/*.txt", "NANOGrav/*.par"]},
    install_requires=[
        "numpy>=2.0.0",
        "scipy<1.15",
        "matplotlib",
        "healpy",
        "astropy",
        "mpmath",
        "glob2",
        "pint-pulsar",

    ],
    author="Ilaria Caporali, Angelo Ricciardone",
    description="A package for gravitational wave background characterization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ilariacaporali/GWBird/tree/master",
    project_urls={
        "Source": "https://github.com/ilariacaporali/GWBird",
        "Paper": "https://arxiv.org/abs/2507.15791",
        "Bug Tracker": "https://github.com/ilariacaporali/GWBird/issues",
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.10",
)