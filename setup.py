"""AEROTICA - Atmospheric Kinetic Energy Mapping and Aero-Elastic Resilience Framework"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aerotica-ake",
    version="1.0.0",
    author="Samir Baladi",
    author_email="gitdeeper@gmail.com",
    description="AEROTICA: Atmospheric Kinetic Energy Mapping and Aero-Elastic Resilience Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/gitdeeper07/aerotica",
    project_urls={
        "Bug Tracker": "https://gitlab.com/gitdeeper07/aerotica/-/issues",
        "Documentation": "https://aerotica.netlify.app/documentation",
        "Source Code": "https://gitlab.com/gitdeeper07/aerotica",
        "DOI": "https://doi.org/10.14293/AEROTICA.2025.001",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "xarray>=2022.3.0",
        "netCDF4>=1.5.8",
        "torch>=2.3.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
        "tqdm>=4.64.0",
    ],
    keywords="atmospheric physics kinetic energy turbulence wind shear neural networks navier-stokes renewable-energy fluid-dynamics",
    license="MIT",
)
