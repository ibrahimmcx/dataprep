from setuptools import setup, find_packages
import os

# Read the contents of your README file
with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="ibrahim-dataprep",
    version="0.1.2",
    author="ibrahimmcx",
    author_email="ibrahimmcx@github.com",
    description="Autonomous Data Science Assistant for Instant Data Preparation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ibrahimmcx/dataprep",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "typer",
        "rich",
        "python-dateutil",
        "openpyxl",
    ],
    entry_points={
        "console_scripts": [
            "dataprep=dataprep.cli:app",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.8',
)
