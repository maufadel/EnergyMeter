from setuptools import setup, find_packages

setup(
    name="energymeter",                   # Package name
    version="1.0.0",                     # Version
    author="Mauricio Fadel Argerich",
    author_email="maufadel@icloud.com",
    description="A library to measure energy consumption of applications",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/maufadel/EnergyMeter",  # Optional: GitHub link
    packages=find_packages(),            # Automatically find packages in the project
    install_requires=[
        "pynvml",                         # Dependencies from PyPI
        "pyRAPL",
        "numpy",
        "pandas",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
    python_requires='>=3.9',             # Minimum Python version
)
