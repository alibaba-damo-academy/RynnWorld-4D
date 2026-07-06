from setuptools import setup, find_packages

setup(
    name="rynnworld4d",
    version="0.1.0",
    packages=find_packages(include=["core", "core.*"]),
    python_requires=">=3.9",
)
