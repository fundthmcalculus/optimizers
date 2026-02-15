from setuptools import setup, find_packages

setup(
    name="optimizers",
    version="0.1.0",
    description="A collection of optimization algorithms (GA, PSO, ACO, GD, etc.)",
    author="fundthmcalculus",
    packages=find_packages(),
    install_requires=["numpy", "kmodes", 'scipy'],
    python_requires=">=3.7",
    url="https://github.com/fundthmcalculus/optimizers",
    license="MIT",
)
