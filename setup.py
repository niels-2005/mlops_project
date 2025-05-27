from setuptools import find_packages, setup

setup(
    name="my-mlops-project",
    version="0.1.0",
    description="MLOps Project with Backend and Frontend",
    author="Niels Scholz",
    author_email="niels@niels.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
