from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

exec(open("neurgoo/__version__.py").read())

setup(
    name="neurgoo",
    version=__version__,
    description="A naive implementation of modular neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NISH1001/neurgoo",
    author="Nish",
    author_email="nishanpantha@gmail.com, np0069@uah.edu",
    license="MIT",
    python_requires=">=3.8",
    packages=[
        "neurgoo",
        "neurgoo.layers",
    ],
    install_requires=required,
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Topic :: Artificial Neural Network",
    ],
    zip_safe=False,
)
