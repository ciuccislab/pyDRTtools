import setuptools
from os.path import exists, join

def readme():
    try:
        with open('README.md') as f:
            return f.read()
    except IOError:
        return ''
entry_points={
        "console_scripts": [
            "pyDRTtoolsui=pyDRTtools.pyDRTtools_GUI:main",
        ],
    }

dependencies = [
    "cvxopt~=1.3",  # cvxopt optimizer
    "cvxpy>=1.3",
    "requests~=2.28",  # Used to check package status on PyPI.
    "scipy==1.10.0",
    "numpy==1.24.1",
    "scikit-learn>=1.2.1",
    "PyQt5==5.15.9",
    "matplotlib==3.7.3",
    "pandas==1.5.3",
]

dev_dependencies = [
    "setuptools~=65.5",  # For Setuptools.
    "build~=0.10",
    "flake8~=6.0",
]

optional_dependencies = {  
    "kvxopt": "kvxopt~=1.3",  
    "dev": dev_dependencies,
}

if __name__ == "__main__":
    with open("requirements.txt", "w") as fp:
        fp.write("\n".join(dependencies))
    with open("dev-requirements.txt", "w") as fp:
        fp.write("\n".join(dev_dependencies))


setuptools.setup(
    name = "pyDRTtools",
    version = "0.2.8.62",
    author = "ciuccislab",
    author_email = "amaradesa@connect.ust.hk",
    description = "pyDRTtools: A Python-based DRTtools to Deconvolve the Distribution of Relaxation Times from Electrochemical Impedance Spectroscopy Data",
    long_description = readme(),
    long_description_content_type = "text/markdown",
    ###
    packages = setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    ###
    url = "https://github.com/ciuccislab/pyDRTtools",
    project_urls = {
        "Source Code": "https://github.com/ciuccislab/pyDRTtools",
        "Bug Tracker": "https://github.com/ciuccislab/pyDRTtools/issues",
    },
    entry_points=entry_points,
    install_requires=dependencies,
    extras_require=optional_dependencies,
    python_requires = ">=3.4",
    
    classifiers = [
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        #"Operating System :: OS Independent",
          
    ],
)
