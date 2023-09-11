import setuptools

def readme():
    try:
        with open('README.md') as f:
            return f.read()
    except IOError:
        return ''

setuptools.setup(
    name = "pyDRTtools",
    version = "0.27",
    author = "ciuccislab",
    author_email = "amaradesa@connect.ust.hk",
    description = "pyDRTtools: A Python-based DRTtools to Deconvolve the Distribution of Relaxation Times from Electrochemical Impedance Spectroscopy Data",
    long_description = readme(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/ciuccislab/pyDRTtools",
    project_urls = {
        "Bug Tracker": "https://github.com/ciuccislab/pyDRTtools",
    },
    entry_points={
        "console_scripts": [
            "pyDRTtoolsui=src.pyDRTtools_GUI:pyDRTtools_GUI",
        ],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # List your dependencies here
        "cvxopt >= 1.2.0",  
        "cvxpy >= 1.1.0",
        "matplotlib >= 3.7.0",
        "pandas >= 1.5.3",
        "numpy >= 1.20.1",
        "scipy >= 1.10.1",
        "scikit-learn >= 1.1.2",
        "PyQt5 >= 12.11.0",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3"
)
