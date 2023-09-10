import setuptools

def readme():
    try:
        with open('README.md') as f:
            return f.read()
    except IOError:
        return ''

setuptools.setup(
    name = "pyDRTtools",
    version = "0.26",
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
            "launch_pydrttool=src.pyDRTtools_GUI:main",
        ],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # List your dependencies here
        "cvxopt ~= 1.3.2",  
        "cvxpy ~= 1.3.2",
        "matplotlib ~= 3.7.2",
        "pandas ~= 2.0.3",
        "numpy ~= 1.25.2",
        "scipy ~= 1.11.2",
        "scikit-learn ~= 1.3.0",
        "PyQt5 ~= 5.15.9",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.6"
)
