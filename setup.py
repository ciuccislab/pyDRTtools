import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "pyDRTtools",
    version = "1.0.1",
    author = "ciuccislab",
    author_email = "amaradesa@connect.ust.hk",
    description = "python-based version of DRTtools",
    long_description_content_type = "pyDRTtools: A python-based DRTtools to Deconvolve the Distribution of Relaxation Times from Electrochemical Impedance Spectroscopy Data",
    url = "https://github.com/ciuccislab/pyDRTtools",
    project_urls = {
        "Bug Tracker": "https://github.com/ciuccislab/pyDRTtools",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.6"
)