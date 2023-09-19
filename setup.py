import setuptools

def readme():
    try:
        with open('README.md') as f:
            return f.read()
    except IOError:
        return ''

setuptools.setup(
    name = "pyDRTtools",
    version = "0.2.8.45",
    author = "ciuccislab",
    author_email = "amaradesa@connect.ust.hk",
    description = "pyDRTtools: A Python-based DRTtools to Deconvolve the Distribution of Relaxation Times from Electrochemical Impedance Spectroscopy Data",
    long_description = readme(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/ciuccislab/pyDRTtools",
    project_urls = {
        "Source Code": "https://github.com/ciuccislab/pyDRTtools",
        "Bug Tracker": "https://github.com/ciuccislab/pyDRTtools/issues",
    },
    entry_points={
        "console_scripts": [
            "pyDRTtoolsui=pyDRTtools.pyDRTtools_GUI:main",
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
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    #packages=['pyDRTtools'], pyqt==5.15.7
    
    install_requires = [
        'cvxopt',
        'cvxpy',
        'scipy',
        'numpy',
        'scikit-learn',
        'pyqt5-tools~=5.15',
        #'matplotlib',
        #'pandas',       
    ],
    #entry_points={
    #"console_scripts": [
        #"pyDRTtoolsgui=pyDRTtools.pyDRTtools_GUI:main",
    #],
    #},
    include_package_data=True,
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3"
)
