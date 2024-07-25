__authors__ = 'Francesco Ciucci, Adeleke Maradesa'

__date__ = '29th June., 2024'

import os
# print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(f"Initializing pyDRTtools from {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")

from . import basics
from . import BHT
from . import cli
from . import GUI
from . import HMC
from . import layout
from.  import nearest_PD
from . import parameter_selection
from . import peak_analysis
from . import runs



modules = [
    "basics", 
    "BHT", 
    "cli", 
    "GUI", 
    "HMC",
    "layout", 
    "nearest_PD",
    "parameter_selection",
    "peak_analysis", 
    "runs"
]

for module in modules:
    try:
        __import__(f"pyDRTtools.{module}", fromlist=[''])
        print(f"Imported {module}")
    except ImportError as e:
        print(f"Failed to import {module}: {e}")



__all__ = [
    'basics',
    'GUI',
    'layout',
    'parameter_selection',
    'peak_analysis',
    'HMC',
    'runs',
    'nearest_PD'
]

print(f"Contents of pyDRTtools package: {__all__}")