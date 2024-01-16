__authors__ = 'Francesco Ciucci, Adeleke Maradesa'

__date__ = '16th Jan., 2024'


import click
from . import GUI

@click.command()
def main():
    """Launch the GUI."""
    GUI.launch_gui()

if __name__ == "__main__":
    main()


