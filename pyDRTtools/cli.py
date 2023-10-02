import click
from pyDRTtools.GUI import launch_gui

@click.command()
def launch():
    """Launch the GUI."""
    launch_gui()

if __name__ == "__main__":
    launch()

