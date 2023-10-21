import click
from . import GUI

@click.command()
def main():
    """Launch the GUI."""
    GUI.launch_gui()

if __name__ == "__main__":
    main()


