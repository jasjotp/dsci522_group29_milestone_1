import click
from pathlib import Path

@click.command()
@click.argument("figure_path", type=click.Path(exists=True))

def test(figure_path):
    path = Path(figure_path)
    files = [file for file in path.iterdir() if file.is_file()]
    assert len(files) == 3

if __name__ == "__main__":
    test()