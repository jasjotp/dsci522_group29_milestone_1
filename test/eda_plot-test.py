from pathlib import Path

def test(figure_path):
    path = Path('/figures')
    files = [file for file in path.iterdir() if file.is_file()]
    assert len(files) == 3