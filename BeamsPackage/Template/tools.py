from importlib import resources
from pathlib import Path

def copy_example_to_cwd(filename: str = "example.json"):
    src = resources.files("BeamsPackage.Template").joinpath("example.json")
    dest = Path.cwd() / filename
    dest.write_bytes(src.read_bytes())
    print(f"Copied example.json to {dest.resolve()}")
