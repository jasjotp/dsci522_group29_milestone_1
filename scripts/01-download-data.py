from pathlib import Path
import click 
import pandas as pd

@click.command()
@click.argument("input_url")      
@click.argument("output_csv")  

def main(input_url, output_csv):
    """
    Downloads data from the URL and outputs it to a specified CSV path 
    """
    df = pd.read_csv(input_url).dropna(subset = ["relationship_duration"])

    out_path = Path(output_csv)

    out_path.parent.mkdir(parents = True, exist_ok = True)

    df.to_csv(out_path, index = False)

    # return the root path of the file
    click.echo(f"Saved data to {out_path.resolve()}")

if __name__ == "__main__":
    main()
