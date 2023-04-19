import click
import json

# Description: Convert a TSV file to a JSONL file

@click.command()
@click.option("-k1")
@click.option("-k2")
@click.option("--sep", "-s", default="\t")
def sidebyside(k1, k2, sep):

    stdin = click.get_text_stream("stdin")

    def json_line(line1, line2):
        return json.dumps({k1: line1, k2: line2})

    for line in stdin:

        line1, line2 = line.strip().split(sep)
        click.echo(json_line(line1, line2), nl=True)
    
if __name__ == "__main__":
    sidebyside()
