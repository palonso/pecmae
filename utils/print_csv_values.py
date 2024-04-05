import yaml
from pathlib import Path
from argparse import ArgumentParser

keys = [
    "dataset",
    "n_protos_per_label",
    "time_summarization",
    "use_discriminator",
]


parser = ArgumentParser()

parser.add_argument("--filename", type=Path)
parser.add_argument("--version", type=str)

args = parser.parse_args()

if args.version:
    filename = f"tb_logs/zinemanet/version_{args.version}/hparams.yaml"
else:
    filename = args.filename

with open(filename, "r") as f:
    data = yaml.unsafe_load(f)

output = [str(data[key]) for key in keys]
print(",".join(output))
