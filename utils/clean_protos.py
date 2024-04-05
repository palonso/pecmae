import glob
from pathlib import Path

protos_file = "xaigenre_proto_sids"
samples_dir = "feats_xai_genre_v1/base/audio/"

with open(protos_file, "r") as f:
    protos = [l.rstrip() for l in f.readlines()]
protos = set(protos)


samples = glob.glob(f"{samples_dir}/*")
print(len(samples), "samples found")

remove = []
for s in samples:
    s = Path(s)
    if s.stem in protos:
        remove.append(s)

print(len(remove), "samples to remove")
for r in remove:
    r.unlink()
