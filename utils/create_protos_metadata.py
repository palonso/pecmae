import pandas as pd
from train_protos import get_labelmap


datasets = ["gtzan", "xai_genre", "medley_solos"]

versions = {
    "medley_solos": 502,
    "gtzan": 503,
    "xai_genre": 504,
}
n_protos_per_label = 20

for dataset in datasets:
    data = dict()
    version = versions[dataset]

    for label in get_labelmap(dataset).keys():
        for n in range(n_protos_per_label):
            filename = f"v{version}_{label}_n{n}_gs1.npy"

            data[filename] = label

    df = pd.DataFrame.from_dict(data, orient="index")
    df.reset_index(inplace=True)

    out_filename = f"prototypes_encoded/{dataset}_gt.tsv"
    df.to_csv(out_filename, header=False, sep="\t", columns=[0, "index"], index=False)
