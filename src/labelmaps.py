gtzan_id2label = {
    0: "blu",
    1: "cla",
    2: "cou",
    3: "dis",
    4: "hip",
    5: "jaz",
    6: "met",
    7: "pop",
    8: "reg",
    9: "roc",
}

gtzan_label2id = {
    "blu": 0,
    "cla": 1,
    "cou": 2,
    "dis": 3,
    "hip": 4,
    "jaz": 5,
    "met": 6,
    "pop": 7,
    "reg": 8,
    "roc": 9,
}

nsynth_id2label = {
    0: "bass",
    1: "brass",
    2: "flute",
    3: "guitar",
    4: "keyboard",
    5: "mallet",
    6: "organ",
    7: "reed",
    8: "string",
    9: "vocal",
}

nsynth_label2id = {
    "bass": 0,
    "brass": 1,
    "flute": 2,
    "guitar": 3,
    "keyboard": 4,
    "mallet": 5,
    "organ": 6,
    "reed": 7,
    "string": 8,
    "vocal": 9,
}

xaigenre_id2label = {
    0: "afrobeat",
    1: "ambient",
    2: "bolero",
    3: "bop",
    4: "bossa nova",
    5: "contemporary jazz",
    6: "cumbia",
    7: "disco",
    8: "doo wop",
    9: "dub",
    10: "electro",
    11: "europop",
    12: "funk",
    13: "gospel",
    14: "heavy metal",
    15: "house",
    16: "indie rock",
    17: "punk",
    18: "techno",
    19: "trance",
    20: "salsa",
    21: "samba",
    22: "soul",
    23: "swing",
}

xaigenre_label2id = {
    "afrobeat": 0,
    "ambient": 1,
    "bolero": 2,
    "bop": 3,
    "bossa nova": 4,
    "contemporary jazz": 5,
    "cumbia": 6,
    "disco": 7,
    "doo wop": 8,
    "dub": 9,
    "electro": 10,
    "europop": 11,
    "funk": 12,
    "gospel": 13,
    "heavy metal": 14,
    "house": 15,
    "indie rock": 16,
    "punk": 17,
    "techno": 18,
    "trance": 19,
    "salsa": 20,
    "samba": 21,
    "soul": 22,
    "swing": 23,
}

medley_solos_id2label = {
    0: "clarinet",
    1: "distorted electric guitar",
    2: "female singer",
    3: "flute",
    4: "piano",
    5: "tenor saxophone",
    6: "trumpet",
    7: "violin",
}


medley_solos_label2id = {
    "clarinet": 0,
    "distorted electric guitar": 1,
    "female singer": 2,
    "flute": 3,
    "piano": 4,
    "tenor saxophone": 5,
    "trumpet": 6,
    "violin": 7,
}


def get_labelmap(dataset: str):
    if dataset == "gtzan":
        return gtzan_label2id
    elif dataset == "nsynth":
        return nsynth_label2id
    elif dataset == "xai_genre":
        return xaigenre_label2id
    elif dataset == "medley_solos":
        return medley_solos_label2id
    else:
        raise ValueError(f"dataset {dataset} not supported")
