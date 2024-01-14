import pandas as pd
from comet import download_model, load_from_checkpoint


def process_data(data_root_path, dataset, src_lang, tgt_lang, data_domain):
    if dataset == "metrics_wmt-mqm-human-evaluation":
        df = pd.read_csv(f"{data_root_path}/wmt-mqm-human-evaluation")
        # Filter out the data
        if data_domain == 'all':
            df = df[
                    (df["year"] == 2022) &
                    (df["lp"] == f"{src_lang}-{tgt_lang}")
                    ]
        else:
            df = df[
                (df["year"] == 2022) &
                (df["lp"] == f"{src_lang}-{tgt_lang}") &
                (df["domain"] == data_domain)
                ]
        # According to https://www.statmt.org/wmt22/metrics/index.html
        # "MQM data for en-de and zh-en was mostly annotated by Google and it ranges -25 to 0 where 0 is a perfect
        # translation and -25 is the worse possible score"
        # So normalize
        min = -25
        max = 0
        score_nomralized = (df['score'].to_numpy() - min) / (max - min)
        if f"{src_lang}-{tgt_lang}" not in ["en-de", "en-zh"]:
            raise RuntimeError(f"Not sure if MQM normalizing is correct for language pair {src_lang}-{tgt_lang}")
        return df['src'].to_list(), df['mt'].to_list(), df['ref'].to_list(), score_nomralized
    else:
        raise RuntimeError(f"Dataset {dataset} unknown")


def format_for_comet(src, mt, ref=None):
    if ref is not None:
        return [{'src': x, 'mt': y, 'ref': z} for x, y, z in zip(src, mt, ref)]
    else:
        return [{'src': x, 'mt': y} for x, y in zip(src, mt)]


def load_comet_model(comet_model_root_path, qe_system):
    if qe_system in ["wmt22-cometkiwi-da", "wmt20-comet-qe-da", "wmt22-comet-da", "wmt23-cometkiwi-da-xl", "XCOMET-XL"]:
        model_path = download_model(f"Unbabel/{qe_system}")
    else:
        model_path = f"{comet_model_root_path}/{qe_system}/checkpoints/model.ckpt"
    model = load_from_checkpoint(model_path)
    return model
