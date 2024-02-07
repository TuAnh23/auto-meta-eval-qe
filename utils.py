import pandas as pd
from comet import download_model, load_from_checkpoint
import numpy as np
import scipy
import json


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
    elif dataset == f"wmt22-testqe-mqm":
        src = load_text_file(
            f"{data_root_path}/wmt-qe-2022-data/test_data-gold_labels/task1_mqm/{src_lang}-{tgt_lang}/test.2022.src"
        )
        mt = load_text_file(
            f"{data_root_path}/wmt-qe-2022-data/test_data-gold_labels/task1_mqm/{src_lang}-{tgt_lang}/test.2022.mt"
        )

        # Getting the references from the bigger wmt22 general MT test set
        src_pool = load_text_file(
            f"{data_root_path}/wmt22_general_MT/src.{src_lang}"
        )
        ref_pool = load_text_file(
            f"{data_root_path}/wmt22_general_MT/tgt.{tgt_lang}"
        )
        src_ref = {k: v for k, v in zip(src_pool, ref_pool)}

        ref = [src_ref[s] for s in src]

        mqm_z_score = load_text_file(
            f"{data_root_path}/wmt-qe-2022-data/test_data-gold_labels/task1_mqm/{src_lang}-{tgt_lang}/test.2022.{src_lang}-{tgt_lang}.mqm_z_score"
        )
        mqm_z_score = [float(x) for x in mqm_z_score]

        return src, mt, ref, mqm_z_score
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


def WilliamsTest(r12, r13, r23, n, one_sided=True):
    """
    Function taken from https://github.com/google-research/mt-metrics-eval/blob/cb65e66184d2ade99ae3f4400ef4cad896ab48d8/mt_metrics_eval/stats.py#L866

    :param r12: correlation between metrics A and human
    :param r13: correlation between metrics B and human
    :param r23: correlation between the two metrics
    :param n: sample size (number of sentences)
    :param one_sided:
    :return: Williams test p-value for given correlations. This function works with arbitrary correlation functions, but
    the interpretation of results other than non-averaged Pearson with complete entries is not clear.
    """
    k = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r23 * r13

    if k < 0:
        raise RuntimeError("Value k should not be negative.")

    rbar = ((r12 + r13) / 2)
    tnum = (r12 - r13) * np.sqrt((n - 1) * (1 + r23))
    tden = np.sqrt(2 * (n - 1) / (n - 3) * k + rbar**2 * (1 - r23)**3)
    p = scipy.stats.t.sf(np.abs(tnum / tden), n - 3)
    return p if one_sided else 2 * p


def load_json(filename):
    with open(filename, 'r') as f:
        out = json.load(f)
    return out


def dump_json(obj, filename):
    with open(filename, 'w') as file:
        json.dump(obj, file, indent=4)


def add_noise(human_scores, noise_level=0.1):
    noise = np.random.normal(loc=0, scale=noise_level, size=human_scores.shape)
    return human_scores + noise


def load_text_file(file_path):
    """
    Load text file into a list, each item is a line of the file
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines


def write_text_file(lines, file_path):
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")
