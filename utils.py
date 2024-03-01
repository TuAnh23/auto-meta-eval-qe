import pandas as pd
import numpy as np
import random
import scipy
import json
from glob import glob


def load_data(data_root_path, dataset, src_lang, tgt_lang, data_domain):
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
        return df['src'].to_list(), {"-": df['mt'].to_list()}, {"-": df['ref'].to_list()}, {"-": score_nomralized}
    elif dataset == f"wmt22-testqe-mqm":
        src = load_text_file(
            f"{data_root_path}/wmt-qe-2022-data/test_data-gold_labels/task1_mqm/{src_lang}-{tgt_lang}/test.2022.src"
        )
        mt = {
            '-': load_text_file(
                f"{data_root_path}/wmt-qe-2022-data/test_data-gold_labels/task1_mqm/{src_lang}-{tgt_lang}/test.2022.mt"
            )
        }

        # Getting the references from the wmt22 news MT test set
        refs = {}
        for ref_file in glob(f"{data_root_path}/wmt22_news_MT/{src_lang}-{tgt_lang}/tgt_ref*.{tgt_lang}"):
            refs[ref_file.split('/')[-1].split('.')[0].replace('tgt_', '')] = load_text_file(ref_file)

        mqm_z_score = load_text_file(
            f"{data_root_path}/wmt-qe-2022-data/test_data-gold_labels/task1_mqm/{src_lang}-{tgt_lang}/test.2022.{src_lang}-{tgt_lang}.mqm_z_score"
        )
        mqm_z_score = {'-': [float(x) for x in mqm_z_score]}

        return src, mt, refs, mqm_z_score

    elif dataset == f"wmt22-testmetrics":
        if data_domain == 'all':
            mt_test_set = "wmt22_general_MT"
        else:
            mt_test_set = f"wmt22_general_MT_{data_domain}"
        src = load_text_file(
            f"{data_root_path}/{mt_test_set}/{src_lang}-{tgt_lang}/src.{src_lang}"
        )

        refs = {}
        for ref_file in glob(f"{data_root_path}/{mt_test_set}/{src_lang}-{tgt_lang}/tgt_ref*.{tgt_lang}"):
            refs[ref_file.split('/')[-1].split('.')[0].replace('tgt_', '')] = load_text_file(ref_file)

        original_data_dir = f"{data_root_path}/.mt-metrics-eval/mt-metrics-eval-v2/wmt22"
        domain_list = load_text_file(f"{original_data_dir}/sources/{src_lang}-{tgt_lang}_domain.txt")

        mt = {}
        for mt_file in glob(
                f"{original_data_dir}/system-outputs/{src_lang}-{tgt_lang}/*.txt"
        ):
            mt[mt_file.split('/')[-1].replace(".txt", "")] = filter_by_domain(
                data_domain, domain_list, load_text_file(mt_file)
            )

        mqm_score = pd.read_csv(
            f"{original_data_dir}/human-scores/{src_lang}-"
            f"{tgt_lang}.mqm.seg.score",
            names=["MT_system", "score"], header=None, delimiter="\t"
        ).groupby(
            by="MT_system"
        )['score'].apply(list).apply(lambda x: filter_by_domain(data_domain, domain_list, x)).to_dict()

        # Remove the scoring on human translations
        ref_keys = [x for x in mqm_score.keys() if "ref" in x]
        for ref_key in ref_keys:
            mqm_score.pop(ref_key)
            mt.pop(ref_key)

        # Remove the scoring from outlier systems
        outlier_keys = []
        if f"{src_lang}-{tgt_lang}" == "en-de":
            outlier_keys.append('M2M100_1.2B-B4')
        else:
            raise RuntimeWarning(f"Check for outlier system for language pair {src_lang}-{tgt_lang}.")

        for outlier_key in outlier_keys:
            mqm_score.pop(outlier_key)
            mt.pop(outlier_key)

        return src, mt, refs, mqm_score

    else:
        raise RuntimeError(f"Dataset {dataset} unknown")


def format_for_comet(src, mt, ref=None):
    if ref is not None:
        return [{'src': x, 'mt': y, 'ref': z} for x, y, z in zip(src, mt, ref)]
    else:
        return [{'src': x, 'mt': y} for x, y in zip(src, mt)]


def format_for_sent_level_sacrebleu(src, mt, refs):
    """
    :param src: [src_sent1, src_sent2, ...]
    :param mt: [hyp_sent1, hyp_sent2, ...]
    :param refs: [[ref1_sent1, ref1_sent2, ...], [ref2_sent1, ref2_sent2, ...], ...]
    :return: src, mt, formatted_refs, where formatted_refs = [[ref1_sent1, ref2_sent1, ...], [ref1_sent2, ref2_sent2, ...], ...]
    """
    assert len(src) == len(mt)
    nr_sents = len(src)
    formatted_refs = []
    for sent_i in range(nr_sents):
        formatted_refs.append([refs[ref_i][sent_i] for ref_i in range(len(refs))])
    return src, mt, formatted_refs


def load_comet_model(comet_model_root_path, qe_system):
    from comet import download_model, load_from_checkpoint

    if qe_system in [
        "wmt22-cometkiwi-da", "wmt20-comet-qe-da", "wmt22-comet-da", "wmt23-cometkiwi-da-xl", "XCOMET-XL", "unite-mup"
    ]:
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
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")


def set_seed(seed=0):
    """Set the random seed for torch.
    Args:
        seed (int, optional): random seed. Default 0
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # If CUDA is not available, this is silently ignored.
    torch.cuda.manual_seed_all(seed)


def init_wandb_run(args, log_keys):
    import wandb

    args_dict = vars(args)
    config_dict = {
        k: '/'.join(args_dict[k]) if isinstance(args_dict[k], list) else args_dict[k] for k in log_keys
    }

    wandb.init(project=args.wandb_project, entity=args.wandb_username, id=args.wandb_run_id, resume="allow")
    for k, v in config_dict.items():
        wandb.config[k] = v


def calculate_pairwise_ranking_accuracy(pred_scores, gold_scores):
    pred_rank_binary = ranking_scores_to_binary(pred_scores)
    gold_rank_binary = ranking_scores_to_binary(gold_scores)
    compare = np.equal(pred_rank_binary, gold_rank_binary).astype(int)
    return sum(compare) / len(compare) if len(compare) > 0 else None


def ranking_scores_to_binary(scores):
    binary_out = []
    for i in range(len(scores)):
        for j in range(i+1, len(scores)):
            binary_out.append(int(scores[i] > scores[j]))
    return np.array(binary_out)


def filter_by_domain(data_domain, domain_list, sent_list):
    if data_domain != 'all':
        sent_list = [s for (s, d) in zip(sent_list, domain_list) if d == data_domain]
    return sent_list