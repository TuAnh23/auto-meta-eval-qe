"""
This module do meta-score the KNN-QE system using ref-based MetricsX-23, as our analysis shows that this is the most
robust ref-based metric to score ref-free metric
"""

import argparse
import os.path

from utils import load_text_file, filter_by_domain, format_for_comet, load_comet_model, write_text_file
from metricx23_predict import calculate_ref_based_score
import numpy as np
from glob import glob
from scipy.stats import spearmanr, pearsonr, kendalltau
import wandb
import json
os.environ["WANDB_SILENT"]="true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['wmt22general.en', 'ted_tst14.en'])
    parser.add_argument('--mt_model', type=str, choices=['deltalm_base_ft_ted', 'ted_new'])
    parser.add_argument('--emb_layer', type=int, choices=[0, 1, 2, 3, 4, 5, 6], default=6)
    parser.add_argument('--datastore', type=str, choices=['ted_new'])
    parser.add_argument('--domain', type=str, default='None',
                        choices=['None', 'all', 'conversation', 'ecommerce', 'news', 'social'])
    parser.add_argument('--data_root_path', type=str)
    parser.add_argument('--knn_qe_root_path', type=str)

    args = parser.parse_args()
    print(args)

    wandb.init(project='knn-qe', entity='tuanh',
               name=f"{args.mt_model}/{args.dataset}/{args.datastore}_{args.emb_layer}",
               sync_tensorboard=False)
    args_dict = vars(args)
    for k, v in args_dict.items():
        if 'root_path' not in k:
            wandb.config[k] = v

    if args.domain != 'None' and 'wmt22' not in args.dataset:
        raise RuntimeError('Only wmt22 dataset is splitted into different domains.')

    knn_qe_path = f'{args.knn_qe_root_path}/{args.mt_model}/{args.dataset}/{args.datastore}_{args.emb_layer}'
    if args.dataset == 'ted_tst14.en':
        src = load_text_file(f'{args.data_root_path}/ted.en-de/tst.de-en.en')
        refs = [load_text_file(f'{args.data_root_path}/ted.en-de/tst.de-en.de')]
        domain_list = None
    elif args.dataset == 'wmt22general.en':
        src = load_text_file(f"{args.data_root_path}/wmt22_general_MT/en-de/src.en")
        domain_list = load_text_file(f"{args.data_root_path}/wmt22_general_MT/en-de/domain.txt")
        refs = [
            load_text_file(f"{args.data_root_path}/wmt22_general_MT/en-de/tgt_ref:A.de"),
            load_text_file(f"{args.data_root_path}/wmt22_general_MT/en-de/tgt_ref:B.de"),
        ]
    else:
        raise RuntimeError(f"Unknown dataset {args.dataset}")

    mt = load_text_file(f'{knn_qe_path}/mt.txt')

    ref_based_scores = []
    for ref in refs:
        metricx_score_path = f'{args.knn_qe_root_path}/{args.mt_model}/{args.dataset}/metricx23.score'
        if os.path.isfile(metricx_score_path):
            scores = load_text_file(metricx_score_path)
            scores = [float(x) for x in scores]
        else:
            print(f"Start evaluation using ref-based metric")
            scores = calculate_ref_based_score(mt, ref)
            write_text_file(scores, metricx_score_path)

        # Invert so that the higher the score, the better the translation
        ref_based_scores.append(-np.array(scores))

    ref_based_score = np.mean(np.array(ref_based_scores), axis=0)
    ref_based_score = filter_by_domain(
        data_domain=args.domain, domain_list=domain_list, sent_list=ref_based_score
    )

    eval_qe_dict = {}
    for qe_score_path in glob(f"{knn_qe_path}/qeScore_*.txt"):
        qe_method = qe_score_path.split('/')[-1].replace('qeScore_', '').replace('.txt', '')
        qe_scores = [float(x) for x in load_text_file(qe_score_path)]
        meta_eval_qe_and_log(args.domain, domain_list, eval_qe_dict, qe_method, qe_scores, ref_based_score)

    # Also eval ensembled qe score
    ensemble_combinations = [
        ['knn_distance_inv_k1', 'sent_similarity_k1'],
        ['knn_distance_inv_k1', 'sent_similarity_k1', 'nr_out_equal_knn_k10', 'nr_diff_knn_inv_k10'],
        ['prob', 'knn_distance_inv_k1', 'sent_similarity_k1'],
        ['prob', 'knn_distance_inv_k1', 'sent_similarity_k1', 'nr_out_equal_knn_k10', 'nr_diff_knn_inv_k10'],

    ]
    for combination in ensemble_combinations:
        qe_scores = np.mean(
            np.array([eval_qe_dict[qe_method]['qe_scores'] for qe_method in combination]),
            axis=0
        ).tolist()
        meta_eval_qe_and_log(args.domain, domain_list, eval_qe_dict, f'ensemble_{combination}', qe_scores,
                             ref_based_score)

    # Also meta-eval COMET supervised QE to use as an upper baseline
    qe_data = format_for_comet(src, mt)
    cometqe_score_path = f'{args.knn_qe_root_path}/{args.mt_model}/{args.dataset}/cometqe.score'
    if os.path.isfile(cometqe_score_path):
        qe_scores = load_text_file(cometqe_score_path)
        qe_scores = [float(x) for x in qe_scores]
    else:
        qe_model = load_comet_model(None, 'wmt22-cometkiwi-da')
        qe_scores = qe_model.predict(qe_data, batch_size=16, gpus=1).scores
        write_text_file(qe_scores, cometqe_score_path)
    meta_eval_qe_and_log(args.domain, domain_list, eval_qe_dict, 'wmt22-cometkiwi-da', qe_scores, ref_based_score)

    # Also log MT-system-level score by the ref-based metric
    mt_sys_score = np.mean(ref_based_score)
    wandb.log({'mt_sys_score_ref_based': mt_sys_score})

    with open(f"{knn_qe_path}/eval_qe.json", 'w') as f:
        json.dump(eval_qe_dict, f)


def meta_eval_qe_and_log(domain, domain_list, eval_qe_dict, qe_method, qe_scores, ref_based_score):
    if len(qe_scores) != len(ref_based_score):
        qe_scores = filter_by_domain(
            data_domain=domain, domain_list=domain_list, sent_list=qe_scores
        )

    eval_qe = spearmanr(ref_based_score, qe_scores)
    eval_qe_pearson = pearsonr(ref_based_score, qe_scores)
    eval_qe_kendalltau = kendalltau(ref_based_score, qe_scores)
    out_dict = {
        'eval_qe': eval_qe,
        'eval_qe_pearson': eval_qe_pearson,
        'eval_qe_kendalltau': eval_qe_kendalltau,
        'qe_scores': qe_scores
    }
    eval_qe_dict[qe_method] = out_dict
    wandb.log({qe_method: eval_qe[0]})


if __name__ == "__main__":
    main()
