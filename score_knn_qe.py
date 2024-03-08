"""
This module do meta-score the KNN-QE system using ref-based MetricsX-23, as our analysis shows that this is the most
robust ref-based metric to score ref-free metric
"""

import argparse
from utils import load_text_file, filter_by_domain
from metricx23_predict import calculate_ref_based_score
import numpy as np
from glob import glob
from scipy.stats import spearmanr, pearsonr, kendalltau
import wandb
import json


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

    wandb.init(project='knn-qe', entity='tuanh', name=f"{args.mt_model}/{args.dataset}/{args.datastore}_{args.emb_layer}")
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

    print(f"Start evaluation using ref-based metric")
    ref_based_scores = []
    for ref in refs:
        scores = calculate_ref_based_score(mt, ref)
        # Invert so that the higher the score, the better the translation
        ref_based_scores.append(-np.array(scores))
    ref_based_score = np.mean(np.array(ref_based_scores), axis=0)

    eval_qe_dict = {}
    for qe_score_path in glob(f"{knn_qe_path}/qeScore_*.txt"):
        qe_method = qe_score_path.split('/')[-1].replace('qeScore_', '').replace('.txt', '')
        qe_scores = [float(x) for x in load_text_file(qe_score_path)]

        if args.domain != 'None' and args.domain != 'all':
            ref_based_score_filtered = filter_by_domain(
                data_domain=args.domain, domain_list=domain_list, sent_list=ref_based_score
            )
            qe_scores_filtered = filter_by_domain(
                data_domain=args.domain, domain_list=domain_list, sent_list=qe_scores
            )
        else:
            ref_based_score_filtered = ref_based_score
            qe_scores_filtered = qe_scores

        eval_qe = spearmanr(ref_based_score_filtered, qe_scores_filtered)
        eval_qe_pearson = pearsonr(ref_based_score_filtered, qe_scores_filtered)
        eval_qe_kendalltau = kendalltau(ref_based_score_filtered, qe_scores_filtered)

        out_dict = {
            'eval_qe': eval_qe,
            'eval_qe_pearson': eval_qe_pearson,
            'eval_qe_kendalltau': eval_qe_kendalltau
        }

        eval_qe_dict[qe_method] = out_dict

        wandb.log({qe_method: eval_qe[0]})

    with open(f"{knn_qe_path}/eval_qe.json", 'w') as f:
        json.dump(eval_qe_dict, f)


if __name__ == "__main__":
    main()
