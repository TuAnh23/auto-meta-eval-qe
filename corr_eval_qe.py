"""
Perform QE on benchmark dataset with human-labeled scores on the sentence-level translation.
Calculate correlation scores, i.e., how good is the QE systems
Rank the QE systems according to the correlation score

Supervised COMET QE models are trained on WMT17 --> WMT20 data.

So we do experiments on 2021 DA data onwards
https://huggingface.co/datasets/RicardoRei/wmt-da-human-evaluation
No en-->de in 2022 DA data, and 2021 DA data is told to be errorneous

So we use MQM data 2021 onwards
https://huggingface.co/datasets/RicardoRei/wmt-mqm-human-evaluation
"""
import argparse
from scipy.stats import spearmanr
import json
from utils import process_data, format_for_comet, load_comet_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["metrics_wmt-mqm-human-evaluation"])
    parser.add_argument('--data-root-path', type=str)
    parser.add_argument('--data-domain',
                        type=str, default='all',
                        choices=['all', 'conversation', 'social', 'news', 'ecommerce'])
    parser.add_argument('--comet-model-root-path', type=str)
    parser.add_argument('--src-lang', type=str)
    parser.add_argument('--tgt-lang', type=str)
    parser.add_argument('--qe-systems', type=str, nargs='*')
    parser.add_argument('--ref-based-system', type=str, choices=['XCOMET-XL', 'wmt22-comet-da'],
                        default='wmt22-comet-da')
    parser.add_argument('--nr-gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=str, help="Path to file that stores the ranking of the given QE systems")
    parser.add_argument('--ground-truth', type=str, choices=["human", "ref_based_comet"])

    args = parser.parse_args()
    print(args)

    src, mt, ref, human_scores = process_data(
        args.data_root_path, args.dataset, args.src_lang, args.tgt_lang, args.data_domain

    )
    qe_data = format_for_comet(src, mt)
    ref_based_data = format_for_comet(src, mt, ref)

    ref_based_model = load_comet_model(args.comet_model_root_path, args.ref_based_system)
    ref_based_score = ref_based_model.predict(ref_based_data, batch_size=args.batch_size, gpus=args.nr_gpus).scores

    meta_eval_qes = {}
    for qe_system in args.qe_systems:
        qe_model = load_comet_model(args.comet_model_root_path, qe_system)

        qe_score = qe_model.predict(qe_data, batch_size=args.batch_size, gpus=args.nr_gpus).scores

        if args.ground_truth == "human":
            corr_score = spearmanr(qe_score, human_scores)
        elif args.ground_truth == "ref_based_comet":
            corr_score = spearmanr(qe_score, ref_based_score)
        else:
            raise RuntimeError(f"Unknown ground_truth {args.ground_truth}.")
        meta_eval_qes[qe_system] = corr_score

    sorted_meta_eval_qes = dict(sorted(meta_eval_qes.items(), key=lambda item: item[1], reverse=True))

    with open(args.output, 'w') as file:
        json.dump(sorted_meta_eval_qes, file, indent=4)


if __name__ == '__main__':
    main()
