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
from utils import process_data, format_for_comet, load_comet_model, WilliamsTest, dump_json


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

    qe_systems_outputs = []

    for qe_system in args.qe_systems:
        qe_model = load_comet_model(args.comet_model_root_path, qe_system)
        qe_score = qe_model.predict(qe_data, batch_size=args.batch_size, gpus=args.nr_gpus).scores
        if args.ground_truth == "human":
            corr_score = spearmanr(qe_score, human_scores)
        elif args.ground_truth == "ref_based_comet":
            corr_score = spearmanr(qe_score, ref_based_score)
        else:
            raise RuntimeError(f"Unknown ground_truth {args.ground_truth}.")
        qe_systems_outputs.append(
            {
                'name': qe_system,
                'sent_level_scores': qe_score,
                'corr_with_gold': corr_score
            }
        )

    ranked_qes = sort_qe_systems(qe_systems_outputs)

    # Remove the sentence-level scores from the output for readability
    for item in ranked_qes:
        item.pop('sent_level_scores')

    dump_json(ranked_qes, args.output)


def sort_qe_systems(qe_systems_outputs):
    # First sort by correlation with the ground truth
    qe_systems_outputs.sort(key=lambda x: x['corr_with_gold'][0], reverse=True)

    # Then use Williams Test to see which ordering is significant
    for qe_system_i in range(0, len(qe_systems_outputs)):
        significantly_worse = []
        for worse_qe_system_i in range(qe_system_i + 1, len(qe_systems_outputs)):
            p_value = WilliamsTest(
                qe_systems_outputs[qe_system_i]['corr_with_gold'][0],
                qe_systems_outputs[worse_qe_system_i]['corr_with_gold'][0],
                spearmanr(qe_systems_outputs[qe_system_i]['sent_level_scores'], qe_systems_outputs[worse_qe_system_i]['sent_level_scores'])[0],
                len(qe_systems_outputs[qe_system_i]['sent_level_scores'])
            )
            if p_value < 0.05:
                significantly_worse.append(qe_systems_outputs[worse_qe_system_i]['name'])
        qe_systems_outputs[qe_system_i]['significantly_worse'] = significantly_worse

    return qe_systems_outputs


if __name__ == '__main__':
    main()
