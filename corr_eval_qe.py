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
from scipy.stats import spearmanr, pearsonr
from utils import \
    load_data, format_for_comet, load_comet_model, WilliamsTest, \
    dump_json, add_noise, load_text_file, format_for_sent_level_sacrebleu, init_wandb_run
import os
import numpy as np
from sacrebleu.metrics import BLEU, CHRF, TER
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["metrics_wmt-mqm-human-evaluation", "wmt22-testqe-mqm"],
                        default="wmt22-testqe-mqm",
                        help="WMT22 QE test is based on Newstest2022, "
                             "which is the only part that XCOMET is not trained on")
    parser.add_argument('--data-root-path', type=str)
    parser.add_argument('--data-domain',
                        type=str, default='all',
                        choices=['all', 'conversation', 'social', 'news', 'ecommerce'])
    parser.add_argument('--comet-model-root-path', type=str)
    parser.add_argument('--src-lang', type=str)
    parser.add_argument('--tgt-lang', type=str)
    parser.add_argument('--comet-qe-systems', type=str, nargs='*', default=[])
    parser.add_argument('--synthetic-qe-systems', type=str, default='None', choices=['None', '20-human-with-noise'])
    parser.add_argument('--qe-official-submissions', type=str, default='None', choices=['None', 'wmt22'])
    parser.add_argument('--ref-based-system', type=str, choices=['XCOMET-XL', 'wmt22-comet-da', 'BLEU', 'CHRF', 'TER'],
                        default='wmt22-comet-da')
    parser.add_argument('--nr-gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output', type=str, help="Path to file that stores the ranking of the given QE systems")
    parser.add_argument('--ground-truth', type=str, choices=["human", "ref_based_metrics"])
    parser.add_argument('--wandb-username', type=str, default="None")
    parser.add_argument('--wandb-run-id', type=str, default="None")
    parser.add_argument('--wandb-project', type=str, default="None")

    args = parser.parse_args()
    print(args)

    log_keys = [
        'dataset', 'data_domain', 'src_lang', 'tgt_lang', 'ref_based_system', 'synthetic_qe_systems',
        'qe_official_submissions', 'comet_qe_systems'
    ]
    init_wandb_run(args, log_keys)

    src, mt, refs, human_scores = load_data(
        args.data_root_path, args.dataset, args.src_lang, args.tgt_lang, args.data_domain
    )
    qe_data = format_for_comet(src, mt)

    wandb.log({'total_nr_refs': len(refs)})

    if args.ground_truth == 'ref_based_metrics':
        if 'comet' in args.ref_based_system.lower():
            ref_based_data_s = [format_for_comet(src, mt, ref) for ref in refs]
            ref_based_model = load_comet_model(args.comet_model_root_path, args.ref_based_system)
            ref_based_scores = [
                ref_based_model.predict(ref_based_data, batch_size=args.batch_size, gpus=args.nr_gpus).scores
                for ref_based_data in ref_based_data_s
            ]
            ref_based_score = np.mean(np.array(ref_based_scores), axis=0)
        elif args.ref_based_system in ['BLEU', 'CHRF', 'TER']:
            metrics_dict = {
                'BLEU': BLEU(effective_order=True),
                'CHRF': CHRF(),
                'TER': TER()
            }
            src, mt, formatted_refs = format_for_sent_level_sacrebleu(src, mt, refs)
            ref_based_score = np.array(
                [
                    metrics_dict[args.ref_based_system].sentence_score(mt[i], formatted_refs[i]).score
                    for i in range(len(mt))
                ]
            )
            if args.ref_based_system == 'TER':
                # Invert so that the higher the score, the beter the translation
                ref_based_score = - ref_based_score
        else:
            raise RuntimeError(f"Unknown ref_based_system {args.ref_based_system}.")

        wandb.log(
            {
                "ref_based_corr_w_human": spearmanr(ref_based_score, human_scores)[0],
                "ref_based_corr_w_human_pearson": pearsonr(ref_based_score, human_scores)[0],
            }
        )

    else:
        ref_based_score = None

    qe_systems_outputs = []

    for qe_system in args.comet_qe_systems:
        qe_model = load_comet_model(args.comet_model_root_path, qe_system)
        qe_score = qe_model.predict(qe_data, batch_size=args.batch_size, gpus=args.nr_gpus).scores
        gather_corr(human_scores, qe_score, qe_system, qe_systems_outputs, ref_based_score, args.ground_truth)

    if args.synthetic_qe_systems == '20-human-with-noise':
        for i in range(1, 21):
            synthetic_qe_score = add_noise(human_scores, noise_level=0.01 * i)
            gather_corr(human_scores=human_scores,
                        qe_score=synthetic_qe_score,
                        qe_system=f'human_with_noise_{i}',
                        qe_systems_outputs=qe_systems_outputs,
                        ref_based_score=ref_based_score,
                        ground_truth=args.ground_truth)

    if args.qe_official_submissions == 'wmt22':
        assert args.dataset == "wmt22-testqe-mqm"
        submissions_root_path = f"{args.data_root_path}/wmt-qe-2022-data/official_submissions/task1_mqm/{args.src_lang}-{args.tgt_lang}"
        participants = os.listdir(submissions_root_path)
        participants = [p for p in participants if os.path.isdir(f"{submissions_root_path}/{p}")]

        for participant in participants:
            qe_system = participant.split(' - ')[0]
            qe_score = load_text_file(f"{submissions_root_path}/{participant}/predictions.txt")[3:]
            qe_score = np.array([float(x.split('\t')[-1]) for x in qe_score])
            gather_corr(human_scores=human_scores,
                        qe_score=qe_score,
                        qe_system=qe_system,
                        qe_systems_outputs=qe_systems_outputs,
                        ref_based_score=ref_based_score,
                        ground_truth=args.ground_truth)

    ranked_qes = sort_qe_systems(qe_systems_outputs)

    # Remove the sentence-level scores from the output for readability
    for item in ranked_qes:
        item.pop('sent_level_scores')

    dump_json(ranked_qes, args.output)

    wandb.log(
        {
            f"rank_qe_by_{args.ground_truth}": '/'.join([x['name'] for x in ranked_qes])
        }
    )


def gather_corr(human_scores, qe_score, qe_system, qe_systems_outputs, ref_based_score, ground_truth):
    """
    Calculate correlation of QE scores to gold scores (human/reference based) and append results to `qe_systems_outputs`
    """
    if ground_truth == "human":
        corr_score = spearmanr(qe_score, human_scores)
        corr_score_pearson = pearsonr(qe_score, human_scores)
    elif ground_truth == "ref_based_metrics":
        corr_score = spearmanr(qe_score, ref_based_score)
        corr_score_pearson = pearsonr(qe_score, ref_based_score)
    else:
        raise RuntimeError(f"Unknown ground_truth {ground_truth}.")
    qe_systems_outputs.append(
        {
            'name': qe_system,
            'sent_level_scores': qe_score,
            'corr_with_gold': corr_score,
            'corr_with_gold_pearson': corr_score_pearson
        }
    )


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
