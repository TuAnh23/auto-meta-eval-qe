import argparse
import numpy as np
from utils import process_data, format_for_comet, load_comet_model, dump_json, add_noise, load_text_file
from sacrebleu.metrics import BLEU, CHRF, TER
import os


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
    parser.add_argument('--comet-qe-systems', type=str, nargs='*')
    parser.add_argument('--synthetic-qe-systems', type=str, default='None', choices=['None', '20-human-with-noise'])
    parser.add_argument('--qe-official-submissions', type=str, default='None', choices=['None', 'wmt22'])
    parser.add_argument('--ref-based-system', type=str, choices=['XCOMET-XL', 'wmt22-comet-da', 'BLEU', 'CHRF', 'TER'],
                        default='wmt22-comet-da')
    parser.add_argument('--nr-gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--kept-portion', type=int, default=90, help="in percentage")
    parser.add_argument('--output', type=str, help="Path to file that stores the ranking of the given QE systems")

    args = parser.parse_args()
    print(args)

    src, mt, ref, human_scores = process_data(args.data_root_path, args.dataset, args.src_lang, args.tgt_lang, args.data_domain)
    qe_data = format_for_comet(src, mt)

    portion_to_keep = args.kept_portion * 0.01
    nr_discarded_samples = int((1 - portion_to_keep) * len(src))

    if 'comet' in args.ref_based_system.lower():
        ref_based_data = format_for_comet(src, mt, ref)
        ref_based_model = load_comet_model(args.comet_model_root_path, args.ref_based_system)
    else:
        ref_based_data = None
        ref_based_model = None

    qe_systems_outputs = []
    for qe_system in args.comet_qe_systems:
        qe_model = load_comet_model(args.comet_model_root_path, qe_system)
        qe_score = qe_model.predict(qe_data, batch_size=args.batch_size, gpus=args.nr_gpus).scores

        gather_score(mt, nr_discarded_samples, qe_score, qe_system, qe_systems_outputs, ref, ref_based_data,
                    ref_based_model, args.ref_based_system, args.batch_size, args.nr_gpus)
    if args.synthetic_qe_systems == '20-human-with-noise':
        for i in range(1, 21):
            synthetic_qe_score = add_noise(human_scores, noise_level=0.01 * i)
            gather_score(
                mt, nr_discarded_samples, synthetic_qe_score, f'human_with_noise_{i}', qe_systems_outputs,
                ref, ref_based_data, ref_based_model, args.ref_based_system, args.batch_size, args.nr_gpus
            )

    if args.qe_official_submissions == 'wmt22':
        assert args.dataset == f"wmt22-testqe-mqm"
        submissions_root_path = f"{args.data_root_path}/wmt-qe-2022-data/official_submissions/task1_mqm/{args.src_lang}-{args.tgt_lang}"
        participants = os.listdir(submissions_root_path)
        participants = [p for p in participants if os.path.isdir(f"{submissions_root_path}/{p}")]

        for participant in participants:
            qe_system = participant.split(' - ')[0]
            qe_score = load_text_file(f"{submissions_root_path}/{participant}/predictions.txt")[3:]
            qe_score = np.array([float(x.split('\t')[-1]) for x in qe_score])
            gather_score(
                mt, nr_discarded_samples, qe_score, qe_system, qe_systems_outputs,
                ref, ref_based_data, ref_based_model, args.ref_based_system, args.batch_size, args.nr_gpus
            )

    ranked_qes = sort_qe_systems(qe_systems_outputs)

    dump_json(ranked_qes, args.output)


def gather_score(
        mt, nr_discarded_samples, qe_score, qe_system, qe_systems_outputs,
        ref, ref_based_data, ref_based_model, ref_based_system, batch_size, nr_gpus
):
    """
    Calculate score of QE systems and append results to `qe_systems_outputs`
    """
    # Keep the samples with the highest score by QE
    kept_indices = np.argsort(qe_score)[nr_discarded_samples:]
    if 'comet' in ref_based_system.lower():
        # Score the kept samples with ref-based COMET
        system_score = ref_based_model.predict(
            select_from_indices(ref_based_data, kept_indices), batch_size=batch_size, gpus=nr_gpus
        ).system_score
    else:
        metrics_dict = {
            'BLEU': BLEU(), 'CHRF': CHRF(), 'TER': TER()
        }
        if ref_based_system not in metrics_dict.keys():
            raise RuntimeError(f"Ref based system {ref_based_system} unknown.")

        system_score = metrics_dict[ref_based_system].corpus_score(
            select_from_indices(mt, kept_indices), [select_from_indices(ref, kept_indices)]
        ).score
    qe_systems_outputs.append(
        {
            "name": qe_system,
            "selection_based_score": system_score
        }
    )


def select_from_indices(a_list, indices):
    return [x for i, x in enumerate(a_list) if i in indices]


def sort_qe_systems(qe_systems_outputs):
    # First sort by selection-based score
    qe_systems_outputs.sort(key=lambda x: x['selection_based_score'], reverse=True)
    return qe_systems_outputs


if __name__ == '__main__':
    main()
