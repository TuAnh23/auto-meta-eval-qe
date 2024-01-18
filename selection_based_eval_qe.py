import argparse
import numpy as np
import json
from utils import process_data, format_for_comet, load_comet_model
from sacrebleu.metrics import BLEU, CHRF, TER


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
    parser.add_argument('--ref-based-system', type=str, choices=['XCOMET-XL', 'wmt22-comet-da', 'BLEU', 'CHRF', 'TER'],
                        default='wmt22-comet-da')
    parser.add_argument('--nr-gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--kept-portion', type=int, default=90, help="in percentage")
    parser.add_argument('--output', type=str, help="Path to file that stores the ranking of the given QE systems")

    args = parser.parse_args()
    print(args)

    src, mt, ref, _ = process_data(args.data_root_path, args.dataset, args.src_lang, args.tgt_lang, args.data_domain)
    qe_data = format_for_comet(src, mt)

    portion_to_keep = args.kept_portion * 0.01
    nr_discarded_samples = int((1 - portion_to_keep) * len(src))

    if 'comet' in args.ref_based_system.lower():
        ref_based_data = format_for_comet(src, mt, ref)
        ref_based_model = load_comet_model(args.comet_model_root_path, args.ref_based_system)

    qe_systems_outputs = []
    for qe_system in args.qe_systems:
        qe_model = load_comet_model(args.comet_model_root_path, qe_system)
        qe_score = qe_model.predict(qe_data, batch_size=args.batch_size, gpus=args.nr_gpus).scores

        # Keep the samples with the highest score by QE
        kept_indices = np.argsort(qe_score)[nr_discarded_samples:]

        if 'comet' in args.ref_based_system.lower():
            # Score the kept samples with ref-based COMET
            system_score = ref_based_model.predict(
                select_from_indices(ref_based_data, kept_indices), batch_size=args.batch_size, gpus=args.nr_gpus
            ).system_score
        else:
            metrics_dict = {
                'BLEU': BLEU(), 'CHRF': CHRF(), 'TER': TER()
            }
            if args.ref_based_system not in metrics_dict.keys():
                raise RuntimeError(f"Ref based system {args.ref_based_system} unknown.")

            system_score = metrics_dict[args.ref_based_system].corpus_score(
                select_from_indices(mt, kept_indices), [select_from_indices(ref, kept_indices)]
            ).score

        qe_systems_outputs.append(
            {
                "name": qe_system,
                "selection_based_score": system_score
            }
        )

    ranked_qes = sort_qe_systems(qe_systems_outputs)

    with open(args.output, 'w') as file:
        json.dump(ranked_qes, file, indent=4)


def select_from_indices(a_list, indices):
    return [x for i, x in enumerate(a_list) if i in indices]


def sort_qe_systems(qe_systems_outputs):
    # First sort by selection-based score
    qe_systems_outputs.sort(key=lambda x: x['selection_based_score'], reverse=True)
    return qe_systems_outputs


if __name__ == '__main__':
    main()
