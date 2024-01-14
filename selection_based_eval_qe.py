import argparse
import numpy as np
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
    parser.add_argument('--kept-portion', type=int, default=90, help="in percentage")
    parser.add_argument('--output', type=str, help="Path to file that stores the ranking of the given QE systems")

    args = parser.parse_args()
    print(args)

    src, mt, ref, _ = process_data(args.data_root_path, args.dataset, args.src_lang, args.tgt_lang, args.data_domain)
    qe_data = format_for_comet(src, mt)
    ref_based_data = format_for_comet(src, mt, ref)

    portion_to_keep = args.kept_portion * 0.01
    nr_discarded_samples = int((1 - portion_to_keep) * len(src))

    ref_based_model = load_comet_model(args.comet_model_root_path, args.ref_based_system)

    meta_eval_qes = {}
    for qe_system in args.qe_systems:
        qe_model = load_comet_model(args.comet_model_root_path, qe_system)
        qe_score = qe_model.predict(qe_data, batch_size=args.batch_size, gpus=args.nr_gpus).scores

        # Keep the samples with the highest score by QE
        kept_indices = np.argsort(qe_score)[nr_discarded_samples:]
        kept_samples = [x for i, x in enumerate(ref_based_data) if i in kept_indices]

        # Score the kept samples with ref-based COMET
        system_score = ref_based_model.predict(
            kept_samples, batch_size=args.batch_size, gpus=args.nr_gpus
        ).system_score

        meta_eval_qes[qe_system] = system_score

    sorted_meta_eval_qes = dict(sorted(meta_eval_qes.items(), key=lambda item: item[1], reverse=True))

    with open(args.output, 'w') as file:
        json.dump(sorted_meta_eval_qes, file, indent=4)


if __name__ == '__main__':
    main()
