import argparse
import numpy as np
import json
from utils import process_data, format_for_comet, load_comet_model, WilliamsTest


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

    qe_systems_outputs = []
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

        qe_systems_outputs.append(
            {
                "name": qe_system,
                "sys": qe_model,
                "selection_based_score": system_score,
                "kept_samples": [x for i, x in enumerate(qe_data) if i in kept_indices]
            }
        )

    ranked_qes = sort_qe_systems(qe_systems_outputs, len(ref_based_data), args.batch_size, args.nr_gpus)

    # Remove the unnecessary info from the output for readability
    remove = ["sys", "kept_samples"]
    for item in ranked_qes:
        for info in remove:
            item.pop(info)

    with open(args.output, 'w') as file:
        json.dump(ranked_qes, file, indent=4)


def sort_qe_systems(qe_systems_outputs, n_samples, batch_size, nr_gpus):
    # First sort by selection-based score
    qe_systems_outputs.sort(key=lambda x: x['selection_based_score'], reverse=True)

    # Then use Williams Test to see which ordering is significant
    for qe_system_i in range(0, len(qe_systems_outputs)):
        significantly_worse = []
        for worse_qe_system_i in range(qe_system_i + 1, len(qe_systems_outputs)):
            sys_vs_gold = qe_systems_outputs[qe_system_i]['selection_based_score']
            worsesys_vs_gold = qe_systems_outputs[worse_qe_system_i]['selection_based_score']
            worsesys_vs_sys = qe_systems_outputs[qe_system_i]['sys'].predict(
                qe_systems_outputs[worse_qe_system_i]['kept_samples'],
                batch_size=batch_size, gpus=nr_gpus
            ).system_score

            p_value = WilliamsTest(
                rescaling_score(sys_vs_gold),
                rescaling_score(worsesys_vs_gold),
                rescaling_score(worsesys_vs_sys),
                n_samples
            )
            if p_value < 0.05:
                significantly_worse.append(qe_systems_outputs[worse_qe_system_i]['name'])
        qe_systems_outputs[qe_system_i]['significantly_worse'] = significantly_worse

    return qe_systems_outputs


def rescaling_score(score):
    """
    Rescaling a score in the [0,1] range to [-1,1] range to fit Williams Test
    """
    assert 0 <= score <= 1
    return 2 * score - 1


if __name__ == '__main__':
    main()
