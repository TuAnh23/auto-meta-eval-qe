import argparse
import os.path
import shutil
from utils import load_text_file, write_text_file, init_wandb_run
from glob import glob
import wandb


def main():
    parser = argparse.ArgumentParser("Gather references for some QE dataset")
    parser.add_argument(
        '--dataset', type=str, choices=[
            "wmt22-testqe-mqm",  # Based on Newstest2022, single MT output, the only part that XCOMET is not trained on
            "wmt22-testmetrics",  # Based on General test 2022, multiple MT outputs, should not use XCOMET because of data leakage
            # wmt23 en-de is on paragraph level.
        ],
        default="wmt22-testqe-mqm")
    parser.add_argument('--data-root-path', type=str)
    parser.add_argument('--src-lang', type=str)
    parser.add_argument('--tgt-lang', type=str)
    parser.add_argument('--nr-refs', type=str, choices=["single", "mul"])
    parser.add_argument('--wandb-username', type=str, default="None")
    parser.add_argument('--wandb-run-id', type=str, default="None")
    parser.add_argument('--wandb-project', type=str, default="None")

    args = parser.parse_args()
    print(args)

    log_keys = []
    init_wandb_run(args, log_keys)

    if args.dataset == "wmt22-testqe-mqm":
        output_dir = f"{args.data_root_path}/wmt22_news_MT/{args.src_lang}-{args.tgt_lang}"

        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        src = load_text_file(
            f"{args.data_root_path}/wmt-qe-2022-data/test_data-gold_labels/task1_mqm/{args.src_lang}-{args.tgt_lang}/test.2022.src"
        )
        write_text_file(src, f"{output_dir}/src.{args.src_lang}")

        # Getting the references from the bigger wmt22 general MT test set
        src_pool = load_text_file(
            f"{args.data_root_path}/wmt22_general_MT/{args.src_lang}-{args.tgt_lang}/src.{args.src_lang}"
        )

        if args.nr_refs == "single":
            ref_files_iter = [
                f"{args.data_root_path}/wmt22_general_MT/{args.src_lang}-{args.tgt_lang}/tgt_ref:A.{args.tgt_lang}"
            ]
        elif args.nr_refs == "mul":
            ref_files_iter = glob(
                f"{args.data_root_path}/wmt22_general_MT/{args.src_lang}-{args.tgt_lang}/tgt_ref*.{args.tgt_lang}"
            )
        else:
            raise RuntimeError(f"nr_refs {args.nr_refs} unknown. Choose `single` or `mul`.")
        count_ref = 0
        for ref_file in ref_files_iter:
            count_ref = count_ref + 1
            ref_pool = load_text_file(ref_file)
            src_ref = {k: v for k, v in zip(src_pool, ref_pool)}
            ref = [src_ref[s] for s in src]
            write_text_file(ref, f"{output_dir}/{ref_file.split('/')[-1]}")

        wandb.log({'nr_original_refs': count_ref})
    elif args.dataset == "wmt22-testmetrics":
        output_dir = f"{args.data_root_path}/wmt22_general_MT/{args.src_lang}-{args.tgt_lang}"
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        original_data_dir = f"{args.data_root_path}/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/"

        shutil.copyfile(f"{original_data_dir}/sources/{args.src_lang}-{args.tgt_lang}.txt",
                        f"{output_dir}/src.{args.src_lang}")
        count_ref = 0
        if args.nr_refs == "single":
            ref_ids = ['A']
        elif args.nr_refs == "mul":
            ref_ids = ['A', 'B', 'C', 'D', 'E']
        else:
            raise RuntimeError(f"nr_refs {args.nr_refs} unknown. Choose `single` or `mul`.")

        for ref_id in ref_ids:
            if os.path.isfile(f"{original_data_dir}/references/{args.src_lang}-{args.tgt_lang}.ref{ref_id}.txt"):
                shutil.copyfile(f"{original_data_dir}/references/{args.src_lang}-{args.tgt_lang}.ref{ref_id}.txt",
                                f"{output_dir}/tgt_ref:{ref_id}.{args.tgt_lang}")
                count_ref = count_ref + 1
        wandb.log({'nr_original_refs': count_ref})


if __name__ == "__main__":
    main()
