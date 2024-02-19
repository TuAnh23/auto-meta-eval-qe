import argparse
import os.path
import shutil
from utils import load_text_file, write_text_file, init_wandb_run
from glob import glob
import wandb


def main():
    parser = argparse.ArgumentParser("Gather references for some QE dataset")
    parser.add_argument('--dataset', type=str, choices=["wmt22-testqe-mqm"],
                        default="wmt22-testqe-mqm",
                        help="WMT22 QE test is based on Newstest2022, "
                             "which is the only part that XCOMET is not trained on")
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


if __name__ == "__main__":
    main()
