import argparse
import os

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from utils import load_text_file, write_text_file, set_seed, init_wandb_run
import torch
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, choices=[
            "wmt22-testqe-mqm",  # Based on Newstest2022, single MT output, the only part that XCOMET is not trained on
            "wmt22-testmetrics",
            # Based on General test 2022, multiple MT outputs, should not use XCOMET because of data leakage
            # wmt23 en-de is on paragraph level.
        ],
        default="wmt22-testqe-mqm")
    parser.add_argument('--data-root-path', type=str)
    parser.add_argument('--src-lang', type=str)
    parser.add_argument('--tgt-lang', type=str)
    parser.add_argument("--paraphraser_name", type=str, default="Lelon/t5-german-paraphraser-large")
    parser.add_argument("--nr_paraphrases", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="(Default: 1.0). Float (0.0-100.0). The temperature of the sampling operation. 1 means "
                             "regular sampling, 0 means always take the highest score, 100.0 is getting closer to "
                             "uniform probability.")
    parser.add_argument('--wandb-username', type=str, default="None")
    parser.add_argument('--wandb-run-id', type=str, default="None")
    parser.add_argument('--wandb-project', type=str, default="None")

    args = parser.parse_args()
    print(args)

    log_keys = ['dataset', 'src_lang', 'tgt_lang', 'paraphraser_name', 'nr_paraphrases', 'temperature']
    init_wandb_run(args, log_keys)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    pipe = pipeline("text2text-generation", model=args.paraphraser_name, device=device)

    if args.dataset in ["wmt22-testqe-mqm", "wmt22-testmetrics"]:
        if args.dataset == "wmt22-testqe-mqm":
            mt_test_set = "wmt22_news_MT"
        elif args.dataset == "wmt22-testmetrics":
            mt_test_set = "wmt22_general_MT"
        else:
            raise RuntimeError(f"mt_test_set not yet defined for {args.dataset}")

        data_dir = f"{args.data_root_path}/{mt_test_set}/{args.src_lang}-{args.tgt_lang}"
        original_ref_list = ["ref:A"]

        # Remove previous paraphrased files
        for file in glob(f"{data_dir}/*paraphrased*"):
            os.remove(file)

        if f"{args.src_lang}-{args.tgt_lang}" in ["en-de", "zh-en"]:
            original_ref_list.append("ref:B")
        for ref_name in original_ref_list:
            # Load input text
            input_sents = load_text_file(f"{data_dir}/tgt_{ref_name}.{args.tgt_lang}")
            for i in range(0, args.nr_paraphrases):
                print(f"Generating {i}/{args.nr_paraphrases-1} paraphrases for {ref_name}")
                set_seed(i)
                if args.temperature > 0:
                    output = pipe(
                        input_sents, batch_size=args.batch_size, do_sample=True, temperature=args.temperature
                    )
                else:
                    output = pipe(
                        input_sents, batch_size=args.batch_size, do_sample=False
                    )
                output_sents = [x["generated_text"] for x in output]
                write_text_file(output_sents, f"{data_dir}/tgt_{ref_name}_paraphrased{i}.{args.tgt_lang}")


if __name__ == "__main__":
    main()
