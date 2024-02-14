import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from utils import load_text_file, write_text_file, set_seed
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=["wmt22-testqe-mqm"],
                        default="wmt22-testqe-mqm",
                        help="WMT22 QE test is based on Newstest2022, "
                             "which is the only part that XCOMET is not trained on")
    parser.add_argument('--data-root-path', type=str)
    parser.add_argument('--src-lang', type=str)
    parser.add_argument('--tgt-lang', type=str)
    parser.add_argument("--model_name", type=str, default="Lelon/t5-german-paraphraser-large")
    parser.add_argument("--nr_paraphrases", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="(Default: 1.0). Float (0.0-100.0). The temperature of the sampling operation. 1 means "
                             "regular sampling, 0 means always take the highest score, 100.0 is getting closer to "
                             "uniform probability.")

    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    pipe = pipeline("text2text-generation", model=args.model_name, device=device)

    if args.dataset == "wmt22-testqe-mqm":
        data_dir = f"{args.data_root_path}/wmt22_news_MT/{args.src_lang}-{args.tgt_lang}"
        original_ref_list = ["ref:A"]
        if f"{args.src_lang}-{args.tgt_lang}" in ["en-de", "zh-en"]:
            original_ref_list.append("ref:B")
        for ref_name in original_ref_list:
            # Load input text
            input_sents = load_text_file(f"{data_dir}/tgt_{ref_name}.{args.tgt_lang}")
            for i in range(0, args.nr_paraphrases):
                print(f"Generating {i}/{args.nr_paraphrases-1} paraphrases for {ref_name}")
                set_seed(i)
                output = pipe(
                    input_sents, batch_size=args.batch_size, do_sample=(args.temperature > 0), temperature=args.temperature
                )
                output_sents = [x["generated_text"] for x in output]
                write_text_file(output_sents, f"{data_dir}/tgt_{ref_name}_paraphrased{i}.{args.tgt_lang}")


if __name__ == "__main__":
    main()
