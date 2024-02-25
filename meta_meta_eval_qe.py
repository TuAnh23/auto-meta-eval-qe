import argparse
from scipy.stats import spearmanr
from utils import load_json, dump_json, init_wandb_run, calculate_pairwise_ranking_accuracy
import wandb
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--gold-ranker", type=str)
    parser.add_argument("--candidate-ranker", type=str)
    parser.add_argument("--eval-ranker-output-path", type=str,
                        help="Path to the output of this script: the score for the candidate ranker")
    parser.add_argument('--wandb-username', type=str, default="None")
    parser.add_argument('--wandb-run-id', type=str, default="None")
    parser.add_argument('--wandb-project', type=str, default="None")

    args = parser.parse_args()
    print(args)

    log_keys = []
    init_wandb_run(args, log_keys)

    eval_candidate_ranker = {}
    for file in glob(f"{args.output_dir}/{args.candidate_ranker}*.txt"):
        out_suffix = file.replace(f"{args.output_dir}/{args.candidate_ranker}", "").replace(".txt", "")

        gold_ranker_out = load_json(f"{args.output_dir}/{args.gold_ranker}{out_suffix}.txt")
        gold_ranker_out = reformat(gold_ranker_out)
        gold_ranker_scores = list(gold_ranker_out.values())
        qe_names = gold_ranker_out.keys()

        if out_suffix == "":
            mt_system = "all"
        else:
            mt_system = out_suffix[1:]
        candidate_ranker_out = load_json(file)
        candidate_ranker_out = reformat(candidate_ranker_out)
        candidate_ranker_scores = [candidate_ranker_out[name] for name in qe_names]
        eval_candidate_ranker[mt_system] = spearmanr(candidate_ranker_scores, gold_ranker_scores)
        pairwise_ranking_accuracy = calculate_pairwise_ranking_accuracy(candidate_ranker_scores, gold_ranker_scores)

        wandb.log(
            {
                f"{args.candidate_ranker}_corr_of_corr{out_suffix}": eval_candidate_ranker[mt_system][0],
                f"{args.candidate_ranker}_pairwise_ranking_accuracy{out_suffix}": pairwise_ranking_accuracy,
            }
        )

    dump_json(eval_candidate_ranker, args.eval_ranker_output_path)


def reformat(ranker_out):
    # Reformat to the form of {qe_sys_name: score}
    reformatted = {}
    for x in ranker_out:
        reformatted[x['name']] = x['corr_with_gold'][0] if 'corr_with_gold' in x.keys() else x['selection_based_score']
    return reformatted


if __name__ == "__main__":
    main()
