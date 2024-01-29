import argparse
from scipy.stats import spearmanr
from utils import load_json, dump_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--gold-ranker", type=str)
    parser.add_argument("--candidate-rankers", type=str, nargs='*')
    parser.add_argument("--eval-ranker-output-path", type=str,
                        help="Path to the output of this script: the score for the candidate ranker")

    args = parser.parse_args()
    print(args)

    gold_ranker_out = load_json(f"{args.output_dir}/{args.gold_ranker}.txt")
    gold_ranker_out = reformat(gold_ranker_out)
    gold_ranker_scores = list(gold_ranker_out.values())
    qe_names = gold_ranker_out.keys()

    eval_candidate_rankers = {}
    for candidate_ranker in args.candidate_rankers:
        candidate_ranker_out = load_json(f"{args.output_dir}/{candidate_ranker}.txt")
        candidate_ranker_out = reformat(candidate_ranker_out)
        candidate_ranker_scores = [candidate_ranker_out[name] for name in qe_names]
        eval_candidate_rankers[candidate_ranker] = spearmanr(candidate_ranker_scores, gold_ranker_scores)

    dump_json(eval_candidate_rankers, args.eval_ranker_output_path)


def reformat(ranker_out):
    # Reformat to the form of {qe_sys_name: score}
    reformatted = {}
    for x in ranker_out:
        reformatted[x['name']] = x['corr_with_gold'][0] if 'corr_with_gold' in x.keys() else x['selection_based_score']
    return reformatted


if __name__ == "__main__":
    main()
