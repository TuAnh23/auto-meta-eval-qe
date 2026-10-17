"""
Perform QE on benchmark dataset with human-labeled scores on the sentence-level translation.
Calculate correlation scores, i.e., how good is the QE systems
Rank the QE systems according to the correlation score

Supervised COMET QE models are trained on WMT17 --> WMT20 data.

So we do experiments on 2021 DA data onwards
https://huggingface.co/datasets/RicardoRei/wmt-da-human-evaluation
No en-->de in 2022 DA data, and 2021 DA data is told to be errorneous

So we use MQM data 2021 onwards
https://huggingface.co/datasets/RicardoRei/wmt-mqm-human-evaluation
"""
import argparse
from scipy.stats import spearmanr, pearsonr, kendalltau
from utils import \
    load_data, format_for_comet, load_comet_model, WilliamsTest, write_text_file, \
    dump_json, add_noise, load_text_file, format_for_sent_level_sacrebleu, init_wandb_run, filter_by_domain, ScoreCache, \
    WMT24_DIR, load_seg_scores, wmt24_segment_mask
import os
import numpy as np
import pandas as pd
import wandb
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        choices=[
            "metrics_wmt-mqm-human-evaluation",
            "wmt22-testqe-mqm",  # Based on Newstest2022, single MT output, the only part that XCOMET is not trained on. Official test set for WMT22 QE
            "wmt22-testmetrics",  # Based on General test 2022, multiple MT outputs, should not use XCOMET because of data leakage. Official test set for WMT22 Metrics
            # wmt23 en-de is on paragraph level.
            "wmt24-testmetrics-esa",  # General test 2024, ESA scores, 1 ref; XCOMET and MetricX-23 are not trained on it
        ],
        default="wmt22-testqe-mqm"
    )
    parser.add_argument('--data-root-path', type=str)
    parser.add_argument('--data-domain',
                        type=str, default='all',
                        choices=['all', 'conversation', 'social', 'news', 'ecommerce',  # WMT22
                                 'literary', 'speech',  # + news, social: WMT24 en-*
                                 'education', 'official', 'personal', 'voice'])  # + news: WMT24 cs-uk
    parser.add_argument('--comet-model-root-path', type=str)
    parser.add_argument('--src-lang', type=str)
    parser.add_argument('--tgt-lang', type=str)
    parser.add_argument('--comet-qe-systems', type=str, nargs='*', default=[])
    parser.add_argument('--synthetic-qe-systems', type=str, default='None', choices=['None', '20-human-with-noise'])
    parser.add_argument('--qe-official-submissions', type=str, default='None', choices=['None', 'wmt22', 'wmt22-qe-metrics', 'wmt24-qe-metrics'])
    parser.add_argument(
        '--ref-based-system',
        type=str,
        choices=[
            'XCOMET-XL', 'wmt22-comet-da', 'BLEU', 'CHRF', 'TER', 'bert-score',
            'bleurt', 'unite-mup', 'metricx_xxl_MQM_2020', 'metricx_xl_MQM_2020',
            'metricx-23-xl-v2p0'
        ],
        default='wmt22-comet-da',
        help="metricx_xxl_MQM_2020 and metricx_xl_MQM_2020 models not available, so you scores submissions,"
             "which is only available for dataset=wmt22-testmetrics"
    )
    parser.add_argument('--nr-gpus', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--output-dir', type=str,
                        help="Directory to save the file that stores the ranking of the given QE systems")
    parser.add_argument('--ground-truth', type=str, choices=["human", "ref_based_metrics"])
    parser.add_argument('--ref-based-cache-dir', type=str, default=None,
                        help="Directory to cache segment scores of model-based ref-based metrics (COMET-style, BERTScore, "
                             "BLEURT, MetricX-23), shared across runs. Scores are memoized within a run either way.")
    parser.add_argument('--wandb-username', type=str, default="None")
    parser.add_argument('--wandb-run-id', type=str, default="None")
    parser.add_argument('--wandb-project', type=str, default="None")

    args = parser.parse_args()
    print(args)

    log_keys = [
        'dataset', 'data_domain', 'src_lang', 'tgt_lang', 'ref_based_system', 'synthetic_qe_systems',
        'qe_official_submissions', 'comet_qe_systems'
    ]
    init_wandb_run(args, log_keys)

    ref_based_cache_dir = None if args.ref_based_cache_dir is None \
        else f"{args.ref_based_cache_dir}/{args.ref_based_system}"
    score_cache = ScoreCache(ref_based_cache_dir, shard_name=args.wandb_run_id)

    src, mt, refs, human_scores = load_data(
        args.data_root_path, args.dataset, args.src_lang, args.tgt_lang, args.data_domain
    )

    print("MT systems to be scored: ", human_scores.keys())

    mt_systems_order = list(mt.keys())
    src_all = []
    refs_all = {k: [] for k in refs.keys()}
    mt_all = []
    human_scores_all = []
    for mt_system in mt_systems_order:
        src_all.extend(src)
        for k in refs.keys():
            refs_all[k].extend(refs[k])
        mt_all.extend(mt[mt_system])
        human_scores_all.append(human_scores[mt_system])
    human_scores_all = np.concatenate(human_scores_all)

    if args.ground_truth == 'ref_based_metrics' and args.ref_based_system in MODEL_BASED_METRICS:
        # Score all segments in one go (one model load); the per-MT-system passes below then read the cache
        for ref in refs_all.values():
            score_cache.score(format_for_comet(src_all, mt_all, ref), lambda samples: score_model_based(args, samples),
                              args.ref_based_system)
    elif args.ground_truth == 'ref_based_metrics' and args.ref_based_system in ['BLEU', 'CHRF', 'TER'] \
            and len(refs_all) == 1:
        metric = sacrebleu_metric(args)
        score_cache.score(format_for_comet(src_all, mt_all, list(refs_all.values())[0]),
                          lambda samples: score_sacrebleu(metric, samples), args.ref_based_system)

    for mt_system in mt_systems_order:
        do_rank_qes(args, human_scores[mt_system], mt[mt_system], refs, src, score_cache, mt_system=mt_system)

    do_rank_qes(
        args, human_scores_all, mt_all, refs_all, src_all, score_cache, mt_system="all",
        mt_systems_order=mt_systems_order
    )


def do_rank_qes(args, human_scores, mt, refs, src, score_cache, mt_system="all", mt_systems_order=None):
    if mt_system == "all":
        out_suffix = ""
    else:
        out_suffix = f"_{mt_system}"
    qe_data = format_for_comet(src, mt)
    if mt_system == "all":
        write_text_file(human_scores, f"{args.output_dir}/human_sent_scores.txt")
    if mt_system == "all":
        wandb.log({'total_nr_refs': len(refs)})
    if args.ground_truth == 'ref_based_metrics':
        if args.ref_based_system in ['BLEU', 'CHRF', 'TER']:
            metric = sacrebleu_metric(args)
            if len(refs) == 1:
                # Cached: TER with character-level tokenization (zh, ja) is slow on long segments
                ref_based_score = np.array(score_cache.score(
                    format_for_comet(src, mt, list(refs.values())[0]),
                    lambda samples: score_sacrebleu(metric, samples), args.ref_based_system
                ))
            else:
                # Multiple references are scored jointly, not averaged per reference: not cached
                src, mt, formatted_refs = format_for_sent_level_sacrebleu(src, mt, list(refs.values()))
                ref_based_score = np.array(
                    [metric.sentence_score(mt[i], formatted_refs[i]).score for i in range(len(mt))]
                )
            if args.ref_based_system == 'TER':
                # Invert so that the higher the score, the beter the translation
                ref_based_score = - ref_based_score
        elif args.ref_based_system in MODEL_BASED_METRICS:
            ref_based_scores = []
            for ref in refs.values():
                # Only score segments not already scored in this run or in a cached previous run
                ref_based_scores.append(score_cache.score(
                    format_for_comet(src, mt, ref), lambda samples: score_model_based(args, samples),
                    args.ref_based_system
                ))
            ref_based_score = np.mean(np.array(ref_based_scores), axis=0)
            if args.ref_based_system == 'metricx-23-xl-v2p0':
                # Invert so that the higher the score, the beter the translation
                ref_based_score = - ref_based_score
        elif args.ref_based_system in ['metricx_xxl_MQM_2020', 'metricx_xl_MQM_2020']:
            assert args.dataset == "wmt22-testmetrics"
            submissions_root_path = f"{args.data_root_path}/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/metric-scores/" \
                                    f"{args.src_lang}-{args.tgt_lang}"
            domain_list = load_text_file(
                f"{args.data_root_path}/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/sources/{args.src_lang}-{args.tgt_lang}_domain.txt"
            )

            ref_based_scores = []
            for ref_name, ref_sents in refs.items():
                if "paraphrased" in ref_name:
                    raise RuntimeError(
                        f"Using {args.ref_based_system} submission scores, which is only available for original refs."
                    )
                ref_name = ref_name.replace(':', '')
                metric_file = f"{submissions_root_path}/{args.ref_based_system}-{ref_name}.seg.score"
                scores = pd.read_csv(
                    metric_file, names=["MT_system", "score"], header=None, delimiter="\t"
                ).groupby(
                    by="MT_system"
                )['score'].apply(list).apply(lambda x: filter_by_domain(args.data_domain, domain_list, x)).to_dict()
                if mt_system == "all":
                    scores = np.concatenate([scores[x] for x in mt_systems_order])
                else:
                    scores = scores[mt_system]
                ref_based_scores.append(scores)
            ref_based_score = np.mean(np.array(ref_based_scores), axis=0)
        else:
            raise RuntimeError(f"Unknown ref_based_system {args.ref_based_system}.")

        human_scores = np.array(human_scores, dtype=float)
        non_none_indices = np.argwhere(~np.isnan(human_scores)).flatten()
        if non_none_indices.shape[0] >= 2:
            wandb.log(
                {
                    f"ref_based_corr_w_human{out_suffix}": spearmanr(ref_based_score[non_none_indices], human_scores[non_none_indices])[0],
                    f"ref_based_corr_w_human_pearson{out_suffix}": pearsonr(ref_based_score[non_none_indices], human_scores[non_none_indices])[0],
                    f"ref_based_corr_w_human_kendalltau{out_suffix}": kendalltau(ref_based_score[non_none_indices], human_scores[non_none_indices])[0],
                }
            )
        if mt_system == "all":
            write_text_file(ref_based_score.tolist(), f"{args.output_dir}/ref_based_sent_scores.txt")

    else:
        ref_based_score = None
    qe_systems_outputs = []
    success = False
    for qe_system in args.comet_qe_systems:
        qe_model = load_comet_model(args.comet_model_root_path, qe_system)
        scores = qe_model.predict(qe_data, batch_size=args.batch_size, gpus=args.nr_gpus).scores
        success = gather_corr(human_scores, scores, qe_system, qe_systems_outputs, ref_based_score, args.ground_truth)
    if args.synthetic_qe_systems == '20-human-with-noise':
        for i in range(1, 21):
            synthetic_qe_score = add_noise(human_scores, noise_level=0.01 * i)
            success = gather_corr(human_scores=human_scores,
                                  qe_score=synthetic_qe_score,
                                  qe_system=f'human_with_noise_{i}',
                                  qe_systems_outputs=qe_systems_outputs,
                                  ref_based_score=ref_based_score,
                                  ground_truth=args.ground_truth)
    if args.qe_official_submissions == 'wmt22':
        assert args.dataset == "wmt22-testqe-mqm"
        submissions_root_path = f"{args.data_root_path}/wmt-qe-2022-data/official_submissions/task1_mqm/" \
                                f"{args.src_lang}-{args.tgt_lang}"
        participants = os.listdir(submissions_root_path)
        participants = [p for p in participants if os.path.isdir(f"{submissions_root_path}/{p}")]

        for participant in participants:
            qe_system = participant.split(' - ')[0]
            scores = load_text_file(f"{submissions_root_path}/{participant}/predictions.txt")[3:]
            scores = np.array([float(x.split('\t')[-1]) for x in scores])
            success = gather_corr(human_scores=human_scores,
                                  qe_score=scores,
                                  qe_system=qe_system,
                                  qe_systems_outputs=qe_systems_outputs,
                                  ref_based_score=ref_based_score,
                                  ground_truth=args.ground_truth)
    elif args.qe_official_submissions == 'wmt22-qe-metrics':
        assert args.dataset == "wmt22-testmetrics"
        domain_list = load_text_file(
            f"{args.data_root_path}/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/sources/{args.src_lang}-"
            f"{args.tgt_lang}_domain.txt"
        )
        submissions_root_path = f"{args.data_root_path}/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/metric-scores/{args.src_lang}-{args.tgt_lang}"
        for qe_file in glob(f"{submissions_root_path}/*-src.seg.score"):
            participant = qe_file.split('/')[-1].replace("-src.seg.score", "")
            scores = pd.read_csv(
                qe_file, names=["MT_system", "score"], header=None, delimiter="\t"
            ).groupby(
                by="MT_system"
            )['score'].apply(list).apply(lambda x: filter_by_domain(args.data_domain, domain_list, x)).to_dict()
            if mt_system == "all":
                scores = np.concatenate([scores[x] for x in mt_systems_order])
            else:
                scores = scores[mt_system]
            success = gather_corr(human_scores=human_scores,
                                  qe_score=scores,
                                  qe_system=participant,
                                  qe_systems_outputs=qe_systems_outputs,
                                  ref_based_score=ref_based_score,
                                  ground_truth=args.ground_truth)
    elif args.qe_official_submissions == 'wmt24-qe-metrics':
        # Reference-free (`*-src`) submissions to the WMT24 metrics shared task
        assert args.dataset == "wmt24-testmetrics-esa"
        lp = f"{args.src_lang}-{args.tgt_lang}"
        mask = wmt24_segment_mask(args.data_root_path, lp, args.data_domain)
        for qe_file in sorted(glob(f"{args.data_root_path}/{WMT24_DIR}/metric-scores/{lp}/*-src.seg.score")):
            participant = qe_file.split('/')[-1].replace("-src.seg.score", "")
            scores = {k: v[np.array(mask)] for k, v in load_seg_scores(qe_file).items()}
            if mt_system == "all":
                scores = np.concatenate([scores[x] for x in mt_systems_order])
            else:
                scores = scores[mt_system]
            success = gather_corr(human_scores=human_scores,
                                  qe_score=scores,
                                  qe_system=participant,
                                  qe_systems_outputs=qe_systems_outputs,
                                  ref_based_score=ref_based_score,
                                  ground_truth=args.ground_truth)

    ranked_qes = sort_qe_systems(qe_systems_outputs)
    # Remove the sentence-level scores from the output for readability
    for item in ranked_qes:
        item.pop('sent_level_scores')
    if args.ground_truth == 'human':
        out_file = f"{args.output_dir}/human_based_rank{out_suffix}.txt"
    elif args.ground_truth == 'ref_based_metrics':
        out_file = f"{args.output_dir}/ref_based_rank{out_suffix}.txt"
    else:
        raise RuntimeError(f"Unknown ground_truth {args.ground_truth}.")
    dump_json(ranked_qes, out_file)

    if mt_system == "all":
        wandb.log(
            {
                f"rank_qe_by_{args.ground_truth}": '/'.join([x['name'] for x in ranked_qes])
            }
        )
    return qe_systems_outputs


def sacrebleu_metric(args):
    # Imported here: the BLEURT venv has an old single-module sacrebleu (1.4.4) that shadows this one
    from sacrebleu.metrics import BLEU, CHRF, TER
    # Tokenization as sacrebleu's defaults for these target languages (13a would not split zh/ja into words)
    tokenize = {'zh': 'zh', 'ja': 'ja-mecab'}.get(args.tgt_lang)
    asian = args.tgt_lang in ['zh', 'ja']
    return {
        'BLEU': lambda: BLEU(effective_order=True, tokenize=tokenize),
        'CHRF': lambda: CHRF(),
        'TER': lambda: TER(normalized=asian, asian_support=asian),
    }[args.ref_based_system]()


def score_sacrebleu(metric, samples):
    """Sentence-level scores of a sacrebleu metric, one reference per sample (TER: lower = better)"""
    return [metric.sentence_score(x['mt'], [x['ref']]).score for x in samples]


MODEL_BASED_METRICS = ['XCOMET-XL', 'wmt22-comet-da', 'unite-mup', 'bert-score', 'bleurt', 'metricx-23-xl-v2p0']
_loaded_models = {}  # loaded once per process, and only if some segment is not cached


def score_model_based(args, samples):
    """Segment scores of a model-based ref-based metric for [{'src', 'mt', 'ref'}] (MetricX: lower = better)"""
    mt, ref = [x['mt'] for x in samples], [x['ref'] for x in samples]
    if 'comet' in args.ref_based_system.lower() or 'unite' in args.ref_based_system.lower():
        if args.ref_based_system not in _loaded_models:
            _loaded_models[args.ref_based_system] = load_comet_model(args.comet_model_root_path, args.ref_based_system)
        return _loaded_models[args.ref_based_system].predict(
            samples, batch_size=args.batch_size, gpus=args.nr_gpus
        ).scores
    elif args.ref_based_system == 'bert-score':
        import bert_score
        _, _, scores = bert_score.score(mt, ref, lang=args.tgt_lang)
        return scores.numpy()
    elif args.ref_based_system == 'bleurt':
        if 'bleurt' not in _loaded_models:
            import sentencepiece as spm
            from bleurt import score
            _loaded_models['bleurt'] = score.BleurtScorer("../models/BLEURT-20")
        return _loaded_models['bleurt'].score(references=ref, candidates=mt)
    elif args.ref_based_system == 'metricx-23-xl-v2p0':
        from metricx23_predict import calculate_ref_based_score
        return calculate_ref_based_score(mt, ref)
    raise RuntimeError(f"Unknown ref_based_system {args.ref_based_system}.")


def gather_corr(human_scores, qe_score, qe_system, qe_systems_outputs, ref_based_score, ground_truth):
    """
    Calculate correlation of QE scores to gold scores (human/reference based) and append results to `qe_systems_outputs`.
    Return True if successfully executed. False if not
    """
    human_scores = np.array(human_scores, dtype=float)
    non_none_indices = np.argwhere(~np.isnan(human_scores)).flatten()
    if non_none_indices.shape[0] < 2 or \
            np.equal(human_scores[non_none_indices], human_scores[non_none_indices][0]).all():
        return False
    if ground_truth == "human":
        corr_score = spearmanr(np.array(qe_score)[non_none_indices], human_scores[non_none_indices])
        corr_score_pearson = pearsonr(np.array(qe_score)[non_none_indices], human_scores[non_none_indices])
        corr_score_kendalltau = kendalltau(np.array(qe_score)[non_none_indices], human_scores[non_none_indices])

    elif ground_truth == "ref_based_metrics":
        corr_score = spearmanr(np.array(qe_score)[non_none_indices], np.array(ref_based_score)[non_none_indices])
        corr_score_pearson = pearsonr(np.array(qe_score)[non_none_indices], np.array(ref_based_score)[non_none_indices])
        corr_score_kendalltau = kendalltau(np.array(qe_score)[non_none_indices], np.array(ref_based_score)[non_none_indices])
    else:
        raise RuntimeError(f"Unknown ground_truth {ground_truth}.")
    qe_systems_outputs.append(
        {
            'name': qe_system,
            'sent_level_scores': qe_score,
            'corr_with_gold': corr_score,
            'corr_with_gold_pearson': corr_score_pearson,
            'corr_with_gold_kendalltau': corr_score_kendalltau
        }
    )
    return True


def sort_qe_systems(qe_systems_outputs):
    # First sort by correlation with the ground truth
    qe_systems_outputs.sort(key=lambda x: x['corr_with_gold'][0], reverse=True)

    # Then use Williams Test to see which ordering is significant
    for qe_system_i in range(0, len(qe_systems_outputs)):
        significantly_worse = []
        for worse_qe_system_i in range(qe_system_i + 1, len(qe_systems_outputs)):
            p_value = WilliamsTest(
                qe_systems_outputs[qe_system_i]['corr_with_gold'][0],
                qe_systems_outputs[worse_qe_system_i]['corr_with_gold'][0],
                spearmanr(qe_systems_outputs[qe_system_i]['sent_level_scores'], qe_systems_outputs[worse_qe_system_i]['sent_level_scores'])[0],
                len(qe_systems_outputs[qe_system_i]['sent_level_scores'])
            )
            if p_value < 0.05:
                significantly_worse.append(qe_systems_outputs[worse_qe_system_i]['name'])
        qe_systems_outputs[qe_system_i]['significantly_worse'] = significantly_worse

    return qe_systems_outputs


if __name__ == '__main__':
    main()
