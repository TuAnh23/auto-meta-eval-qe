"""
Tables of the automatic QE evaluation part of the EAMT 2024 paper (Dinh et al., 2024, "Quality Estimation with
k-nearest Neighbors and Automatic Evaluation for Model-specific Quality Estimation"), with the xCOMET XL entries added.

Values of the other metrics come from the wandb export the paper was made from (analysis/wandb.csv). xCOMET XL values on
the QE-M Task come from the runs of run_xcomet_qem.sh, computed from their output folders the same way corr_eval_qe.py
and meta_meta_eval_qe.py log them to wandb:
  - segment-level: Spearman(ref_based_sent_scores, human_sent_scores) over segments with a human score (Eq. 12)
  - QE ranking: Spearman between the QE ranking by the metric and by humans, meta_meta_eval_qe.txt (Eq. 11)

    python paper_tables_xcomet.py --out-dir <dir>
"""
import argparse
import json
import os
import re
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# (wandb name, name in the paper), in the order of Table 2
METRICS = [
    ('BLEU', 'BLEU'), ('TER', 'TER'), ('CHRF', 'chrF'), ('bert-score', 'BERTScore'), ('bleurt', 'BLEURT'),
    ('unite-mup', 'UniTE-MUP'), ('wmt22-comet-da', 'COMET 22'), ('XCOMET-XL', 'xCOMET XL'),
    ('metricx-23-xl-v2p0', 'MetricX-23 XL'),
]
NAME = dict(METRICS)
# WMT22 MetricX submissions (scores read from the shared task), in the paper only inside the Figure 5 boxes
NAME.update({'metricx_xl_MQM_2020': 'MetricX XL (WMT22 subm.)', 'metricx_xxl_MQM_2020': 'MetricX XXL (WMT22 subm.)'})
NR_REFS = {'single_0': 1, 'mul_0': 2, 'mul_1': 4, 'mul_2': 6}
DOMAINS = ['conversation', 'ecommerce', 'news', 'social']
NEW_RUNS_AFTER = 1790000000000000000  # run ids are `date +%s%N`; > this = runs from Oct 2026
QEM = 'wmt22-qe-metrics'  # QE-M Task: WMT22 QE-as-a-Metric submissions, WMT22 General MT test set
QET = 'wmt22'  # QE Task: WMT22 QE shared task, News, single MT system


def load_paper_rows(wandb_csv):
    df = pd.read_csv(wandb_csv)
    df = df[(df['tgt_lang'] == 'de') & df['qe_official_submissions'].isin([QEM, QET])].copy()
    df['source'] = 'wandb ' + df['Name'].astype(str)
    return df


def load_new_xcomet_rows(output_root):
    rows = []
    pattern = re.compile(rf"rank_{QEM}_(\w+)_XCOMET-XL_(single_0|mul_0|mul_1|mul_2)_(\d+)$")
    for run_dir in sorted(glob(f"{output_root}/rank_{QEM}_*_XCOMET-XL_*")):
        match = pattern.match(os.path.basename(run_dir))
        if match is None or int(match.group(3)) < NEW_RUNS_AFTER:
            continue
        human = np.array([float(x) for x in open(f"{run_dir}/human_sent_scores.txt")])
        ref_based = np.array([float(x) for x in open(f"{run_dir}/ref_based_sent_scores.txt")])
        has_human = ~np.isnan(human)
        rank_corr = json.load(open(f"{run_dir}/meta_meta_eval_qe.txt"))
        row = {
            'qe_official_submissions': QEM, 'data_domain': match.group(1), 'ref_based_system': 'XCOMET-XL',
            'total_nr_refs': NR_REFS[match.group(2)], 'Name': match.group(3), 'source': f"new run {match.group(3)}",
            'ref_based_corr_w_human': spearmanr(ref_based[has_human], human[has_human])[0],
            'ref_based_rank_corr_of_corr': rank_corr['all'][0],
        }
        row.update({f"ref_based_rank_corr_of_corr_{k}": v[0] for k, v in rank_corr.items() if k != 'all'})
        rows.append(row)
    new = pd.DataFrame(rows, columns=None if rows else ['data_domain', 'total_nr_refs', 'Name'])
    # If a config was run more than once, keep the latest run
    return new.sort_values('Name').groupby(['data_domain', 'total_nr_refs']).tail(1)


def fmt(x):
    return '–' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def md_table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(header) - 1)) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(lines)


def pick(df, task, domain, nr_refs, column='ref_based_rank_corr_of_corr'):
    """{metric: value} for one setting."""
    sub = df[(df['qe_official_submissions'] == task) & (df['data_domain'] == domain) & (df['total_nr_refs'] == nr_refs)]
    return {r['ref_based_system']: r.get(column, np.nan) for _, r in sub.iterrows()}


def label(metric, new=True):
    return f"**{NAME[metric]}** (new)" if metric == 'XCOMET-XL' and new else NAME[metric]


def box_stats(values):
    v = pd.Series(values).dropna()
    return [len(v), fmt(v.min()), fmt(v.quantile(.25)), fmt(v.median()), fmt(v.quantile(.75)), fmt(v.max()), fmt(v.mean())]


BOX_HEADER = ['n', 'min', 'Q1', 'median', 'Q3', 'max', 'mean']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb-csv', default='analysis/wandb.csv')
    parser.add_argument('--output-root', default='output')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    paper = load_paper_rows(args.wandb_csv)
    new = load_new_xcomet_rows(args.output_root)
    # QE-M xCOMET XL values: always from the new runs (2024 runs existed only for News, kept for the check below)
    qem_xcomet_2024 = paper[(paper['qe_official_submissions'] == QEM) & (paper['ref_based_system'] == 'XCOMET-XL')]
    rows = pd.concat([paper.drop(qem_xcomet_2024.index), new], ignore_index=True)
    files = {}

    # ---- Table 2
    qe_task = pick(rows, QET, 'news', 1)
    qem_task = pick(rows, QEM, 'all', 2)
    files['table2_overall.md'] = (
        "# Table 2: overall performance on ranking QE metrics, with xCOMET XL on the QE-M Task\n\n"
        "Spearman correlation between the QE ranking made with the reference-based metric and the ranking made with "
        "human MQM scores (Eq. 11). QE Task: WMT22 QE shared task (News, one MT system), **1 reference**. "
        "QE-M Task: WMT22 QE-as-a-Metric (all 4 domains, 14 MT systems), **2 references** (as in the paper).\n\n"
        + md_table(['Metric', 'QE Task', 'QE-M Task'],
                   [[label(m), fmt(qe_task.get(m)), fmt(qem_task.get(m)) + (' \\*' if m == 'XCOMET-XL' else '')]
                    for m, _ in METRICS])
        + "\n\n\\* xCOMET is trained on WMT22 MQM data except the News domain (the paper's footnote), so on the QE-M "
          "Task it has seen the human scores of 3 of the 4 domains: data leakage, the value is not a fair comparison. "
          "The paper's QE Task value of xCOMET XL (0.358) is unchanged; it came from a 2024 run.\n"
    )

    # ---- Figure 2a: per domain
    per_domain = {d: pick(rows, QEM, d, 2) for d in ['all'] + DOMAINS}
    metrics_2a = [m for m, _ in METRICS]
    files['figure2a_by_domain.md'] = (
        "# Figure 2a: QE ranking performance per domain (QE-M Task, 2 references)\n\n"
        "Spearman correlation of the QE ranking per domain (Eq. 11). The paper's box plot (Analysis.ipynb) pools "
        "the 5 values of each metric: the 4 domains and `all`.\n\n"
        + md_table(['Metric', 'all'] + DOMAINS,
                   [[label(m)] + [fmt(per_domain[d].get(m)) for d in ['all'] + DOMAINS] for m in metrics_2a])
        + "\n\nBox statistics over the 5 values (as plotted):\n\n"
        + md_table(['Metric'] + BOX_HEADER,
                   [[label(m)] + box_stats([per_domain[d].get(m) for d in ['all'] + DOMAINS]) for m in metrics_2a])
        + "\n\nxCOMET XL has seen the human scores of all domains except News during training (data leakage), so only "
          "its News value is a fair comparison.\n"
    )

    # ---- Figure 2b: per MT system
    sub = rows[(rows['qe_official_submissions'] == QEM) & (rows['data_domain'] == 'all') & (rows['total_nr_refs'] == 2)]
    mt_cols = sorted(c for c in rows.columns if c.startswith('ref_based_rank_corr_of_corr_')
                     and sub[c].notna().any() and not c.endswith('_-'))
    mt_names = [c.replace('ref_based_rank_corr_of_corr_', '') for c in mt_cols]
    by_metric = sub.set_index('ref_based_system')
    files['figure2b_by_mt_system.md'] = (
        "# Figure 2b: QE ranking performance per MT system (QE-M Task, all domains, 2 references)\n\n"
        "Spearman correlation of the QE ranking on the output of each MT system of WMT22 General (Eq. 11). "
        "14 MT systems are scored, but `chrf_bestmbr` has no MQM scores in WMT22 (all NaN), so its correlation is "
        "undefined for every metric (also in the paper's runs): the paper's box plot is over the remaining 13.\n\n"
        + md_table(['MT system'] + [label(m) for m, _ in METRICS],
                   [[mt] + [fmt(by_metric.loc[m, c]) if m in by_metric.index else '–' for m, _ in METRICS]
                    for mt, c in zip(mt_names, mt_cols)])
        + "\n\nBox statistics over the 13 MT systems (as plotted):\n\n"
        + md_table(['Metric'] + BOX_HEADER,
                   [[label(m)] + box_stats(by_metric.loc[m, mt_cols].astype(float).values) for m, _ in METRICS
                    if m in by_metric.index])
        + "\n\nxCOMET XL: data leakage on the non-News part of this test set (see Table 2).\n"
    )

    # ---- Figure 3: segment level vs QE ranking
    def seg_vs_rank(task, domain, rank_column='ref_based_rank_corr_of_corr', metrics=None):
        seg = pick(rows, task, domain, 2, 'ref_based_corr_w_human')
        rank = pick(rows, task, domain, 2, rank_column)
        metrics = metrics or [m for m, _ in METRICS]
        return md_table(['Metric', 'Segment-level Spearman (Eq. 12)', 'QE ranking Spearman (Eq. 11)'],
                        [[label(m, new=task == QEM), fmt(seg.get(m)), fmt(rank.get(m))]
                         for m in sorted(metrics, key=lambda m: -np.nan_to_num(seg.get(m, np.nan), nan=-9))])

    files['figure3_segment_vs_ranking.md'] = (
        "# Figure 3: segment-level evaluation vs. QE ranking performance (2 references)\n\n"
        "## (a) QE-M Task: multiple domains, multiple MT systems\n\n" + seg_vs_rank(QEM, 'all')
        + "\n\nxCOMET XL: data leakage on the non-News part of this test set (see Table 2).\n\n"
        "## (b) QE Task: single domain, single MT system\n\n"
        "Unchanged from the paper: xCOMET XL was already in this figure. Repeated here for completeness.\n\n"
        + seg_vs_rank(QET, 'news') + "\n"
    )

    # ---- Figure 4: limited to the comet_bestmbr MT system
    files['figure4_single_mt_system.md'] = (
        "# Figure 4: segment-level evaluation vs. QE ranking, limited by MT system and domain (QE-M Task, 2 references)\n\n"
        "QE ranking on the output of the MT system `comet_bestmbr` only. As in the paper's notebook, the segment-level "
        "correlation (x axis) is the one over all MT systems of the domain, not only `comet_bestmbr`.\n\n"
        "## (a) All domains, single MT system (comet_bestmbr)\n\n"
        + seg_vs_rank(QEM, 'all', 'ref_based_rank_corr_of_corr_comet_bestmbr')
        + "\n\n## (b) Single domain (Social), single MT system (comet_bestmbr)\n\n"
        + seg_vs_rank(QEM, 'social', 'ref_based_rank_corr_of_corr_comet_bestmbr')
        + "\n\nxCOMET XL: data leakage, its training data includes the human scores of the Social domain.\n"
    )

    # ---- Figure 5: number of references
    refs_list = [1, 2, 4, 6]
    by_refs = {n: pick(rows, QEM, 'all', n) for n in refs_list}
    all_metrics = [m for m, _ in METRICS] + ['metricx_xl_MQM_2020', 'metricx_xxl_MQM_2020']
    files['figure5_nr_references.md'] = (
        "# Figure 5: effect of the number of references (QE-M Task, all domains)\n\n"
        "QE ranking Spearman correlation (Eq. 11) by total number of references: 1 and 2 are the human references "
        "(refA, refA+refB); 4 and 6 add 1 or 2 paraphrases of each human reference (Lelon/t5-german-paraphraser-large; "
        "4: greedy, 6: sampling with temperature 0.5). The paper's boxes are over the metrics that exist for each "
        "number of references; for 1 and 2 references these include the two WMT22 MetricX submissions, which have "
        "no scores with paraphrased references.\n\n"
        + md_table(['Metric'] + [f"{n} ref{'s' if n > 1 else ''}" for n in refs_list],
                   [[label(m)] + [fmt(by_refs[n].get(m)) for n in refs_list] for m in all_metrics])
        + "\n\nBox statistics over metrics, as in the paper (without xCOMET XL):\n\n"
        + md_table(['Total refs'] + BOX_HEADER,
                   [[n] + box_stats([v for m, v in by_refs[n].items() if m != 'XCOMET-XL']) for n in refs_list])
        + "\n\nBox statistics over metrics, with xCOMET XL added:\n\n"
        + md_table(['Total refs'] + BOX_HEADER, [[n] + box_stats(list(by_refs[n].values())) for n in refs_list])
        + "\n\nxCOMET XL: data leakage on the non-News part of this test set (see Table 2).\n"
    )

    # ---- Extra: per domain with 1 reference (run for completeness, not in the paper)
    per_domain_1 = {d: pick(rows, QEM, d, 1) for d in ['all'] + DOMAINS}
    files['extra_by_domain_1ref.md'] = (
        "# Extra (not in the paper): QE ranking performance per domain with 1 reference (QE-M Task)\n\n"
        + md_table(['Metric', 'all'] + DOMAINS,
                   [[label(m)] + [fmt(per_domain_1[d].get(m)) for d in ['all'] + DOMAINS] for m in metrics_2a])
        + "\n"
    )

    # ---- Provenance of the new runs and the check against the 2024 News runs
    check = []
    for _, old in qem_xcomet_2024.iterrows():
        n = new[(new['data_domain'] == old['data_domain']) & (new['total_nr_refs'] == old['total_nr_refs'])]
        if len(n):
            n = n.iloc[0]
            check.append([old['data_domain'], int(old['total_nr_refs']),
                          fmt(old['ref_based_corr_w_human']), fmt(n['ref_based_corr_w_human']),
                          fmt(old['ref_based_rank_corr_of_corr']), fmt(n['ref_based_rank_corr_of_corr'])])
    files['runs.md'] = (
        "# New xCOMET XL runs (QE-M Task)\n\n"
        + md_table(['Domain', 'Total refs', 'Run id (output folder suffix)', 'Segment-level', 'QE ranking'],
                   [[r['data_domain'], int(r['total_nr_refs']), r['Name'], fmt(r['ref_based_corr_w_human']),
                     fmt(r['ref_based_rank_corr_of_corr'])]
                    for _, r in new.sort_values(['data_domain', 'total_nr_refs']).iterrows()])
        + "\n\n## Reproducibility check: News domain, 2024 runs vs. new runs\n\n"
        + md_table(['Domain', 'Total refs', 'Segment-level 2024', 'Segment-level new', 'QE ranking 2024',
                    'QE ranking new'], check)
        + "\n"
    )

    for name, text in files.items():
        with open(f"{args.out_dir}/{name}", 'w') as f:
            f.write(text)
        print(f"wrote {args.out_dir}/{name}")


if __name__ == '__main__':
    main()
