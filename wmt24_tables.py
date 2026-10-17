"""
Tables of the automatic QE evaluation (QE-M Task) of the EAMT 2024 paper, redone on WMT24 with ESA human scores, for
all language pairs with ESA scores, from the runs of run_wmt24_batch.sh. Single reference (refA), no paraphrases.

  - segment-level: Spearman(ref_based_sent_scores, human_sent_scores) over segments with a human score (paper Eq. 12)
  - QE ranking: Spearman between the QE ranking by the metric and by humans, meta_meta_eval_qe.txt (Eq. 11)

    python wmt24_tables.py --output-root output --out-dir <dir>
"""
import argparse
import json
import os
import re
from glob import glob

import numpy as np
from scipy.stats import spearmanr

METRICS = [
    ('BLEU', 'BLEU'), ('TER', 'TER'), ('CHRF', 'chrF'), ('bert-score', 'BERTScore'), ('bleurt', 'BLEURT'),
    ('unite-mup', 'UniTE-MUP'), ('wmt22-comet-da', 'COMET 22'), ('XCOMET-XL', 'xCOMET XL'),
    ('metricx-23-xl-v2p0', 'MetricX-23 XL'),
]
NAME = dict(METRICS)
LANG_PAIRS = ['en-cs', 'en-es', 'en-hi', 'en-is', 'en-ja', 'en-ru', 'en-uk', 'en-zh', 'cs-uk']
EN_PAIRS = [lp for lp in LANG_PAIRS if lp.startswith('en-')]
DOMAINS = {lp: ['literary', 'news', 'social', 'speech'] for lp in EN_PAIRS}
DOMAINS['cs-uk'] = ['education', 'news', 'official', 'personal', 'voice']
# QE submissions of the same model families as two of the ranking metrics (xCOMET, MetricX), left out in a robustness
# check: a ref-based metric may favour the QE variant of itself
SAME_FAMILY_QE = ['XCOMET-QE', 'MetricX-24-QE', 'MetricX-24-Hybrid-QE']


def load_runs(output_root):
    """{(lp, domain, metric): run folder}, the latest run per configuration"""
    runs = {}
    for d in sorted(glob(f"{output_root}/rank_wmt24-qe-metrics_*_single_0_*")):
        m = re.fullmatch(r"rank_wmt24-qe-metrics_([a-z]+-[a-z]+)_([a-z]+)_(.+)_single_0_(\d+)", os.path.basename(d))
        if m is None or not os.path.isfile(f"{d}/meta_meta_eval_qe.txt"):
            continue
        key = (m.group(1), m.group(2), m.group(3))
        if key not in runs or int(m.group(4)) > int(runs[key][1]):
            runs[key] = (d, m.group(4))
    return {k: v[0] for k, v in runs.items()}


def read_floats(path):
    with open(path) as f:
        return np.array([float(x) for x in f.read().split()])


def run_values(run_dir):
    rank = {k: v[0] for k, v in json.load(open(f"{run_dir}/meta_meta_eval_qe.txt")).items()}
    human = read_floats(f"{run_dir}/human_sent_scores.txt")
    ref_based = read_floats(f"{run_dir}/ref_based_sent_scores.txt")
    ok = ~np.isnan(human)
    seg = spearmanr(ref_based[ok], human[ok])[0]
    return rank, seg


def rank_without(run_dir, excluded):
    """QE ranking correlation ('all' MT systems) recomputed without the QE submissions in `excluded`"""
    def corr(path):
        return {x['name']: x['corr_with_gold'][0] for x in json.load(open(path))}
    human, ref_based = corr(f"{run_dir}/human_based_rank.txt"), corr(f"{run_dir}/ref_based_rank.txt")
    names = [n for n in human if n not in excluded]
    return spearmanr([ref_based[n] for n in names], [human[n] for n in names])[0]


def fmt(x):
    return '–' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def mean(values):
    values = [v for v in values if v is not None and not np.isnan(v)]
    return float(np.mean(values)) if values else None


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(header) - 1)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def bold_best(values):
    """Format a column of values, the best one(s) in bold"""
    best = max([v for v in values if v is not None and not np.isnan(v)], default=None)
    return [f"**{fmt(v)}**" if v is not None and best is not None and round(v, 3) == round(best, 3) else fmt(v)
            for v in values]


def metric_by_column_table(columns, value, average_label='Average'):
    """Rows: metrics; columns: `columns` + average; best value per column in bold"""
    cols = [[value(m, c) for m, _ in METRICS] for c in columns]
    cols.append([mean([value(m, c) for c in columns]) for m, _ in METRICS])
    cols = [bold_best(c) for c in cols]
    return md_table(['Metric'] + columns + [average_label],
                    [[NAME[m]] + [c[i] for c in cols] for i, (m, _) in enumerate(METRICS)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=str, default='output')
    parser.add_argument('--out-dir', type=str)
    args = parser.parse_args()

    runs = load_runs(args.output_root)
    values = {k: run_values(d) for k, d in runs.items()}
    expected = [(lp, dom, m) for lp in LANG_PAIRS for dom in ['all'] + DOMAINS[lp] for m, _ in METRICS]
    missing = [k for k in expected if k not in runs]

    def rank(lp, dom, m, mt='all'):
        return values[(lp, dom, m)][0].get(mt) if (lp, dom, m) in values else None

    def seg(lp, dom, m):
        return values[(lp, dom, m)][1] if (lp, dom, m) in values else None

    def mt_systems(lp, dom='all'):
        any_run = next((values[(lp, dom, m)][0] for m, _ in METRICS if (lp, dom, m) in values), {})
        return [k for k in any_run if k != 'all']

    def mean_over_mt(lp, dom, m):
        return mean([rank(lp, dom, m, mt) for mt in mt_systems(lp, dom)])

    note_avg = ("Average: unweighted mean of the Spearman correlations over the language pairs. Best value per "
                "column in bold.")
    files = {}

    files['table2_overall.md'] = (
        "# Table 2 (WMT24, ESA): overall performance on ranking QE metrics\n\n"
        "QE-M Task on the WMT24 General MT test set: rank the reference-free submissions to the WMT24 metrics shared "
        "task with each reference-based metric, and correlate (Spearman) with the ranking by human ESA scores "
        "(Eq. 11). All domains, all MT systems with ESA scores, 1 reference (refA). " + note_avg + "\n\n"
        + metric_by_column_table(LANG_PAIRS, lambda m, lp: rank(lp, 'all', m))
        + "\n\n## Without the same-family QE submissions\n\n"
        f"Same as above, but without the QE submissions {', '.join(SAME_FAMILY_QE)}: xCOMET XL and MetricX-23 XL "
        "might favour the reference-free variants of their own model family. Computed from the per-QE-metric "
        "correlations in each run (`human_based_rank.txt`, `ref_based_rank.txt`).\n\n"
        + metric_by_column_table(
            LANG_PAIRS, lambda m, lp: rank_without(runs[(lp, 'all', m)], SAME_FAMILY_QE) if (lp, 'all', m) in runs
            else None)
        + "\n"
    )

    per_pair = []
    for lp in LANG_PAIRS:
        per_pair.append(f"## {lp}\n\n" + metric_by_column_table(['all'] + DOMAINS[lp], lambda m, d: rank(lp, d, m),
                                                                 average_label='Mean (5 values, as Fig. 2a pools)'
                                                                 if lp != 'cs-uk' else 'Mean (6 values)'))
    files['figure2a_by_domain.md'] = (
        "# Figure 2a (WMT24, ESA): QE ranking performance per domain\n\n"
        "Spearman correlation of the QE ranking per domain (Eq. 11), 1 reference. The en-* pairs share the domains "
        "literary, news, social and speech; cs-uk has education, news, official, personal and voice. As in the "
        "paper's Fig. 2a, `all` is pooled with the domains in the per-pair mean.\n\n"
        "## Average over the 8 en-* language pairs\n\n"
        + metric_by_column_table(['all', 'literary', 'news', 'social', 'speech'],
                                 lambda m, d: mean([rank(lp, d, m) for lp in EN_PAIRS]),
                                 average_label='Mean (5 values)')
        + "\n\n" + "\n\n".join(per_pair) + "\n"
    )

    blocks = []
    for lp in LANG_PAIRS:
        systems = mt_systems(lp)
        rows = [[mt] + [fmt(rank(lp, 'all', m, mt)) for m, _ in METRICS] for mt in systems]
        rows.append(['**Mean**'] + bold_best([mean_over_mt(lp, 'all', m) for m, _ in METRICS]))
        blocks.append(f"## {lp} ({len(systems)} MT systems)\n\n" + md_table(['MT system'] + [n for _, n in METRICS], rows))
    files['figure2b_by_mt_system.md'] = (
        "# Figure 2b (WMT24, ESA): QE ranking performance per MT system\n\n"
        "Spearman correlation of the QE ranking on the output of each MT system (Eq. 11), all domains, 1 reference. "
        "MT systems: those with ESA scores (WMT24 human evaluation covered a subset of the submitted systems).\n\n"
        "## Mean over MT systems, per language pair\n\n"
        + metric_by_column_table(LANG_PAIRS, lambda m, lp: mean_over_mt(lp, 'all', m))
        + "\n\n" + "\n\n".join(blocks) + "\n"
    )

    def seg_rank_table(lp_list, rank_value):
        segs = {m: mean([seg(lp, 'all', m) for lp in lp_list]) for m, _ in METRICS}
        ranks = {m: mean([rank_value(lp, m) for lp in lp_list]) for m, _ in METRICS}
        order = sorted([m for m, _ in METRICS], key=lambda m: -(segs[m] if segs[m] is not None else -9))
        return md_table(['Metric', 'Segment-level Spearman (Eq. 12)', 'QE ranking Spearman (Eq. 11)'],
                        [[NAME[m], fmt(segs[m]), fmt(ranks[m])] for m in order])

    files['figure3_segment_vs_ranking.md'] = (
        "# Figure 3a (WMT24, ESA): segment-level evaluation vs. QE ranking performance\n\n"
        "Multiple domains, multiple MT systems, 1 reference. Segment-level: Spearman correlation of the "
        "reference-based metric with ESA over all segments with an ESA score (Eq. 12). Metrics sorted by "
        "segment-level correlation. Fig. 3b (QE Task: WMT22 QE shared task) has no WMT24 counterpart and is not "
        "redone.\n\n"
        "## Average over the 9 language pairs\n\n" + seg_rank_table(LANG_PAIRS, lambda lp, m: rank(lp, 'all', m))
        + "\n\n## Segment-level correlation per language pair\n\n"
        + metric_by_column_table(LANG_PAIRS, lambda m, lp: seg(lp, 'all', m))
        + "\n\n" + "\n\n".join(f"## {lp}\n\n" + seg_rank_table([lp], lambda lp_, m: rank(lp_, 'all', m))
                               for lp in LANG_PAIRS) + "\n"
    )

    files['figure4_single_mt_system.md'] = (
        "# Figure 4 (WMT24, ESA): QE ranking limited to a single MT system and domain\n\n"
        "The paper's Fig. 4 shows the QE ranking on the output of one MT system (`comet_bestmbr`, a WMT22 system) "
        "against the segment-level correlation over all MT systems of the domain. WMT24 has different MT systems, so "
        "here the QE ranking is averaged over **every** single MT system, which shows the same effect without picking "
        "one system. 1 reference. The x axis (segment-level) is as in the paper: over all MT systems and domains.\n\n"
        "## (a) All domains, single MT system: mean over MT systems, then over the 9 language pairs\n\n"
        + seg_rank_table(LANG_PAIRS, lambda lp, m: mean_over_mt(lp, 'all', m))
        + "\n\n## (b) Single domain, single MT system: mean over (domain, MT system) cells, then over the 9 pairs\n\n"
        "Cells with fewer than 2 segments with ESA scores, or constant ESA scores, have no correlation and are "
        "skipped.\n\n"
        + seg_rank_table(LANG_PAIRS, lambda lp, m: mean([rank(lp, d, m, mt) for d in DOMAINS[lp]
                                                         for mt in mt_systems(lp, d)]))
        + "\n\n## (b) per language pair\n\n"
        + metric_by_column_table(LANG_PAIRS, lambda m, lp: mean([rank(lp, d, m, mt) for d in DOMAINS[lp]
                                                                  for mt in mt_systems(lp, d)]))
        + "\n"
    )

    run_rows = [[lp, dom, NAME[m], os.path.basename(runs[(lp, dom, m)]).rsplit('_', 1)[1],
                 fmt(seg(lp, dom, m)), fmt(rank(lp, dom, m))]
                for lp in LANG_PAIRS for dom in ['all'] + DOMAINS[lp] for m, _ in METRICS if (lp, dom, m) in runs]
    files['runs.md'] = (
        "# WMT24 ESA runs\n\n"
        f"{len(runs)} runs found, {len(missing)} missing"
        + (": " + ", ".join('/'.join(k) for k in missing) if missing else "") + ".\n\n"
        + md_table(['Lang pair', 'Domain', 'Metric', 'Run id (output folder suffix)', 'Segment-level', 'QE ranking'],
                   run_rows) + "\n"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for name, content in files.items():
        with open(f"{args.out_dir}/{name}", 'w') as f:
            f.write(content)
        print(f"wrote {args.out_dir}/{name}")
    if missing:
        print(f"WARNING: {len(missing)} runs missing")


if __name__ == '__main__':
    main()
