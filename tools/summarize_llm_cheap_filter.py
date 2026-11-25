"""Stage 76: rebuild results/llm-cheap-filter.json from a complete journal without touching the journal.

Used after stage-76 deviation 2 (machine lost after all 24 runs, before the summary was committed). Builds the same
result object as the final step of experiments.llm_cheap_filter.main (no environment entry is appended, so the journal
stays byte-identical to the snapshot). Refuses unless the journal holds exactly the protocol's run grid in order.
Prints counts and the output path only; no loss, accuracy or score is printed.

python -m tools.summarize_llm_cheap_filter --journal J [--summary results/llm-cheap-filter.json]
"""
import argparse
import json
from pathlib import Path

from experiments import llm_cheap_filter as stage
from experiments import llm_gpu as s69


def build(journal_path):
    plan = json.loads(stage.PROTOCOL.read_text())
    if plan['stage'] != 76 or tuple(plan['design']['arms']) != stage.ARMS:
        raise ValueError('Protocol differs from the implementation')
    journal = s69.read_journal(journal_path)
    cals = [r for r in journal if r.get('kind') == 'calibration']
    if len(cals) != 1:
        raise ValueError('Expected exactly one calibration entry')
    calibration = cals[0]
    rows = [r for r in journal if r.get('kind') == 'run']
    if [(r['seed'], r['arm']) for r in rows] != stage.run_grid(plan['design']['seeds']):
        raise ValueError('Journal does not hold the complete protocol run grid in order')
    return dict(stage=76, protocol=plan, journal_sha256=s69.sha256(journal_path),
                environments=[r for r in journal if r.get('kind') == 'environment'],
                calibration={k: v for k, v in calibration.items() if k != 'population'},
                summary=stage.summarize(rows, calibration, plan), runs=len(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--summary', type=Path, default=Path('results/llm-cheap-filter.json'))
    args = parser.parse_args()
    result = build(args.journal)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"{result['runs']} runs, {len(result['environments'])} environment entries; summary written to {args.summary}")


if __name__ == '__main__':
    main()
