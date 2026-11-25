"""Summarize pairwise verdicts against hardware from a committed scores file.

For every method and cache regime, report how often an ordered verdict
("A is faster" or "B is faster") agrees with the hardware, how often it is
reversed, and how many hardware-ordered pairs the method orders correctly.

Usage: python -m validation.pair_precision validation/phase1-holdout-scores.json
"""

import argparse
import json
from pathlib import Path

ORDERED = ("a_faster", "b_faster")


def summarize(confusion):
    """Rows of `confusion` are hardware verdicts, columns are predicted verdicts."""
    cells = [(hw, pred, n) for hw, row in confusion.items() for pred, n in row.items()]
    calls = sum(n for _, pred, n in cells if pred in ORDERED)
    hw_ordered = sum(n for hw, _, n in cells if hw in ORDERED)

    def share(predicate):
        return sum(n for hw, pred, n in cells if pred in ORDERED and predicate(hw, pred))

    return {
        "ordered_calls": calls,
        "agree": share(lambda hw, pred: hw == pred),
        "reversed": share(lambda hw, pred: hw in ORDERED and hw != pred),
        "hardware_equivalent": share(lambda hw, pred: hw == "indistinguishable"),
        "hardware_unresolved": share(lambda hw, pred: hw == "unresolved"),
        "hardware_ordered": hw_ordered,
        "hardware_ordered_found": sum(
            n for hw, pred, n in cells if hw in ORDERED and hw == pred
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.scores.read_text())["summary"]
    print("| Cache | Method | Ordered calls | Agree | Reversed | Hardware equivalent "
          "| Hardware unresolved | Hardware-ordered pairs ordered correctly |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cache, methods in summary.items():
        for method, scores in methods.items():
            confusion = scores.get("pair_confusion")
            if not confusion:
                continue
            s = summarize(confusion)
            calls = s["ordered_calls"]
            print(
                f"| {cache} | {method} | {calls} "
                f"| {s['agree'] / calls:.1%} | {s['reversed'] / calls:.1%} "
                f"| {s['hardware_equivalent'] / calls:.1%} "
                f"| {s['hardware_unresolved'] / calls:.1%} "
                f"| {s['hardware_ordered_found'] / s['hardware_ordered']:.1%} |"
            )


if __name__ == "__main__":
    main()
