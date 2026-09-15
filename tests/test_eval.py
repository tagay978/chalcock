"""Check evaluate() counts outcomes correctly on hand-built cases."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts"))
import eval_realworld as E

ABSTAIN = 50
ok = True


def check(label, got, expected):
    global ok
    failures = [f"{k} expected {v}, got {got[k]}" for k, v in expected.items() if got[k] != v]
    if failures:
        ok = False
        for f in failures:
            print(f"  FAIL {label}: {f}")
    else:
        print(f"  ok   {label}")


# Shared annotations:
#   box A x=0.1 class 0    in-vocabulary
#   box B x=0.3 class 1    in-vocabulary
#   box D x=0.7 class 4    in-vocabulary, nothing predicted on it
#   box C x=0.5 unknown    out-of-vocabulary
truth = {
    "a.jpg": (
        [(0, 0.1, 0.5, 0.06, 0.4), (1, 0.3, 0.5, 0.06, 0.4), (4, 0.7, 0.5, 0.06, 0.4)],
        [(ABSTAIN, 0.5, 0.5, 0.06, 0.4)],
    )
}

# A model with no abstain class: names B wrongly, names the unknown bottle, and fires once
# where there is no annotation at all.
preds = {
    "a.jpg": [
        (0, 0.1, 0.5, 0.06, 0.4, 0.9),
        (2, 0.3, 0.5, 0.06, 0.4, 0.8),
        (3, 0.5, 0.5, 0.06, 0.4, 0.7),
        (7, 0.9, 0.5, 0.06, 0.4, 0.6),
    ]
}
check("no abstain class", E.evaluate(truth, preds, abstain_id=None),
      dict(known=3, localised=2, correct=1, unknown=1, unknown_hit=1, preds=4, named_preds=4))

# The same annotations against a model that can decline: abstaining on the out-of-vocabulary
# bottle is the right answer and must not be counted as a false alarm, and abstaining on an
# in-vocabulary bottle is a declined name rather than a wrong one.
preds_abstain = {
    "a.jpg": [
        (0, 0.1, 0.5, 0.06, 0.4, 0.9),
        (ABSTAIN, 0.3, 0.5, 0.06, 0.4, 0.8),
        (ABSTAIN, 0.5, 0.5, 0.06, 0.4, 0.7),
    ]
}
stats = E.evaluate(truth, preds_abstain, abstain_id=ABSTAIN)
check("abstentions counted apart", stats,
      dict(known=3, localised=2, correct=1, abstained_known=1,
           unknown=1, unknown_hit=0, abstained_unknown=1, preds=3, named_preds=1))

# Precision is over brand names only. Counting abstentions in the denominator would let a model
# look precise by staying quiet.
check("precision denominator excludes abstentions", stats, dict(named_preds=1))

# Two annotations overlapping one prediction must not both be credited.
truth2 = {"b.jpg": ([(0, 0.2, 0.5, 0.2, 0.6), (0, 0.22, 0.5, 0.2, 0.6)], [])}
preds2 = {"b.jpg": [(0, 0.21, 0.5, 0.2, 0.6, 0.9)]}
check("one prediction cannot satisfy two annotations",
      E.evaluate(truth2, preds2, abstain_id=ABSTAIN), dict(correct=1, localised=1))

print("\nPASS" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
