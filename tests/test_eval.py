"""Check evaluate() counts the four outcomes correctly on a hand-built case."""
import sys
sys.path.insert(0, r"C:\Users\서동준\git\scripts")
import eval_realworld as E

names = [f"c{i}" for i in range(50)] + ["unknown_bottle"]
UNK = 50

# One image. Ground truth: two known bottles and one out-of-vocabulary bottle.
#   box A at x=0.1  class 0   -> model names it 0      (found, correct)
#   box B at x=0.3  class 1   -> model names it 2      (found, wrong name)
#   box C at x=0.5  unknown   -> model names it 3      (false alarm)
#   box D at x=0.7  class 4   -> no prediction         (missed entirely)
# Plus a prediction at x=0.9 matching nothing (pure false positive).
truth = {
    "a.jpg": (
        [(0, 0.1, 0.5, 0.06, 0.4), (1, 0.3, 0.5, 0.06, 0.4), (4, 0.7, 0.5, 0.06, 0.4)],
        [(UNK, 0.5, 0.5, 0.06, 0.4)],
    )
}
preds = {
    "a.jpg": [
        (0, 0.1, 0.5, 0.06, 0.4, 0.9),
        (2, 0.3, 0.5, 0.06, 0.4, 0.8),
        (3, 0.5, 0.5, 0.06, 0.4, 0.7),
        (7, 0.9, 0.5, 0.06, 0.4, 0.6),
    ]
}

got = E.evaluate(truth, preds, names)
expected = dict(known=3, localised=2, correct=1, unknown=1, unknown_hit=1, preds=4, matched_pred=3)

ok = True
for key, want in expected.items():
    mark = "ok " if got[key] == want else "FAIL"
    if got[key] != want:
        ok = False
    print(f"  {mark} {key:14} expected {want:3}  got {got[key]:3}")

# A prediction must not be reused: if two GT boxes overlap one prediction, only one matches.
truth2 = {"b.jpg": ([(0, 0.2, 0.5, 0.2, 0.6), (0, 0.22, 0.5, 0.2, 0.6)], [])}
preds2 = {"b.jpg": [(0, 0.21, 0.5, 0.2, 0.6, 0.9)]}
got2 = E.evaluate(truth2, preds2, names)
mark = "ok " if got2["correct"] == 1 else "FAIL"
if got2["correct"] != 1:
    ok = False
print(f"  {mark} one prediction cannot satisfy two ground-truth boxes: correct={got2['correct']}")

print("\nPASS" if ok else "\nFAILED")
sys.exit(0 if ok else 1)
