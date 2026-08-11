# Cabify Data Science Challenge

Two independent exercises. The first is an experiment design: deciding whether a food delivery
platform should keep funding a free professional photography service for its restaurants. The second
is a model prototype: detecting, at scale, when the route a driver actually took differs from the
route a journey was priced on.

They share an approach rather than a dataset. In both cases the interesting work turned out to sit
upstream of the modelling, in deciding what exactly was being estimated and what the number would be
used for.

The original brief is in [`ds_challenge_description_cabify.md`](ds_challenge_description_cabify.md).

---

## Part 1: Experiment design

**The question.** A free professional photography service grew into a multimillion-euro operation.
Early analysis showed that restaurants with professional photos sold more and charged more. Is it
still worth funding?

**The short answer.** The question cannot be answered from the existing evidence, and the reason is
not sample size. Restaurants *request* the photographer, so adopters are self-selected, and they tend
to ask precisely when demand is already rising. In simulation that turns a true +8% effect into a
reported +60.6%; difference-in-differences only claws it back to +13.9%.

**What the notebook argues.**

| | |
|---|---|
| The decision is a portfolio problem, not a per-restaurant test | Written per shoot, the break-even inequality silently assumes the programme has no fixed cost. Once annual overhead is included, whether to keep the service and how widely to offer it become different questions |
| One unmeasured parameter decides the answer | On a marketplace, a treated restaurant's gain is partly demand taken from an untreated rival on the same platform. In simulation a restaurant-level A/B test reports +11.9% where the cluster as a whole gains +3.7%, a 3.2x overstatement. The service breaks even only if incrementality exceeds roughly 37%, and it has never been measured |
| Five randomised tracks, one decision | Restaurant-level encouragement (three arms, to separate photography from the sales contact that delivers it), saturation randomisation over co-consideration clusters, a geo switch-off, a customer-side listing test, and cost-effectiveness arms including a self-serve toolkit and a co-pay |
| "Keep or kill" is the wrong question | Simulated returns run from +€980 to -€920 per shoot. Serving everyone loses €1.7m a year; serving the best-returning 44% generates €20.5m before overhead. Whether that survives overhead depends on a figure that was never provided |

**A note on the numbers.** Every figure in Part 1 comes from a simulation written to show what each
design choice buys and what the output will look like when real data arrives. They are not findings.

---

## Part 2: Model prototyping

**The question.** Given an estimated route and the route actually driven, decide whether a human
would call them the same. Ground truth comes from a labelling exercise: 3,027 annotations covering
2,386 journeys, from 8 annotators across 28 cities in 7 countries.

**The short answer.** Ship a calibrated logistic regression behind a three-way decision rule.
Automate 75.5% of journeys at 97.9% accuracy and route the rest to human review.

**Headline results.**

| | |
|---|---|
| Discrimination | ROC-AUC 0.9902, PR-AUC 0.9931, Brier 0.0402, out of fold |
| Model search | Ten candidates (elastic net, three tree ensembles, random forest, RBF SVM, two neural nets) under two cross-validation schemes. None beat a default logistic regression on both discrimination and calibration |
| Raw geometry | A neural network on normalised raw polylines loses 0.12 ROC-AUC, so at this sample size the 34 hand-engineered features are doing real work |
| Hyperparameters | Not tuned. The regularisation objective is flat across two orders of magnitude and nested cross-validation makes the model marginally worse |
| Human ceiling | The same annotator re-shown the same journey changes their answer 11.5% of the time; two annotators agree 92.7%; the model agrees with the consensus 94.8% |
| Generalisation | Held-out-city ROC-AUC between 0.982 and 0.996, with grouped and random CV giving the same answer |
| Operating point | Moving from the F1-optimal threshold to a cost-derived three-way band takes missed divergences from 78 to 0 while leaving needless flags at 35 against 37 |

**Two decisions that mattered more than the algorithm.** Treating "I don't know" as evidence rather
than missing data, counted as half a vote, so that 174 genuinely ambiguous journeys train the model
instead of being dropped. And deriving the decision threshold from the cost of each error rather than
from F1, which was the single largest improvement in the notebook and required no retraining.

Start with [`Part_2_Technical_Report.md`](Part%202%20Model%20Prototyping/Part_2_Technical_Report.md)
for a standalone summary, or the notebook for the full argument.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── ds_challenge_description_cabify.md      the original brief
├── Part 1 Experiment Design/
│   ├── Part_1_Experiment_Design.ipynb      the design, argued out with simulations
│   ├── Part_1_Experiment_Design.html       rendered, no execution needed
│   ├── Figures/                            six figures, written by the notebook
│   └── je_style.py                         shared plotting style
└── Part 2 Model Prototyping/
    ├── Part_2_Model_Prototyping.ipynb      data, features, models, evaluation, handover
    ├── Part_2_Model_Prototyping.html       rendered, no execution needed
    ├── Part_2_Technical_Report.md          standalone summary with embedded figures
    ├── Data/challenge_dataset.json         3,027 annotations
    ├── Figures/                            nine figures, written by the notebook
    ├── Model/
    │   ├── route_divergence_model.joblib   fitted pipeline, features, decision config
    │   └── route_divergence_config.json    the same metadata without the model
    ├── p2_lib.py                           geometry, features, Dawid-Skene, city assignment
    └── je_style.py                         shared plotting style
```

The HTML exports are the fastest way in: they carry every figure and output, and need nothing
installed.

---

## Reproducing

Python 3.12.

```bash
pip install -r requirements.txt
```

`lightgbm` and `xgboost` are used only in the Part 2 model search; everything else runs without them.
The exact versions the results were produced with are noted in `requirements.txt`.

Each notebook resolves its paths relative to its own folder, so run it from there and nothing needs
editing:

```bash
cd "Part 2 Model Prototyping" && jupyter lab Part_2_Model_Prototyping.ipynb
```

Part 2 reads `Data/challenge_dataset.json` and rewrites `Figures/` and `Model/` as it runs, which
takes about five minutes end to end. Part 1 rewrites its own `Figures/` and takes under a minute,
since its simulations are small. Both are committed already, so nothing has to be executed to read
the work.

Loading the fitted model directly:

```python
import joblib
art = joblib.load("Model/route_divergence_model.joblib")
p = art["model"].predict_proba(features[art["features"]])[:, 1]
cfg = art["decision"]     # accept_below, flag_above, and the costs they were derived from
```

The artefact carries the feature names in order, the decision configuration, the out-of-fold metrics,
a hash of the feature module and the library versions it was built against.

---

## Scope and caveats

- Part 1 is a **design**, not a result. Its numbers are simulated to demonstrate what each choice
  buys, and are labelled as such throughout.
- Part 2's metrics are computed on the 2,212 journeys where the annotators reached a decisive
  majority. The 174 ambiguous journeys train the model but cannot be scored against it, so every
  headline figure describes a marginally easier population than production would.
- "Zero missed divergences" is a measurement on 2,212 journeys, not a guarantee. It means the
  residual risk sits below what this dataset can resolve.
- The cost ratio behind the Part 2 operating point (a missed divergence costs 12 times a needless
  flag) is a stated assumption, not a measurement. The sensitivity of the recommendation to it is
  shown in the notebook.
