# triton 3.0 — newt re-identification by belly pattern

Open-set fine-grained re-ID: a photo of a newt's belly → which individual from the database it is
(with a calibrated score and a visual justification) or a new individual. Master's thesis,
"Applied AI" program (TSU); data provided by the Severtsov Institute of Ecology and Evolution (IPEE RAS).

The first master's-thesis version of the project is complete: the core pipeline blocks are implemented
(data → segmentation and cropping → unrolling → embedder → spot matching → hybrid and demo), with the
final evaluation performed on a once-opened sealed test. The project continues to evolve as an open
research / engineering system for newt re-identification; further work includes improving temporal re-ID,
expanding the test suite, documentation, reproducible releases, and the demo.

## Architecture

Crop of the **belly only** (YOLO seg + pose) → orientation (head/tail) → **ribbon unrolling**
(`unroll_ribbon`) → zero-shot **MegaDescriptor-L-384**, cosine top-K search (numpy; FAISS — a hook for scaling) →
matching the constellation of spot centroids — produces an **overlay of matched spots** (an interpretable
justification of the decision) → calibrated %-score + known/new decision.

Diagrams (use case, component, deployment, sequence, etc.) — in `docs_public/uml/`.

## Installation

```bash
pip install -r requirements.txt   # dependencies (versions pinned to a verified environment)
pip install -e .                  # the package itself (editable); does NOT install deps on purpose — see the comment in pyproject.toml
pip install -e ".[crop]"          # optional (Block 2): ultralytics / scikit-image — segmentation and cropping
```

Weights are not included in the repository: MegaDescriptor-L-384 is downloaded from the HF Hub on first
run; the trained YOLO seg/pose weights (~6.5/6.1 MB) will be published separately (GitHub Releases / HF Hub).

## Quick start

```bash
streamlit run app/demo.py             # demo in the browser → http://localhost:8501 (details — app/README.md)
python -m triton_data.cli build       # build the manifest from raw data (paths — per configs/paths.example.yaml)
python -m triton_data.cli validate    # strict data-quality gates + EDA report
pytest -q                             # project tests
```

The customer's photo data is not included in the repository and is not published.

## Demonstration

Demo (`streamlit run app/demo.py`, the "Identify" tab): upload a belly photo — the system segments and
unrolls the pattern, searches for the nearest individuals, and shows the result.

- **Input:** a belly photo (JPG/PNG).
- **Output:** a ranked list of top-K candidates with a similarity score (0–100 %), a "known / new individual"
  decision, and an **overlay of matched spots** as a visual justification.
- **Example:** query → individual `PW-005`, 99 % confidence; overlay — 15 matched spots (75 % share).

Demonstration screenshots contain real belly crops of individuals (customer data) and are not included in
the public repository. The interaction and data-flow diagrams are in `docs_public/uml/`
(use case, sequence, component); the demonstration is reproduced locally with the command above.

## Results (sealed-test, kpi_core)

The final system — zero-shot MegaDescriptor-L-384 + `unroll_ribbon` — was fixed on dev **before** the test
was opened; the sealed test was opened once, and the open-set threshold was calibrated on dev, not on the
test. Source of the numbers — `artifacts/ab_test_headline.json`.

| Slice | n | recall@1 | recall@5 |
|---|---|---|---|
| overall | 112 | 0.250 | 0.446 |
| PW (sharp-ribbed newt) | 11 | 0.818 | 1.000 |
| TK (Karelin's newt, temporal) | 101 | 0.188 | 0.386 |

- Gallery — 238 images; open-set AUROC 0.446 (known 112 / new 21).
- The PW and TK slices are not comparable to each other: PW — random/same-session enrollment,
  TK — a strict temporal protocol (recaptures across sessions).
- The project KPI (top-1 ≥ 75 % AND top-5 ≥ 95 %) is not met on overall; the bottleneck is the
  temporal TK slice: the same pipeline clears the bar on PW, i.e. the limitation is in the data
  (reproducibility of the pattern across sessions), not in the method.
- Summary of the numbers — `docs_public/ИТОГОВЫЕ_ЧИСЛА.md`; methodological caveats —
  `docs_public/МЕТОДОЛОГИЯ_оговорки.md`.
- Response time — about 0.6 s per photo (CPU/MPS, offline). The relevant quality metric for re-ID
  is recall@k (CMC) per individual, not detection precision/FPS.

## Layout

```
src/triton_data/   data layer: manifest, md5 dedup, deterministic splits, validation, CLI
src/triton_crop/   re-ID pipeline: segmentation/crop, unrolling, embedder, spot matching,
                   hybrid, sealed gates, demo backend
app/               Streamlit demo (app/demo.py; see app/README.md)
configs/           configs; data paths — per configs/paths.example.yaml
                   (the local configs/paths.yaml is not published)
tests/             TDD tests on synthetic mini-cohorts
docs_public/       public documentation: UML, final numbers, methodological caveats
notebooks/         working notebooks (training of auxiliary models)
data/, reports/, artifacts/   generated artifacts (not committed to git,
                   except the sealed-metrics summary artifacts/ab_test_headline.json)
```

## Principles

- A single canonical image reader (`imageio.py`) — one preprocessing everywhere.
- md5 dedup **before** the split (protection against gallery↔probe leakage).
- Deterministic splits (seed=42); temporal and random metrics — measured separately.
- The sealed test is opened once; reopening and tuning on the test are forbidden by CLI gates.

## Licenses

The project code — **GNU AGPL-3.0-or-later** (see `LICENSE`); the choice is driven by the
Ultralytics YOLO dependency (AGPL-3.0). The MegaDescriptor weights are distributed under CC-BY-NC-4.0 —
practical use of the system with them remains non-commercial.
The full list of third-party models, libraries, and data sources — `THIRD_PARTY_NOTICES.md`.

## Versions and materials

- Version and change history — [`CHANGELOG.md`](CHANGELOG.md).
- Dependencies — `requirements.txt` (pinned versions) and `pyproject.toml`.
- The research report and presentation are thesis materials; they contain customer data and are not
  made publicly available (available on request).
