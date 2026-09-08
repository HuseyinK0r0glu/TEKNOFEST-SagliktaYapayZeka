#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PatHUgen — TEKNOFEST 2026 Health AI Competition, Final Delivery System.

Team: PatHUgen  |  Task: missense variant pathogenicity classification
Panels (always independent): GENEL, KANSER, PAH, CFTR

WHAT THIS FILE IS
------------------
This is the entry point. The implementation lives in the `pathugen/`
package next to this file. For each of the four panels, an automated
search compared many combinations of feature set, model family, and
decision-threshold strategy, validated every candidate with leak-free
nested cross-validation, and screened out any candidate that showed signs
of overfitting (a large gap between its optimistic and honest scores, an
unstable choice of "best" model across folds, or failure to beat a naive
baseline). The configuration that survived that process for each panel is
recorded in `champion_config.json` (see `Champion_Config.md`) — not as a
literal in this code.

TRAIN and DELIVER never search anything — those decisions are closed and
fitting/predicting with them is fast and fully self-contained. What they
need is exactly what a competition-day delivery requires:

    1. TRAIN   — fit the champion configuration from the raw training CSVs
                 (run this once, ahead of time, not during the timed test).
    2. DELIVER — given the organization's test CSVs, produce predictions,
                 validate them against the official JSON schema, and package
                 a complete, reproducible delivery folder (code + model +
                 requirements + README) — this is the command run on
                 competition day, once per official/practice submission.

Every panel's winning configuration turned out to be a single tuned XGBoost
model — no averaging or stacking of multiple models beat a single well-tuned
one on any panel once overfitting was accounted for — which is why TRAIN and
DELIVER only need XGBoost, not a larger zoo of model families.

A third command exists for a genuinely different purpose:

    3. RUN     — redo the search itself: try many feature-set / model /
                 threshold combinations per panel, entirely from scratch
                 (no cached or inherited hyperparameters), and keep only the
                 ones that pass an overfitting audit, to find a NEW champion
                 configuration (e.g. if you get new or materially different
                 training data). This is a multi-hour research operation,
                 not a competition-day command, and — unlike TRAIN and
                 DELIVER — it is NOT self-contained: it delegates to this
                 project's search pipeline (the `search/` package) and only
                 works when that sits alongside this file. RUN never
                 changes champion_config.json automatically; adopting a new
                 result found by RUN is a deliberate, separate step.

JSON OUTPUT FORMAT
-------------------
Per `TEKNOFEST_2026_Saglikta_Yapay_Zeka_Final_Uygulama_ve_Sonuc_Teslim_Kilavuzu.pdf`,
section 3.2 (the "University and Above Level" JSON format):

    {
      "team_name": "...", "team_id": "...", "application_id": "...",
      "competition_level": "UNIVERSITE_VE_UZERI",
      "predictions": [
        {"id": "VAR_000001", "panel": "CFTR",
         "predicted_class": "1", "predicted_prob": 0.87}
      ]
    }

predicted_class is the string "0" or "1" (the team's final decision, using
the panel's audited threshold). predicted_prob is the model's probability of
the Pathogenic (1) class, used by the organizers for AUROC-style metrics.
This file's `predicted_class` and `predicted_prob` fields follow section 3.2's
worked example exactly; if the official rules are amended before the
competition, re-check this against the latest guide.

IMPORTANT — VERIFY BEFORE THE COMPETITION
-------------------------------------------
TEAM_NAME / TEAM_ID / APPLICATION_ID (in `pathugen/config.py`) are filled
from this project's existing paperwork (PSR report cover). Confirm they
match your KYS (application system) registration exactly before your first
real submission — override with environment variables if anything changed:

    TEAM_NAME=... TEAM_ID=... APPLICATION_ID=... python PatHUgen.py deliver ...

Usage
-----
    # One-time, ahead of the competition (needs the training CSVs):
    python PatHUgen.py train

    # Every time you have test data to score (competition day):
    python PatHUgen.py deliver --input-dir /path/to/test/csvs

    # See HOW_TO_RUN.md for the full walkthrough.
"""
from __future__ import annotations

from pathugen.cli import main

if __name__ == "__main__":
    main()
