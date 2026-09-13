"""PatHUgen — TEKNOFEST 2026 Health AI, final delivery package.

Everything a competition-day delivery needs (data loading, feature
engineering, the XGBoost model, prediction, result-JSON assembly, and
delivery packaging) lives here. This package is fully self-contained: it
imports only the standard library plus numpy/pandas/joblib/xgboost and
scikit-learn (which xgboost's classifier API already requires anyway), never
anything from `search/` (the development-only research pipeline).

The champion configuration itself (which features, hyperparameters, and
threshold each panel uses) is NOT hardcoded in this package — see
`pathugen/champion_config.py`.
"""
from __future__ import annotations
