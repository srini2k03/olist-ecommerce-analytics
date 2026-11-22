"""Two-stage fashion recommender on the H&M Personalized Fashion Recommendations data.

Data pipeline, time-based split with leakage checks, baselines, two-tower
retrieval, a LambdaRank ranker, cold start, evaluation and serving.
"""

import os

# PyTorch and LightGBM each bring an OpenMP runtime. With the default (active)
# wait policy, idle LightGBM threads spin and slow later PyTorch work in the same
# process several-fold; passive waiting avoids that. Must be set before either
# library initialises OpenMP, i.e. before they are imported.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

__version__ = "0.1.0"
