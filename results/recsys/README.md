# Recommender results

Baseline (and, later, model) results on the real H&M data are written here as
`baselines_<split>.json` by `make recsys-baselines` after the data has been
downloaded with `make recsys-data` (see [docs/recsys.md](../../docs/recsys.md)).

Nothing has been run on the real data yet, so there are no results in this
folder. Synthetic runs (`python -m recsys --synthetic`) write to the gitignored
`data/hm/synthetic/results/` instead and are never committed: their numbers
describe the data generator, not H&M.
