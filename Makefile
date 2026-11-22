PY ?= python
NOTEBOOKS := 01_eda 02_stats 03_segmentation 04_churn

.PHONY: setup data warehouse postgres sql sql-compare notebooks app-data app test lint smoke site all \
	recsys-setup recsys-data recsys-baselines recsys-retrieval recsys-ranking recsys-coldstart recsys-report recsys-all \
	recsys-serve recsys-bench recsys-demo recsys-docker recsys-synthetic

setup:            ## install everything (editable package + dev tools)
	$(PY) -m pip install -r requirements-dev.txt -e .

data:             ## download the 9 Olist CSVs (Kaggle if KAGGLE_* set, else pinned mirror)
	$(PY) scripts/fetch_data.py

warehouse: data   ## build data/olist.duckdb
	$(PY) scripts/build_warehouse.py

postgres:         ## also load the tables into $$OLIST_PG_DSN
	$(PY) scripts/build_warehouse.py --postgres

sql:              ## run all 16 queries on DuckDB and snapshot results/sql/*.csv
	$(PY) scripts/run_sql.py

sql-compare:      ## run all queries on DuckDB and Postgres and require identical results
	$(PY) scripts/run_sql.py --postgres --no-snapshot

notebooks:        ## execute every notebook in place (real outputs)
	cd notebooks && for nb in $(NOTEBOOKS); do \
		$(PY) -m jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.record_timing=False $$nb.ipynb || exit 1; \
	done

app-data:         ## rebuild the small artefacts the Streamlit app reads
	$(PY) scripts/build_app_data.py

app:              ## run the dashboard locally
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

smoke:            ## streamlit run + Playwright end-to-end check
	$(PY) scripts/smoke_app.py

site:             ## build the GitHub Pages site into site/
	$(PY) scripts/build_site.py

all: warehouse sql notebooks app-data test

# ---- Recommender (H&M data, see docs/recsys.md) ----------------------------------------------

recsys-setup:     ## install the recommender's dependencies (CPU-only torch, FAISS, Kaggle CLI)
	$(PY) -m pip install -r requirements-torch-cpu.txt
	$(PY) -m pip install -r requirements-dev.txt -r requirements-recsys.txt -e .

recsys-data:      ## H&M: download (needs KAGGLE_* and accepted rules), convert, subset, split
	$(PY) -m recsys fetch
	$(PY) -m recsys subset
	$(PY) -m recsys split

recsys-baselines: ## evaluate every baseline on the validation week -> results/recsys/
	$(PY) -m recsys baselines --split validation

recsys-retrieval: ## train the two-tower model, build FAISS indexes -> results/recsys/
	$(PY) -m recsys retrieval

recsys-ranking:   ## candidates + features per fold, LambdaRank, ablation -> results/recsys/
	$(PY) -m recsys ranking

recsys-coldstart: ## cold users (blend chosen on validation) and new articles -> results/recsys/
	$(PY) -m recsys coldstart

recsys-report:    ## test-week report of every stage + simulated A/B test -> results/recsys/
	$(PY) -m recsys baselines --split test
	$(PY) -m recsys report

recsys-all:       ## every modelling stage after split, then the README section
	$(PY) -m recsys all
	$(PY) -m recsys readme

recsys-serve:     ## run the API on the exported bundle (data/hm/artifacts/serving)
	$(PY) -m uvicorn recsys.serve.api:app --port 8000

recsys-bench:     ## p50/p95/p99 latency of the in-process service
	$(PY) -m recsys.serve.benchmark --n 1000

recsys-demo:      ## Streamlit demo against the running API ($$RECSYS_API_URL)
	$(PY) -m streamlit run app_recsys/streamlit_app.py

recsys-docker:    ## build the CPU-only API image
	docker build -f docker/recsys.Dockerfile -t recsys-api .

recsys-synthetic: ## the whole recommender pipeline on synthetic data (no download)
	$(PY) -m recsys --synthetic
