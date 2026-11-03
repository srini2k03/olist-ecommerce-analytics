PY ?= python
NOTEBOOKS := 01_eda 02_stats 03_segmentation 04_churn

.PHONY: setup data warehouse postgres sql sql-compare notebooks app-data app test lint smoke site all

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
