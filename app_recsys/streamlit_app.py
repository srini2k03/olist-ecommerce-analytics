"""Demo UI for the H&M recommender API (not part of the Olist stlite dashboard).

Run the API first (``make recsys-serve``), then ``make recsys-demo``.
The API address comes from ``RECSYS_API_URL`` (default http://localhost:8000).
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("RECSYS_API_URL", "http://localhost:8000").rstrip("/")
NEW_USER = "New user (no history)"
ATTR_COLUMNS = ["prod_name", "product_type_name", "colour_group_name", "index_group_name"]


def api_get(path: str, **params) -> dict:
    resp = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


st.set_page_config(page_title="H&M recommender demo", layout="wide")
st.title("H&M recommender demo")

try:
    health = api_get("/health")
except requests.RequestException as exc:
    st.error(f"Cannot reach the API at {API_URL}: {exc}")
    st.stop()

if health["data"] == "synthetic":
    st.warning(
        "This API serves a model trained on SYNTHETIC data: the items are not real H&M articles."
    )
st.caption(
    f"Model state as of {health['as_of']} · {health['users_with_embeddings']:,} customers with "
    f"embeddings · {health['index_items']:,} indexed articles · cold-start path: "
    f"{health['cold_user_path']}"
)

sample = api_get("/users/sample", n=25)["customer_ids"]
with st.sidebar:
    choice = st.selectbox("Customer", [NEW_USER, *sample])
    age = st.slider("Age (new users only)", 16, 90, 30) if choice == NEW_USER else None
    k = st.slider("Recommendations", 4, 48, 12, step=4)

customer = "0" if choice == NEW_USER else choice
params = {"k": k} | ({"age": age} if age is not None else {})
recs = api_get(f"/recommend/{customer}", **params)

left, right = st.columns(2)
with left:
    st.subheader("Recent purchases")
    history = [] if choice == NEW_USER else api_get(f"/users/{customer}/history")["purchases"]
    if history:
        st.dataframe(pd.DataFrame(history)[["t_dat", "article_id", *ATTR_COLUMNS]], hide_index=True)
    else:
        st.info("No purchase history: the cold-start path is used.")
with right:
    st.subheader("Recommendations")
    st.caption(f"Strategy: {recs['strategy']} · {recs['latency_ms']:.0f} ms in the service")
    table = pd.DataFrame(recs["items"])
    st.dataframe(table[["article_id", "score", "sources", *ATTR_COLUMNS]], hide_index=True)

with st.expander("Article details"):
    article = st.selectbox("Article", [i["article_id"] for i in recs["items"]])
    st.json(api_get(f"/items/{article}"))
