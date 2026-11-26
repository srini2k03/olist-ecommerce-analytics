#!/usr/bin/env bash
# One-time setup so GitHub Actions can load the Olist marts into the BigQuery SANDBOX and run
# the analytics queries there. The sandbox is free and needs no billing account; this script
# refuses to continue if the project has billing enabled, so nothing it sets up can cost money.
#
# Run in Cloud Shell (console.cloud.google.com -> the ">_" icon):
#   bash setup_bigquery_sandbox.sh YOUR_PROJECT_ID
#
# It creates: the "olist" dataset (US), a service account that can only run BigQuery jobs and
# write tables, and a Workload Identity Federation pool that only the olist-ecommerce-analytics
# GitHub repository can use. No keys are created. It is safe to re-run.
set -euo pipefail

PROJECT_ID=${1:?usage: bash setup_bigquery_sandbox.sh PROJECT_ID}
GITHUB_REPO=srini2k03/olist-ecommerce-analytics
DATASET=olist
LOCATION=US
POOL=github
PROVIDER=github-olist
SA_NAME=github-bigquery

gcloud config set project "$PROJECT_ID" >/dev/null
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
SA="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

echo "Checking that billing is OFF (BigQuery sandbox)"
gcloud services enable cloudbilling.googleapis.com >/dev/null
BILLING=$(gcloud billing projects describe "$PROJECT_ID" --format='value(billingEnabled)')
if [ "$BILLING" != "False" ] && [ "$BILLING" != "false" ]; then
  cat <<MSG

STOPPED: billing is enabled on $PROJECT_ID (billingEnabled=$BILLING), so this would not be the
free sandbox. Disable it first, then run this script again:
  https://console.cloud.google.com/billing/projects -> find $PROJECT_ID -> Actions (three dots)
  -> Disable billing
MSG
  exit 1
fi
echo "Billing is off: BigQuery runs in sandbox mode (free, 10 GB storage, 1 TB queries a month)."

echo "Enabling APIs"
gcloud services enable bigquery.googleapis.com iam.googleapis.com \
  iamcredentials.googleapis.com sts.googleapis.com

echo "Dataset $DATASET ($LOCATION)"
bq --location="$LOCATION" show --dataset "$PROJECT_ID:$DATASET" >/dev/null 2>&1 ||
  bq --location="$LOCATION" mk --dataset --description "Olist marts (loaded by GitHub Actions)" \
    "$PROJECT_ID:$DATASET"

echo "Service account (BigQuery jobs and table writes only)"
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 ||
  gcloud iam service-accounts create "$SA_NAME" --display-name="GitHub Actions BigQuery loader"
for role in roles/bigquery.jobUser roles/bigquery.dataEditor; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$SA" \
    --role="$role" --condition=None >/dev/null
done

echo "Workload Identity Federation (GitHub OIDC, restricted to $GITHUB_REPO)"
gcloud iam workload-identity-pools describe "$POOL" --location=global >/dev/null 2>&1 ||
  gcloud iam workload-identity-pools create "$POOL" --location=global --display-name="GitHub"
gcloud iam workload-identity-pools providers describe "$PROVIDER" --location=global \
  --workload-identity-pool="$POOL" >/dev/null 2>&1 ||
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location=global \
    --workload-identity-pool="$POOL" --display-name="GitHub Actions (olist)" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='$GITHUB_REPO'"
gcloud iam service-accounts add-iam-policy-binding "$SA" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository/$GITHUB_REPO" >/dev/null

cat <<OUT

Done. Add these three repository VARIABLES (not secrets) on GitHub:
  https://github.com/$GITHUB_REPO/settings/variables/actions/new

  GCP_PROJECT_ID       = $PROJECT_ID
  GCP_WIF_PROVIDER     = projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/providers/$PROVIDER
  GCP_SERVICE_ACCOUNT  = $SA

Then run the "BigQuery" workflow (Actions tab -> BigQuery -> Run workflow).
OUT
