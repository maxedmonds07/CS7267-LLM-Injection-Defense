SHELL := /bin/bash
.DEFAULT_GOAL := help

DVC := uv run dvc
REMOTE := r2
R2_PREFIX ?= dvcstore

.PHONY: help install r2-remote r2-credentials check-remote data push-data

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install: ## Install project and dev dependencies
	uv sync

# One-time: writes the (non-secret) remote location to .dvc/config, which is committed.
r2-remote: ## Configure the R2 remote: make r2-remote R2_ACCOUNT_ID=... R2_BUCKET=...
	@test -n "$(R2_ACCOUNT_ID)" || { echo "R2_ACCOUNT_ID is required"; exit 1; }
	@test -n "$(R2_BUCKET)" || { echo "R2_BUCKET is required"; exit 1; }
	$(DVC) remote add --default --force $(REMOTE) s3://$(R2_BUCKET)/$(R2_PREFIX)
	$(DVC) remote modify $(REMOTE) endpointurl https://$(R2_ACCOUNT_ID).r2.cloudflarestorage.com
	$(DVC) remote modify $(REMOTE) region auto

# Per machine: writes keys to .dvc/config.local, which is git-ignored.
# Reads R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY from the environment, else prompts without echoing.
r2-credentials: ## Store R2 API token keys locally (never committed)
	@key_id="$${R2_ACCESS_KEY_ID}"; secret="$${R2_SECRET_ACCESS_KEY}"; \
	if [ -z "$$key_id" ]; then read -rsp "R2 Access Key ID: " key_id; echo; fi; \
	if [ -z "$$secret" ]; then read -rsp "R2 Secret Access Key: " secret; echo; fi; \
	$(DVC) remote modify --local $(REMOTE) access_key_id "$$key_id" && \
	$(DVC) remote modify --local $(REMOTE) secret_access_key "$$secret" && \
	echo "Credentials written to .dvc/config.local"

check-remote: ## Verify the remote is reachable and compare cache with it
	$(DVC) remote list
	$(DVC) status --cloud

data: ## Pull data from R2 and rebuild any stale pipeline stages
	$(DVC) pull
	@if [ -f dvc.yaml ]; then $(DVC) repro; else echo "No dvc.yaml yet; skipping repro"; fi

push-data: ## Upload tracked data to R2
	$(DVC) push
