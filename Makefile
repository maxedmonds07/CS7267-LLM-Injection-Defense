SHELL := /bin/bash
.DEFAULT_GOAL := help

DVC := uv run dvc
REMOTE := r2
R2_PREFIX ?= dvcstore

# NewsQA sources (see config.yaml `generated.bipia_webqa`); both come with terms of use.
NEWSQA_CACHE := .cache/newsqa
NEWSQA_ZIP_URL := https://download.microsoft.com/download/1/d/8/1d830cee-f8d1-4807-9224-de35a8f08dc4/newsqa-data-v1.zip
CNN_STORIES_DRIVE_ID := 0BwmD_VLjROrfTHk4NFg2SndKcjQ
NEWSQA_CSV := data/external/newsqa/combined-newsqa-data-v1.csv
NEWSQA_COMMIT := d5bb9e9640e2ed7a31e209393376549d737d276b
NEWSQA_IMAGE := bryant1410/newsqa@sha256:be80e12652517a01bded32156578abe406f5bbb1f643350f46f6007c6be65423

.PHONY: help install r2-remote r2-credentials check-remote data push-data newsqa

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

# One-time, by one teammate: everyone else gets the CSV from R2 via `make data`.
# The bipia_webqa stage is registered here, not up front, so `dvc repro` never sees a
# stage whose input doesn't exist yet. Commit the resulting .dvc, dvc.yaml and dvc.lock.
# Downloads need ACCEPT_TERMS=1, confirming you accept the terms of:
#   NewsQA       https://www.microsoft.com/en-us/download/details.aspx?id=57162 (LICENSE.pdf in the zip)
#   CNN stories  https://cs.nyu.edu/~kcho/DMQA/
newsqa: $(NEWSQA_CACHE)/newsqa-data-v1.tar.gz $(NEWSQA_CACHE)/cnn_stories.tgz ## Build the NewsQA CSV, track it with DVC, and add the bipia_webqa stage (ACCEPT_TERMS=1)
	rm -rf $(NEWSQA_CACHE)/repo && mkdir -p $(NEWSQA_CACHE)/repo
	curl -sSfL https://codeload.github.com/Maluuba/newsqa/tar.gz/$(NEWSQA_COMMIT) | tar xz --strip-components=1 -C $(NEWSQA_CACHE)/repo
	cp $(NEWSQA_CACHE)/newsqa-data-v1.tar.gz $(NEWSQA_CACHE)/cnn_stories.tgz $(NEWSQA_CACHE)/repo/maluuba/newsqa/
	docker run --rm --platform linux/amd64 -v $(abspath $(NEWSQA_CACHE)/repo):/usr/src/newsqa $(NEWSQA_IMAGE)
	mkdir -p $(dir $(NEWSQA_CSV))
	cp $(NEWSQA_CACHE)/repo/combined-newsqa-data-v1.csv $(NEWSQA_CSV)
	$(DVC) add $(NEWSQA_CSV)
	$(DVC) stage add --force --name bipia_webqa \
		--deps src/data/generate_bipia.py --deps src/data/fetch.py \
		--deps data/raw/bipia/benchmark/qa --deps $(NEWSQA_CSV) \
		--params config.yaml:paths.generated_dir,generated.bipia_webqa \
		--outs data/generated/bipia/qa \
		python src/data/generate_bipia.py --config config.yaml --task bipia_webqa
	$(DVC) repro bipia_webqa

define require_terms
	@test "$(ACCEPT_TERMS)" = 1 || { echo "Downloading $(@F) requires ACCEPT_TERMS=1 (see the newsqa target's comment)"; exit 1; }
endef

$(NEWSQA_CACHE)/newsqa-data-v1.zip:
	$(require_terms)
	mkdir -p $(@D)
	curl -sSfL -o $@.part $(NEWSQA_ZIP_URL) && mv $@.part $@

# Maluuba's container only reads the tarball (its CMD deletes an extracted newsqa-data-*.csv),
# while Microsoft now ships a zip; repack the CSV at the tarball root.
$(NEWSQA_CACHE)/newsqa-data-v1.tar.gz: $(NEWSQA_CACHE)/newsqa-data-v1.zip
	unzip -o -j -d $(@D) $< newsqa-data-v1/newsqa-data-v1.csv
	tar czf $@ -C $(@D) newsqa-data-v1.csv
	rm $(@D)/newsqa-data-v1.csv

# gdown handles Drive's virus-scan confirmation page for large files, which plain curl saves instead.
$(NEWSQA_CACHE)/cnn_stories.tgz:
	$(require_terms)
	mkdir -p $(@D)
	uvx gdown@6.4.0 $(CNN_STORIES_DRIVE_ID) -O $@.part && mv $@.part $@
