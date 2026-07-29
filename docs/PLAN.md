# Plan: dzetsaka Adoption + Reproducibility Roadmap (Execution Version)

## Objective
Ship an offline-first "Recipe + Autopilot + Trust" workflow that improves first-run success,
reproducibility, and community reuse without breaking current classification behavior.

## Current Status (as of 2026-02-07)
- [ ] Phase 0 complete: baseline and architecture decisions documented
- [ ] Phase 1 complete: recipe schema v2 and compatibility layer
- [ ] Phase 2 complete: trust artifacts and reproducibility bundle
- [ ] Phase 3 complete: local recommender and QGIS processing entrypoint
- [ ] Phase 4 complete: recipe feed/offline cache and benchmark harness
- [ ] Phase 5 complete: docs, onboarding flow, and release hardening

## Success Metrics
1. Median time-to-first-valid-map for new users: `< 10 min`.
2. 90-day retention: release-over-release increase.
3. Runs using saved/shared recipes after 3 months: `>= 40%`.
4. Behavioral parity with v5.0.0 on reference datasets.

## Scope
In scope:
- Versioned recipe format and migration
- Local recommendation flow (no cloud hard dependency)
- Reproducibility artifacts (`run_manifest.json`, `trust_card.json`)
- Offline-capable recipe cache + benchmark workflow
- Onboarding and docs updates

Out of scope (this wave):
- Mandatory cloud inference
- Deep-learning segmentation track
- Breaking changes to existing processing APIs

## Repo-Aligned Implementation Plan
### Phase 0: Baseline and RFC (1 sprint)
Deliverables:
- Freeze reference behavior on selected datasets.
- Define recipe v2 schema and migration rules.
- Define trust-card contract and acceptance checks.

Code targets:
- `scripts/` for reference run automation
- `docs/` for RFC + acceptance checklist
- `tests/integration/` baseline parity fixtures

Acceptance:
- Reproducible baseline artifacts committed and documented.
- Explicit compatibility matrix for recipe v1 -> v2.

### Phase 1: Recipe Schema v2 + Migration (1-2 sprints)
Deliverables:
- Recipe model with `schema_version`, provenance, constraints, compatibility.
- Loader that accepts v1 and migrates to v2 at read time.
- Signature verification primitive (local verification only).

Code targets:
- `domain/` recipe entities/value objects
- `services/` recipe migration + validation
- `factories/` recipe serialization/deserialization
- `tests/unit/` schema, migration, signature validation

Acceptance:
- v1 recipes load successfully through migration adapter.
- Invalid signatures are surfaced with actionable errors.

### Phase 2: Trust Artifacts + Repro Bundle (1 sprint)
Deliverables:
- Emit `run_manifest.json` and `trust_card.json` for each classification run.
- Include lineage: data source, parameters, model type, validation mode.

Code targets:
- `processing/` run orchestration hooks
- `services/` artifact writer
- `tests/integration/` run -> artifact validation

Acceptance:
- Replay of run manifest reproduces equivalent outputs within defined tolerance.

### Phase 3: Local Autopilot Recommender (1-2 sprints)
Deliverables:
- Dataset fingerprinting (bands, sample size, label distribution, nodata profile).
- Ranked local recommendations with dependency filtering.
- New processing algorithm: `Recommend Recipe`.

Code targets:
- `services/` fingerprint + ranking
- `processing/` algorithm registration and output contract
- `dzetsaka_provider.py` provider exposure
- `tests/unit/` deterministic ranking tests

Acceptance:
- Deterministic recommendation order for fixed input.
- Recommender works fully offline.

### Phase 4: Community Reuse Layer (1 sprint)
Deliverables:
- Remote feed ingestion with robust local cache fallback.
- Benchmark harness to compare recipe performance reproducibly.

Code targets:
- `services/` feed client + cache
- `scripts/` benchmark execution and report generation
- `tests/integration/` cache/offline behavior

Acceptance:
- Cached recipes are discoverable and runnable with network disabled.

### Phase 5: UX + Release Hardening (1 sprint)
Deliverables:
- Guided first-run flow for "choose recipe -> run -> inspect trust card".
- Feature flags for safe rollout.
- Docs and changelog updates for plugin release.

Code targets:
- `ui/` guided workflow components
- `docs/` user guide and migration notes
- `metadata.txt`, `CHANGELOG.md`

Acceptance:
- Quick-start flow validated in manual QA and non-QGIS test path.

## Test Plan
1. Unit:
- Schema validation, migration edge cases, signature verification, ranking determinism.
2. Integration:
- Recommend -> run -> emit artifacts -> replay/import.
3. Regression:
- Existing processing algorithms and guided workflow behavior.
4. Non-functional:
- Offline mode checks for recommender and cache.

Suggested commands:
- `make quick-test`
- `pytest tests/unit/`
- `pytest tests/integration/ -m "not qgis"`

## Risks and Mitigations
- Recipe drift across releases:
  Mitigation: schema versioning + migration tests per release.
- Non-deterministic recommendation quality:
  Mitigation: fixed tie-break rules and locked fixtures.
- User confusion around trust metrics:
  Mitigation: constrained trust-card vocabulary + docs examples.

## Definition of Done
- All phase acceptance criteria checked.
- Parity tests pass against v5.0.0 reference datasets.
- Documentation includes:
  - recipe authoring
  - migration behavior
  - trust-card interpretation
  - offline usage expectations

## Next Implementation Slice
1. Land Phase 0 RFC + baseline dataset harness.
2. Implement recipe schema v2 + v1 migration adapter.
3. Add unit tests for schema/migration before recommender work.
