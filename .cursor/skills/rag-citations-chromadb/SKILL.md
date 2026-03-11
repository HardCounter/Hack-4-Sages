---
name: rag-citations-chromadb
description: >-
  Manages RAG citations via ChromaDB vector store in modules/rag_citations.py.
  Ensures AstroAgent LLM outputs are cross-referenced with 40 peer-reviewed
  papers and formatted as Markdown citations. Use when editing rag_citations.py,
  adding papers to the vector store, fixing citation formatting, debugging
  ChromaDB retrieval, or modifying the cite_scientific_literature tool.
---

# RAG Citations & ChromaDB Vector Store

Scope: `modules/rag_citations.py`, `tests/test_rag_citations.py`, and the
`cite_scientific_literature` tool in `modules/agent_setup.py`.

## Architecture

```
User query → AstroAgent (LangChain AgentExecutor)
                 ↓ calls cite_scientific_literature(query, topics)
           modules/agent_setup.py::cite_scientific_literature(query, topics)
                 ↓ imports and calls
           modules/rag_citations.py::cite_literature(query, n_results=5, topics)
                 ↓ hybrid search
           _hybrid_search() — Reciprocal Rank Fusion of:
             ├─ ChromaDB semantic search (all-MiniLM-L6-v2 embeddings)
             └─ TF-IDF weighted keyword search
                 ↓ with optional topic filtering (Python-level)
           Top-5 results by fused RRF score
                 ↓ formatted by
           format_citations_markdown(citations) → Markdown reference block
```

## Invariants — NEVER Violate

1. **40-paper corpus.** The `_PAPERS` list must contain exactly the 40 papers
   catalogued in [reference.md](reference.md). Do not remove entries. New
   papers may only be appended after scientific justification and must include
   all eight required fields.
2. **Eight required fields per paper:** `id`, `title`, `authors`, `year`,
   `journal`, `abstract`, `topics`, `key_findings`. All strings except
   `topics` (list of str) and `key_findings` (list of str).
3. **Deterministic IDs.** Paper `id` values use the pattern
   `<first_author_surname><year>` (e.g., `kopparapu2013`). Joint authorship
   uses `<a>_<b><year>` (e.g., `chen_kipping2017`).
4. **No LLM physics.** The LLM (Qwen / AstroAgent) must NEVER compute
   physics. It invokes `cite_scientific_literature` to retrieve grounded
   citations and synthesises the narrative. Formulas come from the deterministic
   Python tools, never from the LLM.
5. **Graceful degradation.** If `chromadb` or `sentence-transformers` are
   missing at runtime, all public functions fall back to TF-IDF keyword search
   (no exceptions raised). The fallback path must always work with zero external
   dependencies beyond the standard library.

## ChromaDB Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Client | `chromadb.PersistentClient(path="data/chroma_db")` | Persistent across restarts |
| Collection | `astro_papers` | Single collection for all papers |
| Embedding | `all-MiniLM-L6-v2` via `SentenceTransformerEmbeddingFunction` | Fast, 384-dim, sufficient for abstract-level retrieval |
| Documents | Composite: abstract + key findings | Richer semantic content than abstract alone |
| Metadata | title, authors, year, journal | Stored in ChromaDB metadatas |
| Re-seeding | Automatic when `collection.count() != len(_PAPERS)` | Handles corpus expansions without manual cache invalidation |
| Lazy init | `_get_collection()` with module-level `_collection` singleton | Avoids import-time side effects |

## Hybrid Search (Reciprocal Rank Fusion)

The search pipeline fuses two retrieval strategies:

1. **Semantic search:** ChromaDB cosine similarity on sentence embeddings
   of the composite document (abstract + key findings).
2. **TF-IDF keyword search:** Log-weighted IDF scoring over tokenised
   composite documents with stop-word removal.
3. **Fusion:** Reciprocal Rank Fusion (Cormack et al. 2009) with k=60:
   `score(paper) = 1/(k + rank_semantic) + 1/(k + rank_keyword)`
4. **Topic filter:** Optional Python-level filter retaining papers matching
   any of the requested topic tags (union semantics).

## Topic Taxonomy

Papers are tagged with topic labels from this taxonomy:

| Domain | Tags |
|--------|------|
| Habitable Zone | `habitable_zone`, `hz_boundaries` |
| Habitability Metrics | `habitability_metrics`, `esi`, `sephi` |
| Climate Modeling | `climate_modeling`, `gcm`, `climate_states`, `cloud_feedback`, `ocean_heat_transport` |
| Tidal Locking | `tidal_locking`, `atmospheric_collapse`, `atmospheric_dynamics` |
| Biosignatures | `biosignatures`, `false_positives`, `spectroscopy`, `remote_detection` |
| Atmospheric Science | `atmospheric_escape`, `atmospheric_chemistry`, `runaway_greenhouse`, `atmospheric_retention` |
| Stellar Context | `m_dwarf`, `stellar_activity`, `stellar_flares`, `uv_environment`, `pre_main_sequence` |
| Planetary Interiors | `planetary_interior`, `plate_tectonics`, `volcanism`, `tidal_heating`, `magnetic_field`, `mass_radius`, `composition` |
| Observational | `jwst`, `transit_spectroscopy`, `thermal_emission`, `water_detection` |
| Astrobiology | `astrobiology`, `habitability`, `photosynthesis`, `alternative_biochemistry`, `limits_of_life` |

## Citation Formatting Rules

`format_citations_markdown` must produce this exact pattern:

```markdown
**References:**
- Kopparapu, R.K. et al. (2013). *Habitable Zones Around Main-Sequence Stars: New Estimates*. ApJ, 765, 131.
- Shields, A.L. et al. (2016). *The habitability of planets orbiting M-dwarf stars*. Physics Reports, 663, 1-38.
```

Rules:
- Line prefix: `- ` (dash + space)
- Authors verbatim from `_PAPERS["authors"]`, followed by ` (<year>)`
- Title in `*italics*`
- Journal verbatim, terminated by `.`
- No DOI, no URL — keep compact for LLM context window

The `cite_scientific_literature` tool in `agent_setup.py` additionally renders
per-paper detail blocks (bold author-year, italic journal, truncated abstract,
key findings) above the formatted references.

## Editing Workflow

### Adding a new paper

1. Verify the paper is peer-reviewed and relevant to exoplanet habitability.
2. Append a new dict to `_PAPERS` with all eight fields (including `topics`
   and `key_findings`).
3. Update [reference.md](reference.md) with the new entry.
4. Add a test in `tests/test_rag_citations.py` confirming the new paper is
   retrievable via `_fallback_keyword_search` for a relevant query.
5. Delete `data/chroma_db/` directory to force re-indexing on next startup.
6. Run `pytest tests/test_rag_citations.py -v` to confirm no regressions.

### Modifying retrieval logic

- `cite_literature` must always return `List[Dict[str, str]]` with keys:
  `title`, `authors`, `year`, `journal`, `abstract`, `relevance_score`,
  and optionally `key_findings`.
- `relevance_score` is a stringified float: RRF score for hybrid, or
  normalised TF-IDF for fallback.
- Keep `n_results` parameter with default `5`.
- Keep `topics` parameter with default `None` (no filter).

### Modifying the ChromaDB store

- The collection name `astro_papers` is referenced in `_get_collection()` only.
- The persist path is `data/chroma_db` (set in `_CHROMA_PATH`).
- Re-seeding triggers automatically when paper count changes.
- Embedding model changes require deleting `data/chroma_db/` and re-indexing.

## Testing Contract

`tests/test_rag_citations.py` must cover:

| Test class | What it validates |
|------------|-------------------|
| `TestCorpusIntegrity` | 40 papers, required fields, unique IDs, topics, key_findings |
| `TestTokenize` | Tokenisation and stop-word removal |
| `TestTfidfScore` | Relevant docs score higher than irrelevant |
| `TestFallbackSearch` | Returns correct count, Kopparapu ranked, respects n_results, returns key_findings, topic filter |
| `TestTopicFiltering` | Single topic, multi-topic union, no-match, domain coverage |
| `TestCompositeDocument` | Abstract and key findings included |
| `TestFormatCitations` | Empty, single, multiple formatting |
| `TestCiteLiterature` | Returns list, default n=5, topic restricts, required keys |
| `TestDomainCoverage` | Atmospheric escape, JWST, clouds, carbonate-silicate, OHT retrievable |

All tests must pass without `chromadb` installed (fallback path).

## Integration Points

| File | Symbol | Role |
|------|--------|------|
| `modules/agent_setup.py` | `cite_scientific_literature` (LangChain `@tool`) | Wraps `cite_literature` + `format_citations_markdown`; accepts `topics` param |
| `modules/agent_setup.py` | `SYSTEM_PROMPT` CITATION POLICY section | Instructs AstroAgent to always cite, use topic filters, include 2-3 citations per claim |
| `modules/agent_setup.py` | `tools` list | Registers the tool with `AgentExecutor` |
| `METHODOLOGY.md` §12 | 17 bibliography entries (subset of RAG catalog) | Canonical reference list for implemented formulas |

## Common Pitfalls

- **Swallowed exceptions.** The bare `except Exception` in `_get_collection`
  and `_hybrid_search` is intentional for graceful degradation. Do NOT narrow
  these to specific exception types without also adding a fallback.
- **Stale singleton.** The module-level `_collection` is never invalidated at
  runtime. If `_PAPERS` is modified at runtime (test monkeypatching), reset
  `_collection = None` first.
- **Abstract truncation.** `cite_scientific_literature` truncates abstracts to
  500 chars (`c['abstract'][:500]`). This is for LLM context budget only; the
  full abstract remains in ChromaDB and `_PAPERS`.
- **Persistent storage path.** `data/chroma_db/` should be in `.gitignore` as
  it is a generated artifact rebuilt from `_PAPERS` on first access.
- **IDF cache.** `_idf_cache` is lazily built once and never invalidated. If
  papers are modified at runtime, set `_idf_cache = None` to force rebuild.
- **RRF constant.** `_RRF_K = 60` follows the Cormack et al. (2009) standard.
  Lowering it increases weight of top-ranked results; raising it smooths scores.

## Quick Reference — Paper IDs (40 papers)

| ID | First Author | Year | Domain |
|----|-------------|------|--------|
| `kopparapu2013` | Kopparapu | 2013 | HZ boundaries |
| `schulze-makuch2011` | Schulze-Makuch | 2011 | ESI |
| `kasting1993` | Kasting | 1993 | HZ (seminal) |
| `rodriguez-mozos2017` | Rodríguez-Mozos | 2017 | SEPHI |
| `turbet2016` | Turbet | 2016 | Proxima Cen b GCM |
| `shields2016` | Shields | 2016 | M-dwarf habitability |
| `leconte2013` | Leconte | 2013 | Tidally locked GCM |
| `pierrehumbert2011` | Pierrehumbert | 2011 | Climate states |
| `wordsworth2015` | Wordsworth | 2015 | Atmospheric collapse |
| `meadows2018` | Meadows | 2018 | O₂ false positives |
| `seager2016` | Seager | 2016 | Biosignature gases |
| `luger2015` | Luger & Barnes | 2015 | Abiotic O₂ |
| `schwieterman2018` | Schwieterman | 2018 | Biosignatures review |
| `catling2018` | Catling | 2018 | Bayesian biosignature framework |
| `chen_kipping2017` | Chen & Kipping | 2017 | Mass-radius |
| `kite2009` | Kite | 2009 | Outgassing |
| `zeng2019` | Zeng | 2019 | M-R-composition |
| `stamenkovic2012` | Stamenković | 2012 | Super-Earth tectonics |
| `walker1981` | Walker | 1981 | Carbonate-silicate cycle |
| `driscoll_barnes2015` | Driscoll & Barnes | 2015 | Tidal heating |
| `petkowski2020` | Petkowski | 2020 | Silicon biochemistry |
| `owen_wu2013` | Owen & Wu | 2013 | Atmospheric escape |
| `goldblatt2013` | Goldblatt | 2013 | Runaway greenhouse |
| `zahnle_catling2017` | Zahnle & Catling | 2017 | Cosmic shoreline |
| `tian2015` | Tian | 2015 | Water loss |
| `wolf_toon2015` | Wolf & Toon | 2015 | Cloud feedbacks |
| `lammer2009` | Lammer | 2009 | Habitability factors |
| `ramirez_kaltenegger2014` | Ramirez & Kaltenegger | 2014 | Pre-MS HZ |
| `segura2010` | Segura | 2010 | Stellar flares |
| `france2013` | France | 2013 | M-dwarf UV |
| `madhusudhan2023` | Madhusudhan | 2023 | JWST K2-18 b |
| `lustig-yaeger2023` | Lustig-Yaeger | 2023 | JWST LHS 475 b |
| `greene2023` | Greene | 2023 | JWST TRAPPIST-1 b |
| `benneke2019` | Benneke | 2019 | K2-18 b water |
| `yang2013` | Yang | 2013 | Cloud feedback HZ |
| `hu_yang2014` | Hu & Yang | 2014 | Ocean heat transport |
| `joshi1997` | Joshi | 1997 | First TL GCM |
| `del_genio2019` | Del Genio | 2019 | ROCKE-3D Proxima b |
| `cockell2016` | Cockell | 2016 | Habitability review |
| `raven_cockell2006` | Raven & Cockell | 2006 | Photosynthesis limits |

For full titles, journals, and abstracts see [reference.md](reference.md).
