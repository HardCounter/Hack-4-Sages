---
name: rag-citations-chromadb
description: >-
  Manages RAG citations via ChromaDB vector store in modules/rag_citations.py.
  Ensures AstroAgent LLM outputs are cross-referenced with 15 peer-reviewed
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
                 ↓ calls cite_scientific_literature tool
           modules/agent_setup.py::cite_scientific_literature(query)
                 ↓ imports and calls
           modules/rag_citations.py::cite_literature(query, n_results=3)
                 ↓ semantic search
           ChromaDB collection "astro_papers"
              (all-MiniLM-L6-v2 embeddings, in-memory)
                 ↓ if ChromaDB unavailable
           _fallback_keyword_search() — simple term-overlap scoring
                 ↓ results formatted by
           format_citations_markdown(citations) → Markdown reference block
```

## Invariants — NEVER Violate

1. **15-paper canon.** The `_PAPERS` list must contain exactly the 15 papers
   catalogued in [reference.md](reference.md). Do not remove entries. New
   papers may only be appended after scientific justification and must include
   all six required fields.
2. **Six required fields per paper:** `id`, `title`, `authors`, `year`,
   `journal`, `abstract`. All strings.
3. **Deterministic IDs.** Paper `id` values use the pattern
   `<first_author_surname><year>` (e.g., `kopparapu2013`). Joint authorship
   uses `<a>_<b><year>` (e.g., `chen_kipping2017`).
4. **No LLM physics.** The LLM (Qwen / AstroAgent) must NEVER compute
   physics. It invokes `cite_scientific_literature` to retrieve grounded
   citations and synthesizes the narrative. Formulas come from the deterministic
   Python tools, never from the LLM.
5. **Graceful degradation.** If `chromadb` or `sentence-transformers` are
   missing at runtime, all public functions return empty results (not
   exceptions). The fallback path (`_fallback_keyword_search`) must always work
   with zero external dependencies.

## ChromaDB Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Client | `chromadb.Client()` (in-memory) | Hackathon demo; no persistent dir needed |
| Collection | `astro_papers` | Single collection for all papers |
| Embedding | `all-MiniLM-L6-v2` via `SentenceTransformerEmbeddingFunction` | Fast, 384-dim, sufficient for abstract-level retrieval |
| Documents | Paper abstracts only | Titles/authors/year/journal stored as ChromaDB `metadatas` |
| Lazy init | `_get_collection()` with module-level `_collection` singleton | Avoids import-time side effects |

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
per-paper detail blocks (bold author-year, italic journal, truncated abstract)
above the formatted references.

## Editing Workflow

### Adding a new paper

1. Verify the paper is peer-reviewed and relevant to exoplanet habitability.
2. Append a new dict to `_PAPERS` with all six fields.
3. Update [reference.md](reference.md) with the new entry.
4. Add a test in `tests/test_rag_citations.py` confirming the new paper is
   retrievable via `_fallback_keyword_search` for a relevant query.
5. Run `pytest tests/test_rag_citations.py -v` to confirm no regressions.

### Modifying retrieval logic

- `cite_literature` must always return `List[Dict[str, str]]` with keys:
  `title`, `authors`, `year`, `journal`, `abstract`, `relevance_score`.
- `relevance_score` is `1 - cosine_distance` for ChromaDB results, or
  normalised keyword overlap count for fallback results.
- Keep `n_results` parameter with default `3`.

### Modifying the ChromaDB store

- The collection name `astro_papers` is referenced in `_get_collection()` only.
- If switching to persistent storage, use `chromadb.PersistentClient(path=...)`,
  update the singleton guard, and ensure `_collection.count() == 0` seeding
  logic still runs on first access.
- Embedding model changes require re-indexing (delete collection, re-add).

## Testing Contract

`tests/test_rag_citations.py` must cover:

| Test class | What it validates |
|------------|-------------------|
| `TestFallbackSearch` | Keyword search returns correct count; Kopparapu ranked for HZ queries |
| `TestFormatCitations` | Empty input → `""`; single citation formats correctly |
| `TestCiteLiterature` | Returns `list`, respects `n_results` cap |

All tests must pass without `chromadb` installed (fallback path).

## Integration Points

| File | Symbol | Role |
|------|--------|------|
| `modules/agent_setup.py` | `cite_scientific_literature` (LangChain `@tool`) | Wraps `cite_literature` + `format_citations_markdown` |
| `modules/agent_setup.py` | `SYSTEM_PROMPT` line "8. Cite scientific literature…" | Instructs AstroAgent to use the tool |
| `modules/agent_setup.py` | `tools` list | Registers the tool with `AgentExecutor` |
| `METHODOLOGY.md` §12 | 17 bibliography entries (superset of RAG catalog) | Canonical reference list for the project |

## Common Pitfalls

- **Swallowed exceptions.** The bare `except Exception` in `_get_collection`
  and `cite_literature` is intentional for graceful degradation. Do NOT narrow
  these to specific exception types without also adding a fallback.
- **Stale singleton.** The module-level `_collection` is never invalidated.
  If `_PAPERS` is modified at runtime (test monkeypatching), reset
  `_collection = None` first.
- **Abstract truncation.** `cite_scientific_literature` truncates abstracts to
  300 chars (`c['abstract'][:300]`). This is for LLM context budget only; the
  full abstract remains in ChromaDB and `_PAPERS`.

## Quick Reference — Paper IDs

| ID | First Author | Year |
|----|-------------|------|
| `kopparapu2013` | Kopparapu | 2013 |
| `schulze-makuch2011` | Schulze-Makuch | 2011 |
| `turbet2016` | Turbet | 2016 |
| `shields2016` | Shields | 2016 |
| `kasting1993` | Kasting | 1993 |
| `meadows2018` | Meadows | 2018 |
| `petkowski2020` | Petkowski | 2020 |
| `chen_kipping2017` | Chen & Kipping | 2017 |
| `rodriguez-mozos2017` | Rodríguez-Mozos | 2017 |
| `leconte2013` | Leconte | 2013 |
| `pierrehumbert2011` | Pierrehumbert | 2011 |
| `wordsworth2015` | Wordsworth | 2015 |
| `kite2009` | Kite | 2009 |
| `seager2016` | Seager | 2016 |
| `luger2015` | Luger & Barnes | 2015 |

For full titles, journals, and abstracts see [reference.md](reference.md).
