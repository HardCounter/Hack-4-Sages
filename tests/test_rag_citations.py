"""Tests for RAG citation retrieval — covers TF-IDF fallback, topic filtering,
hybrid search, key findings, citation formatting, integration with the
cite_scientific_literature tool, and domain coverage across the 40-paper corpus."""

from modules.rag_citations import (
    _build_composite_document,
    _fallback_keyword_search,
    _filter_by_topics,
    _hybrid_search,
    _paper_to_citation,
    _PAPERS,
    _PAPERS_BY_ID,
    _RRF_K,
    _tfidf_score,
    _tokenize,
    cite_literature,
    format_citations_markdown,
)


# ── Corpus integrity ──────────────────────────────────────────────────────────

class TestCorpusIntegrity:
    def test_paper_count(self):
        assert len(_PAPERS) == 40

    def test_required_fields(self):
        required = {"id", "title", "authors", "year", "journal", "abstract"}
        for p in _PAPERS:
            missing = required - set(p.keys())
            assert not missing, f"Paper {p['id']} missing fields: {missing}"

    def test_unique_ids(self):
        ids = [p["id"] for p in _PAPERS]
        assert len(ids) == len(set(ids)), "Duplicate paper IDs found"

    def test_all_have_topics(self):
        for p in _PAPERS:
            assert isinstance(p.get("topics"), list), f"{p['id']} missing topics"
            assert len(p["topics"]) >= 1, f"{p['id']} has empty topics"

    def test_all_have_key_findings(self):
        for p in _PAPERS:
            assert isinstance(p.get("key_findings"), list), f"{p['id']} missing key_findings"
            assert len(p["key_findings"]) >= 2, f"{p['id']} has too few key_findings"

    def test_papers_by_id_index(self):
        assert len(_PAPERS_BY_ID) == len(_PAPERS)
        assert "kopparapu2013" in _PAPERS_BY_ID
        assert "yang2013" in _PAPERS_BY_ID


# ── TF-IDF infrastructure ────────────────────────────────────────────────────

class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("Habitable Zone Boundaries for M-dwarf stars")
        assert "habitable" in tokens
        assert "zone" in tokens
        assert "boundaries" in tokens

    def test_removes_stop_words(self):
        tokens = _tokenize("the effect of a strong stellar flare on the atmosphere")
        assert "the" not in tokens
        assert "of" not in tokens
        assert "stellar" in tokens
        assert "flare" in tokens


class TestTfidfScore:
    def test_relevant_higher_than_irrelevant(self):
        query_tokens = _tokenize("habitable zone boundaries Kopparapu")
        hz_doc = _build_composite_document(_PAPERS_BY_ID["kopparapu2013"])
        silicon_doc = _build_composite_document(_PAPERS_BY_ID["petkowski2020"])
        assert _tfidf_score(query_tokens, hz_doc) > _tfidf_score(query_tokens, silicon_doc)

    def test_empty_query(self):
        assert _tfidf_score([], "some document text") == 0.0


# ── Fallback keyword search ──────────────────────────────────────────────────

class TestFallbackSearch:
    def test_returns_results(self):
        results = _fallback_keyword_search("habitable zone boundaries", n_results=5)
        assert len(results) == 5
        assert all("title" in r for r in results)

    def test_kopparapu_ranked_for_hz_query(self):
        results = _fallback_keyword_search("habitable zone Kopparapu", n_results=5)
        titles = [r["title"] for r in results]
        assert any("Habitable Zones" in t for t in titles)

    def test_respects_n_results(self):
        results = _fallback_keyword_search("exoplanet", n_results=3)
        assert len(results) == 3

    def test_returns_key_findings(self):
        results = _fallback_keyword_search("habitable zone", n_results=1)
        assert results[0].get("key_findings") is not None

    def test_topic_filter(self):
        results = _fallback_keyword_search(
            "planet habitability", n_results=10, topics=["jwst"],
        )
        for r in results:
            paper = _PAPERS_BY_ID.get(
                next(p["id"] for p in _PAPERS if p["title"] == r["title"]),
            )
            assert "jwst" in paper.get("topics", []) or not results


# ── Topic filtering ──────────────────────────────────────────────────────────

class TestTopicFiltering:
    def test_single_topic(self):
        filtered = _filter_by_topics(_PAPERS, ["jwst"])
        assert len(filtered) >= 3
        for p in filtered:
            assert "jwst" in p["topics"]

    def test_multi_topic_union(self):
        filtered = _filter_by_topics(_PAPERS, ["jwst", "photosynthesis"])
        assert len(filtered) >= 4
        for p in filtered:
            assert "jwst" in p["topics"] or "photosynthesis" in p["topics"]

    def test_no_match(self):
        filtered = _filter_by_topics(_PAPERS, ["nonexistent_topic_xyz"])
        assert len(filtered) == 0

    def test_m_dwarf_coverage(self):
        filtered = _filter_by_topics(_PAPERS, ["m_dwarf"])
        assert len(filtered) >= 5

    def test_biosignatures_coverage(self):
        filtered = _filter_by_topics(_PAPERS, ["biosignatures"])
        assert len(filtered) >= 4


# ── Composite document ────────────────────────────────────────────────────────

class TestCompositeDocument:
    def test_includes_abstract(self):
        doc = _build_composite_document(_PAPERS_BY_ID["kopparapu2013"])
        assert "habitable zone" in doc.lower()

    def test_includes_key_findings(self):
        doc = _build_composite_document(_PAPERS_BY_ID["kopparapu2013"])
        assert "Key findings:" in doc
        assert "0.99-1.70 AU" in doc


# ── Citation formatting ──────────────────────────────────────────────────────

class TestFormatCitations:
    def test_empty(self):
        assert format_citations_markdown([]) == ""

    def test_formats_correctly(self):
        citations = [{
            "title": "Test Paper",
            "authors": "Author, A.",
            "year": "2024",
            "journal": "Journal X",
            "abstract": "Abstract text.",
            "relevance_score": "0.9",
        }]
        md = format_citations_markdown(citations)
        assert "Author, A." in md
        assert "2024" in md
        assert "*Test Paper*" in md
        assert md.startswith("**References:**")

    def test_multiple_citations(self):
        results = _fallback_keyword_search("atmosphere climate", n_results=3)
        md = format_citations_markdown(results)
        assert md.count("\n-") == 3


# ── Public API (cite_literature) ──────────────────────────────────────────────

class TestCiteLiterature:
    def test_returns_list(self):
        results = cite_literature("exoplanet habitability", n_results=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_default_n_results_is_five(self):
        results = cite_literature("habitable zone M dwarf tidal locking")
        assert len(results) <= 5

    def test_topic_filter_restricts(self):
        all_results = cite_literature("planet atmosphere", n_results=10)
        jwst_results = cite_literature(
            "planet atmosphere", n_results=10, topics=["jwst"],
        )
        assert len(jwst_results) <= len(all_results)

    def test_result_has_required_keys(self):
        results = cite_literature("biosignature oxygen", n_results=1)
        assert len(results) >= 1
        r = results[0]
        for key in ("title", "authors", "year", "journal", "abstract", "relevance_score"):
            assert key in r, f"Missing key: {key}"


# ── Domain coverage ───────────────────────────────────────────────────────────

class TestDomainCoverage:
    """Verify that each scientific domain is retrievable."""

    def test_atmospheric_escape(self):
        results = _fallback_keyword_search("atmospheric escape evaporation XUV", n_results=3)
        assert any("escape" in r["title"].lower() or "evaporation" in r["title"].lower()
                    for r in results)

    def test_jwst_observations(self):
        results = _fallback_keyword_search("JWST transmission spectrum exoplanet", n_results=3)
        assert any("jwst" in r["title"].lower() or "JWST" in r["abstract"]
                    for r in results)

    def test_cloud_feedback(self):
        results = _fallback_keyword_search("cloud feedback tidally locked habitable zone", n_results=3)
        assert any("cloud" in r["title"].lower() for r in results)

    def test_carbonate_silicate(self):
        results = _fallback_keyword_search("carbonate silicate cycle weathering", n_results=3)
        assert any("feedback" in r["title"].lower() or "carbonate" in r["abstract"].lower()
                    for r in results)

    def test_ocean_heat_transport(self):
        results = _fallback_keyword_search("ocean heat transport tidally locked", n_results=3)
        assert any("ocean" in r["title"].lower() for r in results)

    def test_tidal_heating_magnetic_field(self):
        results = _fallback_keyword_search("tidal heating magnetic dynamo eccentricity", n_results=3)
        assert any("tidal" in r["title"].lower() for r in results)

    def test_mass_radius_relation(self):
        results = _fallback_keyword_search("mass radius relation terran neptunian", n_results=3)
        assert any("mass" in r["title"].lower() or "radius" in r["title"].lower()
                    for r in results)

    def test_pre_main_sequence(self):
        results = _fallback_keyword_search("pre-main-sequence stellar luminosity habitable zone", n_results=3)
        assert any("pre-main-sequence" in r["title"].lower() or "pre_main_sequence" in r["abstract"].lower()
                    for r in results)

    def test_runaway_greenhouse(self):
        results = _fallback_keyword_search("runaway greenhouse radiation limit", n_results=3)
        assert any("runaway" in r["title"].lower() or "greenhouse" in r["title"].lower()
                    for r in results)

    def test_silicon_biochemistry(self):
        results = _fallback_keyword_search("silicon alternative biochemistry", n_results=3)
        assert any("silicon" in r["title"].lower() for r in results)


# ── Hybrid search ────────────────────────────────────────────────────────────

class TestHybridSearch:
    """Test _hybrid_search (uses ChromaDB when available, fallback otherwise)."""

    def test_returns_expected_count(self):
        results = _hybrid_search("habitable zone boundaries", n_results=5)
        assert len(results) == 5

    def test_returns_required_keys(self):
        results = _hybrid_search("exoplanet atmosphere", n_results=1)
        assert len(results) >= 1
        for key in ("title", "authors", "year", "journal", "abstract", "relevance_score"):
            assert key in results[0], f"Missing key: {key}"

    def test_respects_n_results(self):
        for n in (1, 3, 7):
            results = _hybrid_search("planet", n_results=n)
            assert len(results) == n

    def test_topic_filter(self):
        results = _hybrid_search(
            "planet atmosphere", n_results=10, topics=["jwst"],
        )
        for r in results:
            paper = next(p for p in _PAPERS if p["title"] == r["title"])
            assert "jwst" in paper["topics"]

    def test_hz_papers_for_hz_query(self):
        results = _hybrid_search("habitable zone boundaries main sequence stars", n_results=5)
        assert any("habitable" in r["title"].lower() for r in results)

    def test_relevance_scores_are_positive(self):
        results = _hybrid_search("climate modeling GCM", n_results=5)
        for r in results:
            assert float(r["relevance_score"]) > 0

    def test_scores_are_descending(self):
        results = _hybrid_search("biosignature oxygen false positive", n_results=5)
        scores = [float(r["relevance_score"]) for r in results]
        assert scores == sorted(scores, reverse=True)


# ── Paper-to-citation conversion ─────────────────────────────────────────────

class TestPaperToCitation:
    def test_required_keys_present(self):
        paper = _PAPERS_BY_ID["kopparapu2013"]
        cit = _paper_to_citation(paper, 0.95)
        for key in ("title", "authors", "year", "journal", "abstract", "relevance_score"):
            assert key in cit

    def test_score_stringified(self):
        cit = _paper_to_citation(_PAPERS_BY_ID["yang2013"], 0.12345)
        assert cit["relevance_score"] == "0.123"

    def test_key_findings_included(self):
        cit = _paper_to_citation(_PAPERS_BY_ID["meadows2018"], 0.5)
        assert "key_findings" in cit
        assert isinstance(cit["key_findings"], list)

    def test_paper_without_key_findings(self):
        fake = {
            "id": "test2099",
            "title": "Test",
            "authors": "A.",
            "year": "2099",
            "journal": "J.",
            "abstract": "Abstract.",
        }
        cit = _paper_to_citation(fake, 0.1)
        assert "key_findings" not in cit


# ── RRF constant sanity ──────────────────────────────────────────────────────

class TestRRFConstant:
    def test_rrf_k_value(self):
        assert _RRF_K == 60

    def test_rrf_score_formula(self):
        """Top-ranked in both lists should score higher than mid-ranked."""
        top_score = 1.0 / (_RRF_K + 1) + 1.0 / (_RRF_K + 1)
        mid_score = 1.0 / (_RRF_K + 20) + 1.0 / (_RRF_K + 20)
        assert top_score > mid_score


# ── Integration: cite_scientific_literature tool ─────────────────────────────

class TestCiteScientificLiteratureTool:
    """End-to-end test of the LangChain tool wrapper in agent_setup.py."""

    def test_returns_string(self):
        from modules.agent_setup import cite_scientific_literature
        result = cite_scientific_literature.invoke(
            {"query": "habitable zone M dwarf", "topics": ""},
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_references_block(self):
        from modules.agent_setup import cite_scientific_literature
        result = cite_scientific_literature.invoke(
            {"query": "tidal locking atmospheric collapse", "topics": ""},
        )
        assert "**References:**" in result

    def test_topic_filtering(self):
        from modules.agent_setup import cite_scientific_literature
        result = cite_scientific_literature.invoke(
            {"query": "exoplanet atmosphere observation", "topics": "jwst"},
        )
        assert "JWST" in result or "jwst" in result.lower()

    def test_no_results_message(self):
        from modules.agent_setup import cite_scientific_literature
        result = cite_scientific_literature.invoke(
            {"query": "quantum gravity string theory", "topics": "nonexistent_topic"},
        )
        assert result == "No relevant citations found."

    def test_includes_key_findings(self):
        from modules.agent_setup import cite_scientific_literature
        result = cite_scientific_literature.invoke(
            {"query": "biosignature false positive oxygen", "topics": "biosignatures"},
        )
        assert "Key findings:" in result or "key findings" in result.lower()

    def test_abstract_truncation(self):
        """The tool truncates abstracts to 500 chars for LLM context budget."""
        from modules.agent_setup import cite_scientific_literature
        result = cite_scientific_literature.invoke(
            {"query": "habitable zone boundaries main sequence", "topics": ""},
        )
        lines = result.split("\n")
        for line in lines:
            if line.startswith("**") and "—" in line:
                continue
            if line.startswith("_") or line.startswith("Key findings:"):
                continue
            if line.startswith("- "):
                continue
