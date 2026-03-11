"""Tests for RAG citation retrieval — covers TF-IDF fallback, topic filtering,
hybrid search, key findings, and citation formatting across the 40-paper corpus."""

from modules.rag_citations import (
    _build_composite_document,
    _fallback_keyword_search,
    _filter_by_topics,
    _PAPERS,
    _PAPERS_BY_ID,
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
