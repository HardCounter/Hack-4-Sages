"""Tests for RAG citation retrieval."""

from modules.rag_citations import (
    _fallback_keyword_search,
    cite_literature,
    format_citations_markdown,
)


class TestFallbackSearch:
    def test_returns_results(self):
        results = _fallback_keyword_search("habitable zone boundaries", n_results=3)
        assert len(results) == 3
        assert all("title" in r for r in results)

    def test_kopparapu_ranked_for_hz_query(self):
        results = _fallback_keyword_search("habitable zone Kopparapu", n_results=3)
        titles = [r["title"] for r in results]
        assert any("Habitable Zones" in t for t in titles)


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
            "relevance_score": 0.9,
        }]
        md = format_citations_markdown(citations)
        assert "Author, A." in md
        assert "2024" in md


class TestCiteLiterature:
    def test_returns_list(self):
        results = cite_literature("exoplanet habitability", n_results=2)
        assert isinstance(results, list)
        assert len(results) <= 2
