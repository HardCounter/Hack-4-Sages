"""
Retrieval-Augmented Generation (RAG) for scientific literature citations.

Maintains a local ChromaDB vector store of key exoplanet / astrobiology
paper abstracts. Provides a ``cite_literature`` function that retrieves
the most relevant papers for a given query and returns formatted
citations.

Gracefully degrades: if ChromaDB or sentence-transformers are not
installed, all public functions return empty results rather than raising.
"""

import json
from typing import Dict, List, Optional

_PAPERS: List[Dict[str, str]] = [
    {
        "id": "kopparapu2013",
        "title": "Habitable Zones Around Main-Sequence Stars: New Estimates",
        "authors": "Kopparapu, R.K. et al.",
        "year": "2013",
        "journal": "ApJ, 765, 131",
        "abstract": (
            "We present updated habitable zone boundaries using 1-D climate models "
            "with revised H2O and CO2 absorption coefficients. The inner edge is "
            "defined by the moist greenhouse limit and the outer edge by the maximum "
            "greenhouse effect of a CO2 atmosphere. Our new estimates place the "
            "conservative HZ boundaries at 0.99-1.70 AU for the Sun."
        ),
    },
    {
        "id": "schulze-makuch2011",
        "title": "A Two-Tiered Approach to Assessing the Habitability of Exoplanets",
        "authors": "Schulze-Makuch, D. et al.",
        "year": "2011",
        "journal": "Astrobiology, 11(10), 1041-1052",
        "abstract": (
            "We propose the Earth Similarity Index (ESI) and the Planetary "
            "Habitability Index (PHI) as complementary metrics for evaluating "
            "exoplanet habitability. ESI measures physical similarity to Earth "
            "using radius, density, escape velocity, and surface temperature."
        ),
    },
    {
        "id": "turbet2016",
        "title": "The habitability of Proxima Centauri b",
        "authors": "Turbet, M. et al.",
        "year": "2016",
        "journal": "A&A, 596, A112",
        "abstract": (
            "We use a 3-D Global Climate Model to simulate Proxima Centauri b "
            "under various atmospheric compositions. For synchronous rotation, "
            "an eyeball climate state emerges with open ocean on the dayside. "
            "With 1 bar N2 and variable CO2, the planet can maintain surface "
            "liquid water for a wide range of conditions."
        ),
    },
    {
        "id": "shields2016",
        "title": "The habitability of planets orbiting M-dwarf stars",
        "authors": "Shields, A.L. et al.",
        "year": "2016",
        "journal": "Physics Reports, 663, 1-38",
        "abstract": (
            "M-dwarf habitability depends on tidal locking, stellar activity, "
            "and atmospheric erosion. Tidally locked planets can maintain "
            "habitable conditions via atmospheric heat transport. The eyeball "
            "Earth model shows liquid water at the substellar point surrounded "
            "by a frozen surface."
        ),
    },
    {
        "id": "kasting1993",
        "title": "Habitable Zones Around Main Sequence Stars",
        "authors": "Kasting, J.F., Whitmire, D.P., Reynolds, R.T.",
        "year": "1993",
        "journal": "Icarus, 101, 108-128",
        "abstract": (
            "The seminal habitable zone calculation defining inner and outer "
            "edges based on water loss and maximum CO2 greenhouse effect. "
            "Establishes the framework later refined by Kopparapu et al. (2013)."
        ),
    },
    {
        "id": "meadows2018",
        "title": "Exoplanet Biosignatures: Understanding Oxygen as a Biosignature in the Context of Its Environment",
        "authors": "Meadows, V.S. et al.",
        "year": "2018",
        "journal": "Astrobiology, 18(6), 630-662",
        "abstract": (
            "Oxygen can be produced abiotically through photolysis of CO2 and "
            "H2O, particularly around M-dwarf stars with high UV output. "
            "False positive identification requires understanding the stellar "
            "UV environment, atmospheric composition, and geological context."
        ),
    },
    {
        "id": "petkowski2020",
        "title": "On the Potential of Silicon as a Building Block for Life",
        "authors": "Petkowski, J.J., Bains, W., Seager, S.",
        "year": "2020",
        "journal": "Life, 10(6), 84",
        "abstract": (
            "Alternative biochemistries beyond carbon-based life may use silicon "
            "in specific environments. However, silicon's chemical limitations "
            "make carbon-based life far more probable in known planetary conditions."
        ),
    },
    {
        "id": "chen_kipping2017",
        "title": "Probabilistic Forecasting of the Masses and Radii of Other Worlds",
        "authors": "Chen, J. & Kipping, D.M.",
        "year": "2017",
        "journal": "ApJ, 834, 17",
        "abstract": (
            "We present a probabilistic mass-radius relation for exoplanets "
            "spanning the range from sub-Earths to super-Jupiters. For rocky "
            "planets, R ~ M^0.27, transitioning to R ~ M^0.59 for Neptunian worlds."
        ),
    },
    {
        "id": "rodriguez-mozos2017",
        "title": "SEPHI: A Scoring System for Exoplanet Habitability",
        "authors": "Rodríguez-Mozos, J.M. & Moya, A.",
        "year": "2017",
        "journal": "MNRAS, 471(4), 4628-4636",
        "abstract": (
            "SEPHI evaluates habitability via thermal, atmospheric retention, "
            "and magnetic field criteria. Planets must have surface temperatures "
            "allowing liquid water, sufficient escape velocity to retain an "
            "atmosphere, and adequate mass for a magnetic dynamo."
        ),
    },
    {
        "id": "leconte2013",
        "title": "3D climate modeling of close-in land planets",
        "authors": "Leconte, J. et al.",
        "year": "2013",
        "journal": "A&A, 554, A69",
        "abstract": (
            "Tidally locked planets exhibit strong day-night temperature contrasts "
            "modulated by atmospheric circulation. The terminator region is critical "
            "for habitability assessment. GCM simulations show that even thin "
            "atmospheres can transport enough heat to prevent atmospheric collapse."
        ),
    },
    {
        "id": "pierrehumbert2011",
        "title": "A Palette of Climates for Gliese 581g",
        "authors": "Pierrehumbert, R.T.",
        "year": "2011",
        "journal": "ApJ Letters, 726, L8",
        "abstract": (
            "Climate states of tidally locked planets include the eyeball state "
            "(hot substellar point, frozen elsewhere), the lobster state (warm "
            "dayside with temperate terminator), and the snowball state. The "
            "climate topology depends on atmospheric composition and heat transport."
        ),
    },
    {
        "id": "wordsworth2015",
        "title": "Atmospheric Heat Redistribution and Collapse on Tidally Locked Rocky Planets",
        "authors": "Wordsworth, R.",
        "year": "2015",
        "journal": "ApJ, 806, 180",
        "abstract": (
            "Atmospheric collapse occurs on tidally locked planets when the "
            "nightside temperature drops below the condensation point of major "
            "atmospheric constituents. This sets a minimum atmospheric mass "
            "required for habitability."
        ),
    },
    {
        "id": "kite2009",
        "title": "Geodynamics and Rate of Volcanism on Massive Earth-like Planets",
        "authors": "Kite, E.S. et al.",
        "year": "2009",
        "journal": "ApJ, 700, 1732",
        "abstract": (
            "Volcanic outgassing on super-Earths depends on mantle convection "
            "vigor, which scales with surface gravity and internal heating. "
            "Higher gravity increases convective stress but also increases "
            "lithospheric strength, leading to a non-monotonic relationship."
        ),
    },
    {
        "id": "seager2016",
        "title": "Toward a List of Molecules as Potential Biosignature Gases",
        "authors": "Seager, S. et al.",
        "year": "2016",
        "journal": "Astrobiology, 16(6), 465-485",
        "abstract": (
            "A comprehensive list of small molecules that could serve as "
            "biosignature gases. Each molecule is evaluated for its likelihood "
            "of biological vs. abiotic production. Key biosignatures include "
            "O2, O3, CH4, N2O, and dimethyl sulfide (DMS)."
        ),
    },
    {
        "id": "luger2015",
        "title": "Extreme Water Loss and Abiotic O2 Buildup on Planets Throughout the Habitable Zones of M Dwarfs",
        "authors": "Luger, R. & Barnes, R.",
        "year": "2015",
        "journal": "Astrobiology, 15(2), 119-143",
        "abstract": (
            "M-dwarf planets can lose up to several Earth-oceans of water during "
            "the pre-main-sequence super-luminous phase. The resulting oxygen "
            "buildup can produce detectable O2 and O3 without any biological "
            "activity — a major false-positive concern."
        ),
    },
]

_collection = None


def _get_collection():
    """Lazy-init the ChromaDB collection, indexing all papers on first call."""
    global _collection
    if _collection is not None:
        return _collection
    try:
        import chromadb
        from chromadb.utils import embedding_functions

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.Client()
        _collection = client.get_or_create_collection(
            name="astro_papers", embedding_function=ef,
        )
        if _collection.count() == 0:
            _collection.add(
                ids=[p["id"] for p in _PAPERS],
                documents=[p["abstract"] for p in _PAPERS],
                metadatas=[
                    {"title": p["title"], "authors": p["authors"],
                     "year": p["year"], "journal": p["journal"]}
                    for p in _PAPERS
                ],
            )
        return _collection
    except Exception:
        return None


def cite_literature(query: str, n_results: int = 3) -> List[Dict[str, str]]:
    """Retrieve the most relevant papers for a scientific query.

    Returns a list of dicts with keys: title, authors, year, journal,
    abstract, relevance_score.
    """
    coll = _get_collection()
    if coll is None:
        return _fallback_keyword_search(query, n_results)

    try:
        results = coll.query(query_texts=[query], n_results=n_results)
        citations = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i] if results.get("distances") else 0
            citations.append({
                "title": meta["title"],
                "authors": meta["authors"],
                "year": meta["year"],
                "journal": meta["journal"],
                "abstract": results["documents"][0][i],
                "relevance_score": round(1 - dist, 3) if dist else 1.0,
            })
        return citations
    except Exception:
        return _fallback_keyword_search(query, n_results)


def _fallback_keyword_search(query: str, n_results: int = 3) -> List[Dict[str, str]]:
    """Simple keyword overlap when ChromaDB is unavailable."""
    query_lower = query.lower()
    scored = []
    for p in _PAPERS:
        text = f"{p['title']} {p['abstract']}".lower()
        score = sum(1 for w in query_lower.split() if w in text)
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "title": p["title"],
            "authors": p["authors"],
            "year": p["year"],
            "journal": p["journal"],
            "abstract": p["abstract"],
            "relevance_score": round(s / max(len(query.split()), 1), 3),
        }
        for s, p in scored[:n_results]
    ]


def format_citations_markdown(citations: List[Dict[str, str]]) -> str:
    """Format a list of citations as a markdown reference block."""
    if not citations:
        return ""
    lines = ["**References:**"]
    for c in citations:
        lines.append(
            f"- {c['authors']} ({c['year']}). *{c['title']}*. {c['journal']}."
        )
    return "\n".join(lines)
