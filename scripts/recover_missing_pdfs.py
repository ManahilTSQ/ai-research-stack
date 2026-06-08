"""
recover_missing_pdfs.py — Recover missing PDFs for abstract-only papers.

This script:
1. Inspects ChromaDB to find papers with < 5 chunks (abstract-only)
2. For each such paper, queries Unpaywall for open-access PDF
3. Downloads the PDF if found
4. Outputs a list of recovered papers for re-ingestion

Usage:
    python scripts/recover_missing_pdfs.py
"""

import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path("c:/Users/PMLS/OneDrive/Desktop/AI Research Stack")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from vector_store import VectorStoreService
from paper_discovery import PaperDiscoveryService
from config import settings


def get_papers_with_few_chunks(min_chunks: int = 5):
    """
    Identify papers with fewer than min_chunks (likely abstract-only).
    
    Returns:
        dict: {title: chunk_count} for papers below threshold
    """
    store = VectorStoreService()
    data = store.collection.get(include=["metadatas"])
    
    title_counts = Counter(
        m.get("title", "Unknown") 
        for m in data["metadatas"] if m
    )
    
    # Filter to papers with < min_chunks
    low_chunk_papers = {
        title: count 
        for title, count in title_counts.items() 
        if count < min_chunks
    }
    
    return low_chunk_papers


def get_doi_for_paper(title: str, papers_metadata: dict) -> str | None:
    """
    Get DOI for a paper from metadata.
    """
    meta = papers_metadata.get(title, {})
    doi = meta.get("doi", "")
    if doi and doi.lower() != "n/a":
        # Clean DOI
        doi = doi.replace("https://doi.org/", "").strip()
        return doi
    return None


def recover_pdfs():
    """
    Main recovery workflow.
    """
    print("=== PDF Recovery for Abstract-Only Papers ===\n")
    
    # Step 1: Find papers with < 5 chunks
    print("Step 1: Identifying abstract-only papers...")
    low_chunk_papers = get_papers_with_few_chunks(min_chunks=5)
    
    if not low_chunk_papers:
        print("✓ All papers have sufficient chunks (>= 5). No recovery needed.")
        return
    
    print(f"Found {len(low_chunk_papers)} papers with < 5 chunks:\n")
    for title, count in sorted(low_chunk_papers.items(), key=lambda x: x[1]):
        status = "⚠️" if count == 1 else "⚠️"
        print(f"  {status} {count:3d} chunks | {title[:70]}")
    
    # Step 2: Get papers metadata for DOI lookup
    print("\nStep 2: Loading papers metadata...")
    store = VectorStoreService()
    papers_metadata = {}
    data = store.collection.get(include=["metadatas"])
    for m in data["metadatas"]:
        if m:
            title = m.get("title", "")
            if title:
                papers_metadata[title] = m
    
    # Step 3: Try to recover PDFs via cascade of sources
    print("\nStep 3: Attempting PDF recovery via cascade of sources...")
    discovery = PaperDiscoveryService()

    recovered = []
    failed = []

    for title, chunk_count in sorted(low_chunk_papers.items(), key=lambda x: x[1]):
        doi = get_doi_for_paper(title, papers_metadata)

        if not doi:
            print(f"  ✗ No DOI available for: {title[:60]}")
            failed.append((title, "No DOI"))
            continue

        pdf_url = None
        source_used = None

        # Cascade order: ArXiv → Unpaywall → Core.ac.uk → MDPI API → PMC E-utilities → OpenAlex
        print(f"\n  → Attempting PDF recovery for '{title}' (DOI: {doi})")

        # 1. Try ArXiv first (never blocked, best for CS/ML papers)
        pdf_url = discovery.fetch_arxiv_pdf_url(doi)
        if pdf_url:
            source_used = "ArXiv"
            print(f"    ✓ Found via ArXiv: {pdf_url[:80]}...")

        # 2. Try Unpaywall
        if not pdf_url:
            pdf_url = discovery.fetch_open_access_pdf_url(doi)
            if pdf_url:
                source_used = "Unpaywall"
                print(f"    ✓ Found via Unpaywall: {pdf_url[:80]}...")

        # 3. Try Core.ac.uk (by title)
        if not pdf_url:
            pdf_url = discovery.fetch_core_ac_pdf_url(title)
            if pdf_url:
                source_used = "Core.ac.uk"
                print(f"    ✓ Found via Core.ac.uk: {pdf_url[:80]}...")

        # 4. Try MDPI research API (for MDPI DOIs)
        if not pdf_url:
            pdf_url = discovery.fetch_mdpi_api_pdf_url(doi)
            if pdf_url:
                source_used = "MDPI API"
                print(f"    ✓ Found via MDPI API: {pdf_url[:80]}...")

        # 5. Try PMC E-utilities (if PMCID available from OpenAlex)
        if not pdf_url:
            # Query OpenAlex to get PMCID
            openalex_data = discovery.fetch_openalex_metadata(doi)
            if openalex_data:
                # Extract PMCID from OpenAlex IDs
                ids = openalex_data.get("ids", {})
                pmcid = ids.get("pmcid") if ids else None
                if pmcid:
                    pdf_url = discovery.fetch_pmc_eutils_pdf_url(pmcid)
                    if pdf_url:
                        source_used = "PMC E-utilities"
                        print(f"    ✓ Found via PMC E-utilities: {pdf_url[:80]}...")

        # 6. Try OpenAlex PDF URLs as final fallback
        if not pdf_url:
            openalex_urls = discovery.fetch_all_openalex_pdf_urls(doi)
            if openalex_urls:
                pdf_url = openalex_urls[0]
                source_used = "OpenAlex"
                print(f"    ✓ Found via OpenAlex: {pdf_url[:80]}...")

        if pdf_url:
            # Generate safe filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title[:50]  # Truncate long titles
            filename = f"{safe_title}.pdf"

            # Download PDF
            save_path = discovery.download_pdf(pdf_url, filename)
            if save_path:
                print(f"    ✓ Downloaded to: {save_path} (source: {source_used})")
                recovered.append((title, doi, str(save_path), source_used))
            else:
                print(f"    ✗ Download failed (source: {source_used})")
                failed.append((title, f"Download failed ({source_used})"))
        else:
            print(f"    ✗ No OA version found from any source")
            failed.append((title, "No OA version from any source"))
    
    # Summary
    print("\n=== Recovery Summary ===")
    print(f"Total abstract-only papers: {len(low_chunk_papers)}")
    print(f"Successfully recovered: {len(recovered)}")
    print(f"Failed to recover: {len(failed)}")
    
    if recovered:
        print("\n=== Recovered Papers (ready for re-ingestion) ===")
        for title, doi, path, source in recovered:
            print(f"  - {title[:70]}")
            print(f"    DOI: {doi}")
            print(f"    Path: {path}")
            print(f"    Source: {source}")
    
    if failed:
        print("\n=== Failed Papers ===")
        for title, reason in failed:
            print(f"  - {title[:70]} ({reason})")
    
    if recovered:
        print("\n=== Next Steps ===")
        print("To re-ingest the recovered PDFs, run:")
        print("  python scripts/batch_reingest.py")
        print("\nOr re-ingest specific papers:")
        for _, _, path, _ in recovered:
            print(f"  python scripts/main.py -f \"{path}\"")


if __name__ == "__main__":
    recover_pdfs()
