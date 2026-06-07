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
    
    # Step 3: Try to recover PDFs via Unpaywall
    print("\nStep 3: Querying Unpaywall for open-access PDFs...")
    discovery = PaperDiscoveryService()
    
    recovered = []
    failed = []
    
    for title, chunk_count in sorted(low_chunk_papers.items(), key=lambda x: x[1]):
        doi = get_doi_for_paper(title, papers_metadata)
        
        if not doi:
            print(f"  ✗ No DOI available for: {title[:60]}")
            failed.append((title, "No DOI"))
            continue
        
        print(f"  → Querying Unpaywall for DOI: {doi}")
        pdf_url = discovery.fetch_open_access_pdf_url(doi)
        
        if pdf_url:
            print(f"    ✓ Found OA PDF: {pdf_url[:80]}...")
            
            # Generate safe filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_title = safe_title[:50]  # Truncate long titles
            filename = f"{safe_title}.pdf"
            
            # Download PDF
            save_path = discovery.download_pdf(pdf_url, filename)
            if save_path:
                print(f"    ✓ Downloaded to: {save_path}")
                recovered.append((title, doi, str(save_path)))
            else:
                print(f"    ✗ Download failed")
                failed.append((title, "Download failed"))
        else:
            print(f"    ✗ No OA version found")
            failed.append((title, "No OA version"))
    
    # Summary
    print("\n=== Recovery Summary ===")
    print(f"Total abstract-only papers: {len(low_chunk_papers)}")
    print(f"Successfully recovered: {len(recovered)}")
    print(f"Failed to recover: {len(failed)}")
    
    if recovered:
        print("\n=== Recovered Papers (ready for re-ingestion) ===")
        for title, doi, path in recovered:
            print(f"  - {title[:70]}")
            print(f"    DOI: {doi}")
            print(f"    Path: {path}")
    
    if failed:
        print("\n=== Failed Papers ===")
        for title, reason in failed:
            print(f"  - {title[:70]} ({reason})")
    
    if recovered:
        print("\n=== Next Steps ===")
        print("To re-ingest the recovered PDFs, run:")
        print("  python scripts/batch_reingest.py")
        print("\nOr re-ingest specific papers:")
        for _, _, path in recovered:
            print(f"  python scripts/main.py -f \"{path}\"")


if __name__ == "__main__":
    recover_pdfs()
