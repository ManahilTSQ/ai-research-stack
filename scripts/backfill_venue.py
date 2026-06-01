"""
backfill_venue.py — Backfill venue information for existing papers in ChromaDB.

This script queries Semantic Scholar to retrieve venue information for papers
that were ingested before the venue field was added to the metadata schema.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import settings
from vector_store import VectorStoreService
from manifest_manager import ManifestManagerService
from paper_discovery import PaperDiscoveryService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def backfill_venue_for_paper(title: str, doi: str, vector_store, discovery_service) -> bool:
    """
    Query Semantic Scholar for venue information and update ChromaDB metadata.
    
    Args:
        title: Paper title
        doi: Paper DOI (may be None or "N/A")
        vector_store: VectorStoreService instance
        discovery_service: PaperDiscoveryService instance
    
    Returns:
        True if venue was successfully updated, False otherwise
    """
    try:
        # Try DOI lookup first if available
        if doi and doi != "N/A":
            logger.info(f"Looking up venue for DOI: {doi}")
            paper_details = discovery_service.get_paper_details(doi)
            if paper_details:
                venue = paper_details.get("venue") or "N/A"
                if venue != "N/A":
                    # Update metadata in ChromaDB
                    success = vector_store.update_paper_metadata(
                        title=title,
                        authors=None,  # Don't change authors
                        year=None,     # Don't change year
                        doi=doi,
                        venue=venue,
                        new_title=None  # Don't change title
                    )
                    if success:
                        logger.info(f"✓ Updated venue for '{title}': {venue}")
                        return True
                    else:
                        logger.warning(f"✗ Failed to update venue for '{title}'")
                        return False
        
        # If DOI lookup failed or no DOI, try title search
        logger.info(f"Searching by title: {title[:80]}...")
        results = discovery_service.search_papers(title, limit=5)
        
        for result in results:
            result_title = (result.get("title") or "").lower().strip()
            title_lower = title.lower().strip()
            
            # Check for title match (allowing for minor variations)
            if (result_title == title_lower or 
                result_title in title_lower or 
                title_lower in result_title):
                
                venue = result.get("venue") or "N/A"
                result_doi = (result.get("externalIds") or {}).get("DOI") or doi
                
                if venue != "N/A":
                    # Update metadata in ChromaDB
                    success = vector_store.update_paper_metadata(
                        title=title,
                        authors=None,  # Don't change authors
                        year=None,     # Don't change year
                        doi=result_doi if result_doi != "N/A" else None,
                        venue=venue,
                        new_title=None  # Don't change title
                    )
                    if success:
                        logger.info(f"✓ Updated venue for '{title}': {venue}")
                        return True
                    else:
                        logger.warning(f"✗ Failed to update venue for '{title}'")
                        return False
        
        logger.warning(f"✗ No venue found for '{title}'")
        return False
        
    except Exception as e:
        logger.error(f"✗ Error backfilling venue for '{title}': {e}")
        return False


def main():
    """Main entry point for venue backfill."""
    logger.info("=" * 60)
    logger.info("VENUE BACKFILL UTILITY")
    logger.info("=" * 60)
    
    # Initialize services
    vector_store = VectorStoreService()
    manifest_service = ManifestManagerService()
    discovery_service = PaperDiscoveryService()
    
    # Get all papers from ChromaDB
    stats = vector_store.get_collection_stats()
    papers_metadata = stats.get("papers_metadata", {})
    total_papers = len(papers_metadata)
    
    logger.info(f"Found {total_papers} papers in ChromaDB")
    
    if total_papers == 0:
        logger.info("No papers to backfill. Exiting.")
        return
    
    # Count papers without venue
    papers_without_venue = [
        (title, meta) for title, meta in papers_metadata.items()
        if meta.get("venue", "N/A") == "N/A"
    ]
    
    logger.info(f"Papers without venue: {len(papers_without_venue)}")
    
    if len(papers_without_venue) == 0:
        logger.info("All papers already have venue information. Exiting.")
        return
    
    # Confirm with user
    print(f"\nThis will attempt to backfill venue information for {len(papers_without_venue)} papers.")
    print("Each paper will be queried against Semantic Scholar.")
    print("Continue? (y/n): ", end="")
    
    try:
        response = input().strip().lower()
        if response != 'y':
            logger.info("Backfill cancelled by user.")
            return
    except (EOFError, KeyboardInterrupt):
        logger.info("Backfill cancelled.")
        return
    
    # Process each paper
    success_count = 0
    fail_count = 0
    
    for idx, (title, meta) in enumerate(papers_without_venue, 1):
        logger.info(f"\n[{idx}/{len(papers_without_venue)}] Processing: {title[:80]}...")
        
        doi = meta.get("doi", "N/A")
        
        if backfill_venue_for_paper(title, doi, vector_store, discovery_service):
            success_count += 1
        else:
            fail_count += 1
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total papers processed: {len(papers_without_venue)}")
    logger.info(f"Successfully updated: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info("=" * 60)
    
    # Update manifest to reflect new venue data
    logger.info("\nSyncing manifest with updated ChromaDB metadata...")
    manifest_service.sync_with_vector_store(vector_store)
    logger.info("Manifest sync complete.")


if __name__ == "__main__":
    main()
