"""
Cleanup Duplicate Folders Utility
Roosevelt's Trust-Busting Tool for Folder Deduplication

This utility finds and merges duplicate folders that were created
due to race conditions during concurrent file syncing.

Usage:
    python -m utils.cleanup_duplicate_folders

Note: The SQL migration (03_fix_folder_duplication.sql) handles this automatically,
but this script can be run manually if needed for debugging or re-cleaning.
"""

import asyncio
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def find_duplicate_folders() -> List[Dict[str, Any]]:
    """
    Find all duplicate folder combinations
    Returns list of duplicate groups with folder_ids to merge
    """
    try:
        from services.database_manager.database_helpers import fetch_all
        
        logger.info("Searching for duplicate folders")
        
        duplicates = await fetch_all("""
            SELECT 
                user_id, 
                name, 
                parent_folder_id, 
                collection_type,
                array_agg(folder_id ORDER BY created_at ASC) as folder_ids,
                array_agg(created_at ORDER BY created_at ASC) as created_dates,
                COUNT(*) as duplicate_count
            FROM document_folders
            GROUP BY user_id, name, parent_folder_id, collection_type
            HAVING COUNT(*) > 1
            ORDER BY COUNT(*) DESC, name
        """)
        
        if not duplicates:
            logger.info("✅ No duplicate folders found! System is clean!")
            return []
        
        logger.info(f"📁 Found {len(duplicates)} duplicate folder groups")
        
        total_duplicates = sum(row['duplicate_count'] - 1 for row in duplicates)
        logger.info(f"📊 Total duplicate instances to merge: {total_duplicates}")
        
        return duplicates
        
    except Exception as e:
        logger.error(f"❌ Error finding duplicates: {e}")
        raise


async def merge_duplicate_folder(duplicate_info: Dict[str, Any]) -> int:
    """
    Merge a group of duplicate folders into the oldest one
    Returns number of duplicate folders merged
    """
    try:
        from services.database_manager.database_helpers import execute, fetch_one
        
        name = duplicate_info['name']
        folder_ids = duplicate_info['folder_ids']
        keeper_id = folder_ids[0]  # Keep oldest
        duplicate_ids = folder_ids[1:]  # Merge these
        
        user_id = duplicate_info['user_id'] or 'GLOBAL'
        parent_id = duplicate_info['parent_folder_id'] or 'ROOT'
        
        logger.info(f"📁 Merging '{name}' (parent: {parent_id}, user: {user_id})")
        logger.info(f"   Keeping: {keeper_id} (created: {duplicate_info['created_dates'][0]})")
        
        total_docs_moved = 0
        total_folders_moved = 0
        
        for i, dup_id in enumerate(duplicate_ids):
            created_date = duplicate_info['created_dates'][i + 1]
            logger.info(f"   Merging: {dup_id} (created: {created_date})")
            
            # Move documents from duplicate to keeper
            result = await execute("""
                UPDATE document_metadata 
                SET folder_id = $1 
                WHERE folder_id = $2
            """, keeper_id, dup_id)
            
            docs_moved = result if isinstance(result, int) else 0
            if docs_moved > 0:
                logger.info(f"      → Moved {docs_moved} documents")
                total_docs_moved += docs_moved
            
            # Move subfolders from duplicate to keeper
            result = await execute("""
                UPDATE document_folders
                SET parent_folder_id = $1
                WHERE parent_folder_id = $2
            """, keeper_id, dup_id)
            
            folders_moved = result if isinstance(result, int) else 0
            if folders_moved > 0:
                logger.info(f"      → Moved {folders_moved} subfolders")
                total_folders_moved += folders_moved
            
            # Delete the duplicate folder
            await execute("DELETE FROM document_folders WHERE folder_id = $1", dup_id)
            logger.info(f"      ✅ Deleted duplicate: {dup_id}")
        
        logger.info(f"   ✅ Merge complete: {len(duplicate_ids)} duplicates → 1 keeper")
        logger.info(f"   📊 Totals: {total_docs_moved} docs, {total_folders_moved} subfolders moved")
        
        return len(duplicate_ids)
        
    except Exception as e:
        logger.error(f"❌ Error merging duplicate folder '{name}': {e}")
        raise


async def cleanup_duplicate_folders():
    """
    Main cleanup function
    Finds and merges all duplicate folders
    """
    try:
        logger.info("Starting duplicate folder cleanup")
        logger.info("=" * 70)
        
        # Find all duplicates
        duplicates = await find_duplicate_folders()
        
        if not duplicates:
            logger.info("✅ No cleanup needed - all folders are unique!")
            return
        
        logger.info("=" * 70)
        logger.info("🔧 Starting merge operations...")
        logger.info("")
        
        # Merge each duplicate group
        total_merged = 0
        for dup_info in duplicates:
            merged_count = await merge_duplicate_folder(dup_info)
            total_merged += merged_count
            logger.info("")  # Blank line between groups
        
        logger.info("=" * 70)
        logger.info("Duplicate folder cleanup complete")
        logger.info(f"📊 Final Statistics:")
        logger.info(f"   - Duplicate folder groups found: {len(duplicates)}")
        logger.info(f"   - Duplicate instances merged: {total_merged}")
        logger.info(f"   - System now has unique folders only!")
        logger.info("🎯 Trust-busting successful! Race conditions defeated!")
        
    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        raise


async def verify_no_duplicates():
    """
    Verify that no duplicate folders exist
    Returns True if clean, False if duplicates still exist
    """
    try:
        duplicates = await find_duplicate_folders()
        
        if not duplicates:
            logger.info("✅ VERIFICATION PASSED: No duplicate folders exist")
            return True
        else:
            logger.warning(f"⚠️ VERIFICATION FAILED: {len(duplicates)} duplicate groups still exist")
            return False
            
    except Exception as e:
        logger.error(f"❌ Verification error: {e}")
        return False


async def main():
    """Main entry point"""
    try:
        # Run cleanup
        await cleanup_duplicate_folders()
        
        # Verify results
        logger.info("")
        logger.info("🔍 Running verification...")
        success = await verify_no_duplicates()
        
        if success:
            logger.info("🎉 All systems operational! Folder structure is clean!")
            return 0
        else:
            logger.error("❌ Verification failed - manual investigation needed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Cleanup utility failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)













