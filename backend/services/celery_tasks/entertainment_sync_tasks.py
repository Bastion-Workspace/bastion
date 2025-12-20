"""
Entertainment Sync Celery Tasks
Background processing for Sonarr/Radarr content synchronization
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID

from services.celery_app import celery_app, update_task_progress, TaskStatus
from celery.exceptions import SoftTimeLimitExceeded

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="services.celery_tasks.entertainment_sync_tasks.scheduled_entertainment_sync")
def scheduled_entertainment_sync_task(self) -> Dict[str, Any]:
    """
    Scheduled task (runs every hour via Celery Beat)
    - Find all enabled sync configs
    - Queue individual sync tasks for each config
    - Track overall progress
    """
    try:
        logger.info("🎬 ENTERTAINMENT SYNC: Starting scheduled sync task")
        
        update_task_progress(self, 1, 3, "Fetching enabled sync configurations...")
        
        # Create new event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_async_scheduled_sync(self))
        finally:
            loop.close()
        
        logger.info("✅ ENTERTAINMENT SYNC: Scheduled sync task completed")
        return result
        
    except Exception as e:
        logger.error(f"❌ ENTERTAINMENT SYNC ERROR: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Scheduled entertainment sync failed"
        }


async def _async_scheduled_sync(task) -> Dict[str, Any]:
    """Async helper for scheduled sync"""
    from services.entertainment_sync_service import EntertainmentSyncService
    
    sync_service = EntertainmentSyncService()
    configs = await sync_service.get_enabled_configs()
    
    if not configs:
        logger.info("No enabled sync configurations found")
        return {
            "success": True,
            "configs_processed": 0,
            "message": "No enabled sync configurations"
        }
    
    logger.info(f"Found {len(configs)} enabled sync configurations")
    
    # Queue individual sync tasks
    task_ids = []
    for config in configs:
        config_id = str(config['config_id'])
        try:
            sync_task = sync_entertainment_source_task.delay(config_id)
            task_ids.append(sync_task.id)
            logger.info(f"Queued sync task for config {config_id}: {sync_task.id}")
        except Exception as e:
            logger.error(f"Failed to queue sync task for config {config_id}: {e}")
    
    update_task_progress(task, 3, 3, f"Queued {len(task_ids)} sync tasks")
    
    return {
        "success": True,
        "configs_found": len(configs),
        "tasks_queued": len(task_ids),
        "task_ids": task_ids,
        "message": f"Queued {len(task_ids)} sync tasks"
    }


@celery_app.task(bind=True, name="services.celery_tasks.entertainment_sync_tasks.sync_entertainment_source")
def sync_entertainment_source_task(self, config_id: str) -> Dict[str, Any]:
    """
    Sync a single Radarr or Sonarr source
    - Fetch all items from API
    - Compare with stored items (detect adds/updates/deletes)
    - Process new/updated items (transform -> chunk -> embed -> store)
    - Delete removed items from vectors
    - Update sync state
    """
    try:
        logger.info(f"🎬 ENTERTAINMENT SYNC: Starting sync for config {config_id}")
        
        update_task_progress(self, 1, 5, "Initializing sync...")
        
        # Create new event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_async_sync_source(self, config_id))
        finally:
            loop.close()
        
        logger.info(f"✅ ENTERTAINMENT SYNC: Completed sync for config {config_id}")
        return result
        
    except SoftTimeLimitExceeded as e:
        logger.error(f"❌ ENTERTAINMENT SYNC: Time limit exceeded for config {config_id}")
        return {
            "success": False,
            "error": "TimeLimitExceeded",
            "message": "Sync exceeded time limit"
        }
    except Exception as e:
        logger.error(f"❌ ENTERTAINMENT SYNC ERROR: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Sync failed for config {config_id}"
        }


async def _async_sync_source(task, config_id: str) -> Dict[str, Any]:
    """Async helper for syncing a source"""
    from services.entertainment_sync_service import EntertainmentSyncService
    from services.radarr_service import RadarrService
    from services.sonarr_service import SonarrService
    
    sync_service = EntertainmentSyncService()
    
    # Get configuration with decrypted API key
    config = await sync_service.get_config_with_api_key(UUID(config_id), None)
    if not config:
        return {
            "success": False,
            "error": "Configuration not found",
            "message": f"Config {config_id} not found"
        }
    
    user_id = config['user_id']
    source_type = config['source_type']
    api_url = config['api_url']
    api_key = config['api_key']
    
    # Update status to running
    await sync_service.update_sync_status(config_id, 'running')
    
    update_task_progress(task, 2, 5, f"Fetching {source_type} content...")
    
    try:
        # Fetch items from API
        if source_type == 'radarr':
            service = RadarrService(api_url, api_key)
            api_items = await service.fetch_all_movies()
            item_type = 'movie'
        elif source_type == 'sonarr':
            service = SonarrService(api_url, api_key)
            series_list = await service.fetch_all_series()
            
            # Fetch episodes for each series
            api_items = []
            for series in series_list:
                # Add series as an item
                api_items.append({
                    'type': 'series',
                    'data': series
                })
                
                # Fetch episodes
                episodes = await service.fetch_series_episodes(series['id'])
                for episode in episodes:
                    api_items.append({
                        'type': 'episode',
                        'data': episode,
                        'series': series
                    })
            item_type = 'series'
        else:
            return {
                "success": False,
                "error": f"Unknown source type: {source_type}"
            }
        
        logger.info(f"Fetched {len(api_items)} items from {source_type}")
        
        update_task_progress(task, 3, 5, "Detecting changes...")
        
        # Detect changes
        changes = await _detect_changes(sync_service, UUID(config_id), api_items, source_type)
        
        logger.info(f"Changes detected: {len(changes['added'])} added, "
                   f"{len(changes['updated'])} updated, {len(changes['deleted'])} deleted")
        
        update_task_progress(task, 4, 5, f"Processing {len(changes['added']) + len(changes['updated'])} items...")
        
        # Process new and updated items
        processed_count = 0
        for item_data in changes['added'] + changes['updated']:
            try:
                process_entertainment_item_task.delay(
                    config_id,
                    item_data,
                    source_type
                )
                processed_count += 1
            except Exception as e:
                logger.error(f"Failed to queue item processing: {e}")
        
        # Handle deletions
        if changes['deleted']:
            await _handle_deletions(sync_service, UUID(config_id), changes['deleted'], user_id)
        
        # Update sync status
        total_items = len(api_items) if source_type == 'radarr' else len([i for i in api_items if i['type'] == 'series'])
        await sync_service.update_sync_status(
            UUID(config_id),
            'success',
            items_synced=total_items
        )
        
        update_task_progress(task, 5, 5, "Sync completed")
        
        return {
            "success": True,
            "items_fetched": len(api_items),
            "items_added": len(changes['added']),
            "items_updated": len(changes['updated']),
            "items_deleted": len(changes['deleted']),
            "items_queued": processed_count,
            "message": f"Sync completed: {processed_count} items queued for processing"
        }
        
    except Exception as e:
        logger.error(f"Sync failed for config {config_id}: {e}")
        await sync_service.update_sync_status(
            UUID(config_id),
            'failed',
            error=str(e)
        )
        raise


async def _detect_changes(
    sync_service: 'EntertainmentSyncService',
    config_id: UUID,
    api_items: List[Dict[str, Any]],
    source_type: str
) -> Dict[str, List[Any]]:
    """Detect adds, updates, and deletes"""
    # Get existing items
    existing_items = await sync_service.get_sync_items(config_id)
    
    # Build maps for comparison
    existing_map = {}
    for item in existing_items:
        key = f"{item.external_id}_{item.external_type}"
        existing_map[key] = item
    
    # Process API items
    added = []
    updated = []
    api_ids = set()
    
    for item_data in api_items:
        if source_type == 'radarr':
            external_id = str(item_data['id'])
            external_type = 'movie'
            item = item_data
        else:  # sonarr
            item_type = item_data['type']
            if item_type == 'series':
                external_id = str(item_data['data']['id'])
                external_type = 'series'
                item = item_data['data']
            else:  # episode
                external_id = str(item_data['data']['id'])
                external_type = 'episode'
                item = item_data['data']
                series = item_data['series']
        
        key = f"{external_id}_{external_type}"
        api_ids.add(key)
        
        # Calculate metadata hash
        if source_type == 'radarr':
            from services.radarr_service import RadarrService
            service = RadarrService("", "")  # Dummy service for hash calculation
            metadata_hash = service.calculate_metadata_hash(item)
        else:
            from services.sonarr_service import SonarrService
            service = SonarrService("", "")
            metadata_hash = service.calculate_metadata_hash(item)
        
        if key in existing_map:
            # Check if updated
            existing_item = existing_map[key]
            if existing_item.metadata_hash != metadata_hash:
                if source_type == 'radarr':
                    updated.append(item)
                else:
                    if item_type == 'series':
                        updated.append({'type': 'series', 'data': item})
                    else:
                        updated.append({'type': 'episode', 'data': item, 'series': series})
        else:
            # New item
            if source_type == 'radarr':
                added.append(item)
            else:
                if item_type == 'series':
                    added.append({'type': 'series', 'data': item})
                else:
                    added.append({'type': 'episode', 'data': item, 'series': series})
    
    # Find deleted items
    deleted = []
    for key, item in existing_map.items():
        if key not in api_ids:
            deleted.append(item)
    
    return {
        'added': added,
        'updated': updated,
        'deleted': deleted
    }


async def _handle_deletions(
    sync_service: 'EntertainmentSyncService',
    config_id: UUID,
    deleted_items: List[Any],
    user_id: str
):
    """Handle deletion of items from vector database"""
    from services.service_container import get_service_container
    
    service_container = await get_service_container()
    
    for item in deleted_items:
        try:
            vector_doc_id = item.vector_document_id
            if vector_doc_id:
                # Delete from Qdrant via embedding_manager
                if hasattr(service_container, 'embedding_manager') and service_container.embedding_manager:
                    await service_container.embedding_manager.delete_document_chunks(
                        vector_doc_id,
                        user_id
                    )
                
                # Delete from Neo4j
                if hasattr(service_container, 'knowledge_graph_service') and service_container.knowledge_graph_service:
                    await service_container.knowledge_graph_service.delete_document_entities(vector_doc_id)
            
            # Delete from tracking table
            await sync_service.delete_sync_item(item.item_id)
            
            logger.info(f"Deleted item {item.external_id} ({item.external_type})")
        except Exception as e:
            logger.error(f"Failed to delete item {item.item_id}: {e}")


@celery_app.task(bind=True, name="services.celery_tasks.entertainment_sync_tasks.process_entertainment_item")
def process_entertainment_item_task(
    self,
    config_id: str,
    item_data: Dict[str, Any],
    source_type: str
) -> Dict[str, Any]:
    """
    Process individual item (movie/series/episode)
    - Transform to text
    - Generate chunks
    - Create embeddings
    - Store in user's Qdrant collection with proper tags
    - Extract entities for Neo4j
    - Update sync_items table
    """
    try:
        logger.info(f"🎬 PROCESSING: Starting item processing for config {config_id}")
        
        # Create new event loop
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_async_process_item(self, config_id, item_data, source_type))
        finally:
            loop.close()
        
        logger.info(f"✅ PROCESSING: Completed item processing")
        return result
        
    except Exception as e:
        logger.error(f"❌ PROCESSING ERROR: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Item processing failed"
        }


async def _async_process_item(
    task,
    config_id: str,
    item_data: Dict[str, Any],
    source_type: str
) -> Dict[str, Any]:
    """Async helper for processing an item"""
    from services.entertainment_sync_service import EntertainmentSyncService
    from services.radarr_service import RadarrService
    from services.sonarr_service import SonarrService
    from services.service_container import get_service_container
    from utils.document_processor import DocumentProcessor
    
    sync_service = EntertainmentSyncService()
    
    # Get configuration
    config = await sync_service.get_config_with_api_key(UUID(config_id), None)
    if not config:
        return {"success": False, "error": "Configuration not found"}
    
    user_id = config['user_id']
    api_url = config['api_url']
    api_key = config['api_key']
    
    # Get services
    service_container = await get_service_container()
    
    # Ensure embedding_manager is initialized in this event loop context
    # In Celery workers, we need to re-initialize to ensure gRPC channels use the current event loop
    if service_container.embedding_manager:
        # Force re-initialization in Celery worker context to ensure event loop compatibility
        if hasattr(service_container.embedding_manager, 'vector_service_client') and service_container.embedding_manager.vector_service_client:
            # Close existing client if it exists (may be tied to different event loop)
            try:
                await service_container.embedding_manager.vector_service_client.close()
            except:
                pass
            service_container.embedding_manager.vector_service_client = None
            service_container.embedding_manager._initialized = False
        await service_container.embedding_manager.initialize()
    
    # Ensure knowledge_graph_service is initialized in this event loop context
    # Neo4j driver must be created in the current event loop
    if service_container.knowledge_graph_service:
        # Check if driver exists and close it if it's tied to a different event loop
        if hasattr(service_container.knowledge_graph_service, 'driver') and service_container.knowledge_graph_service.driver:
            try:
                await service_container.knowledge_graph_service.driver.close()
            except:
                pass
            service_container.knowledge_graph_service.driver = None
        # Re-initialize in current event loop context
        await service_container.knowledge_graph_service.initialize()
    
    document_processor = DocumentProcessor.get_instance()
    await document_processor.initialize()
    
    # Process based on source type
    if source_type == 'radarr':
        # Process movie
        service = RadarrService(api_url, api_key)
        movie = item_data
        
        # Validate data before processing
        is_valid, error_msg = service.validate_movie_data(movie)
        if not is_valid:
            logger.error(f"❌ Validation failed for Radarr movie: {error_msg}")
            return {
                "success": False,
                "error": f"Data validation failed: {error_msg}",
                "message": "Item processing failed due to invalid data"
            }
        
        # Transform to text
        text_content = service.transform_movie_to_text(movie)
        tags = service.extract_tags(movie)
        metadata_hash = service.calculate_metadata_hash(movie)
        
        external_id = str(movie['id'])
        external_type = 'movie'
        title = movie.get('title', 'Unknown')
        tmdb_id = movie.get('tmdbId')
        vector_doc_id = f"entertainment_radarr_{external_id}"
        
    else:  # sonarr
        service = SonarrService(api_url, api_key)
        item_type = item_data['type']
        
        if item_type == 'series':
            series = item_data['data']
            
            # Validate data before processing
            is_valid, error_msg = service.validate_series_data(series)
            if not is_valid:
                logger.error(f"❌ Validation failed for Sonarr series: {error_msg}")
                return {
                    "success": False,
                    "error": f"Data validation failed: {error_msg}",
                    "message": "Item processing failed due to invalid data"
                }
            
            text_content = service.transform_series_to_text(series)
            tags = service.extract_tags(series)
            metadata_hash = service.calculate_metadata_hash(series)
            
            external_id = str(series['id'])
            external_type = 'series'
            title = series.get('title', 'Unknown')
            tvdb_id = series.get('tvdbId')
            vector_doc_id = f"entertainment_sonarr_series_{external_id}"
        else:  # episode
            episode = item_data['data']
            series = item_data['series']
            
            # Validate data before processing
            is_valid, error_msg = service.validate_episode_data(episode, series)
            if not is_valid:
                logger.error(f"❌ Validation failed for Sonarr episode: {error_msg}")
                return {
                    "success": False,
                    "error": f"Data validation failed: {error_msg}",
                    "message": "Item processing failed due to invalid data"
                }
            
            text_content = service.transform_episode_to_text(episode, series)
            tags = service.extract_tags(series, episode)
            metadata_hash = service.calculate_metadata_hash(episode)
            
            external_id = str(episode['id'])
            external_type = 'episode'
            title = f"{series.get('title', 'Unknown')} - S{episode.get('seasonNumber', 0):02d}E{episode.get('episodeNumber', 0):02d}"
            tvdb_id = episode.get('tvdbId')
            season_num = episode.get('seasonNumber')
            episode_num = episode.get('episodeNumber')
            parent_series_id = str(series['id'])
            vector_doc_id = f"entertainment_sonarr_episode_{external_id}"
    
    # Generate chunks
    metadata = {
        'source': source_type,
        'external_id': external_id,
        'external_type': external_type,
        'title': title
    }
    
    if source_type == 'radarr':
        if tmdb_id:
            metadata['tmdb_id'] = tmdb_id
    else:
        if tvdb_id:
            metadata['tvdb_id'] = tvdb_id
        if external_type == 'episode':
            metadata['season_number'] = season_num
            metadata['episode_number'] = episode_num
            metadata['parent_series_id'] = parent_series_id
    
    chunks = await document_processor.process_text_content(
        text_content,
        vector_doc_id,
        metadata
    )
    
    if not chunks:
        return {"success": False, "error": "Failed to generate chunks"}
    
    # Store embeddings
    await service_container.embedding_manager.embed_and_store_chunks(
        chunks,
        user_id=user_id,
        document_category='entertainment',
        document_tags=tags,
        document_title=title
    )
    
    # Extract and store entities
    if service_container.knowledge_graph_service:
        from services.entertainment_kg_extractor import get_entertainment_kg_extractor
        from models.api_models import DocumentInfo, DocumentType, ProcessingStatus, DocumentCategory
        from datetime import datetime
        
        # Create a pseudo DocumentInfo for KG extraction
        doc_info = DocumentInfo(
            document_id=vector_doc_id,
            filename=f"{title}.txt",
            doc_type=DocumentType.TXT,
            tags=tags,
            category=DocumentCategory.ENTERTAINMENT,
            upload_date=datetime.now(),
            file_size=len(text_content.encode('utf-8')),  # Approximate file size in bytes
            status=ProcessingStatus.COMPLETED
        )
        
        kg_extractor = get_entertainment_kg_extractor()
        entities, relationships = kg_extractor.extract_entities_and_relationships(
            text_content,
            doc_info
        )
        
        if entities:
            # Convert dictionaries to Entity objects
            from models.api_models import Entity
            entity_objects = []
            for entity_dict in entities:
                entity_objects.append(Entity(
                    name=entity_dict.get('name', ''),
                    entity_type=entity_dict.get('type', 'OTHER'),
                    confidence=entity_dict.get('confidence', 0.5),
                    source_chunk=vector_doc_id,  # Use document ID as source chunk
                    metadata=entity_dict.get('properties', {})
                ))
            
            await service_container.knowledge_graph_service.store_entities(entity_objects, vector_doc_id)
    
    # Update sync_items table
    if source_type == 'radarr':
        item_id = await sync_service.upsert_sync_item(
            UUID(config_id),
            external_id,
            external_type,
            title,
            metadata_hash,
            vector_doc_id,
            tmdb_id=tmdb_id
        )
    else:
        if item_type == 'series':
            item_id = await sync_service.upsert_sync_item(
                UUID(config_id),
                external_id,
                external_type,
                title,
                metadata_hash,
                vector_doc_id,
                tvdb_id=tvdb_id
            )
        else:  # episode
            item_id = await sync_service.upsert_sync_item(
                UUID(config_id),
                external_id,
                external_type,
                title,
                metadata_hash,
                vector_doc_id,
                tvdb_id=tvdb_id,
                season_number=season_num,
                episode_number=episode_num,
                parent_series_id=parent_series_id
            )
    
    # Post-processing verification: Verify stored data matches what we processed
    try:
        # Get all sync items for this config and find the one we just stored
        all_items = await sync_service.get_sync_items(UUID(config_id))
        stored_item = None
        for item in all_items:
            if item.external_id == external_id and item.external_type == external_type:
                stored_item = item
                break
        
        if not stored_item:
            logger.warning(f"⚠️ Verification failed: Stored item not found for {external_type} {external_id}")
            return {
                "success": False,
                "error": "Post-processing verification failed: Item not found in database",
                "message": "Item processing completed but verification failed"
            }
        
        # Verify critical fields match
        verification_errors = []
        if stored_item.title != title:
            verification_errors.append(f"Title mismatch: stored '{stored_item.title}' vs processed '{title}'")
        if stored_item.metadata_hash and stored_item.metadata_hash != metadata_hash:
            verification_errors.append(f"Metadata hash mismatch: stored '{stored_item.metadata_hash}' vs processed '{metadata_hash}'")
        if stored_item.external_id != external_id:
            verification_errors.append(f"External ID mismatch: stored '{stored_item.external_id}' vs processed '{external_id}'")
        
        if verification_errors:
            logger.error(f"❌ Post-processing verification failed for {external_type} {external_id}: {'; '.join(verification_errors)}")
            return {
                "success": False,
                "error": f"Post-processing verification failed: {'; '.join(verification_errors)}",
                "message": "Item processing completed but stored data doesn't match processed data"
            }
        
        logger.info(f"✅ Post-processing verification passed for {external_type} {external_id} ({title})")
        
    except Exception as e:
        logger.warning(f"⚠️ Post-processing verification error (non-fatal): {e}")
        # Don't fail the whole operation if verification has issues
    
    return {
        "success": True,
        "chunks_created": len(chunks),
        "vector_doc_id": vector_doc_id,
        "message": f"Successfully processed and verified {external_type} {title}"
    }

