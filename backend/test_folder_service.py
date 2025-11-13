#!/usr/bin/env python3
"""
Simple test script for FolderService
Run with: python test_folder_service.py
"""

import asyncio
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.folder_service import FolderService
from models.api_models import FolderCreateRequest, FolderUpdateRequest

async def test_folder_service():
    """Test basic folder service functionality"""
    print("🧪 Testing FolderService...")
    
    # Initialize service
    folder_service = FolderService()
    await folder_service.initialize()
    
    # Test user ID (you can change this)
    test_user_id = "test-user-123"
    
    try:
        # Test 1: Create root folder
        print("\n📁 Test 1: Creating root folder...")
        root_folder = await folder_service.create_folder(
            name="Test Root",
            user_id=test_user_id,
            collection_type="user"
        )
        print(f"✅ Created root folder: {root_folder.name} ({root_folder.folder_id})")
        
        # Test 2: Create subfolder
        print("\n📁 Test 2: Creating subfolder...")
        subfolder = await folder_service.create_folder(
            name="Test Subfolder",
            parent_folder_id=root_folder.folder_id,
            user_id=test_user_id,
            collection_type="user"
        )
        print(f"✅ Created subfolder: {subfolder.name} ({subfolder.folder_id})")
        
        # Test 3: Get folder tree
        print("\n📁 Test 3: Getting folder tree...")
        folder_tree = await folder_service.get_folder_tree(test_user_id, "user")
        print(f"✅ Folder tree has {len(folder_tree)} root folders")
        
        # Test 4: Get folder contents
        print("\n📁 Test 4: Getting folder contents...")
        contents = await folder_service.get_folder_contents(root_folder.folder_id, test_user_id)
        if contents:
            print(f"✅ Folder contains {contents.total_subfolders} subfolders and {contents.total_documents} documents")
        
        # Test 5: Update folder
        print("\n📁 Test 5: Updating folder...")
        update_request = FolderUpdateRequest(name="Updated Root Folder")
        updated_folder = await folder_service.update_folder(root_folder.folder_id, update_request, test_user_id)
        if updated_folder:
            print(f"✅ Updated folder name to: {updated_folder.name}")
        
        # Test 6: Create default folders
        print("\n📁 Test 6: Creating default folders...")
        default_folders = await folder_service.create_default_folders(test_user_id)
        print(f"✅ Created {len(default_folders)} default folders")
        
        # Test 7: Get updated folder tree
        print("\n📁 Test 7: Getting updated folder tree...")
        updated_tree = await folder_service.get_folder_tree(test_user_id, "user")
        print(f"✅ Updated folder tree has {len(updated_tree)} root folders")
        
        # Test 8: Delete test folders
        print("\n📁 Test 8: Cleaning up test folders...")
        await folder_service.delete_folder(root_folder.folder_id, test_user_id, recursive=True)
        print("✅ Cleaned up test folders")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_folder_service()) 