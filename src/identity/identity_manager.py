"""
Identity Management Module
Maintains consistent person IDs across cameras and time
"""

from typing import Dict, List, Set
from collections import defaultdict


class IdentityManager:
    """
    Manages global identity assignment across multiple cameras
    Maps local track IDs to global person IDs
    """
    
    def __init__(self):
        """Initialize identity manager"""
        # Mapping: (camera_id, track_id) -> global_id
        self.local_to_global = {}
        
        # Mapping: global_id -> set of (camera_id, track_id)
        self.global_to_local = defaultdict(set)
        
        # Next available global ID
        self.next_global_id = 1
        
        # Active global IDs
        self.active_ids = set()
    
    def assign_global_id(self, camera_id: int, track_id: int) -> int:
        """
        Assign or retrieve global ID for a local track
        
        Args:
            camera_id: Camera identifier
            track_id: Local track ID
            
        Returns:
            Global person ID
        """
        key = (camera_id, track_id)
        
        # Check if already assigned
        if key in self.local_to_global:
            global_id = self.local_to_global[key]
            self.active_ids.add(global_id)
            return global_id
        
        # Assign new global ID
        global_id = self.next_global_id
        self.next_global_id += 1
        
        # Store mapping
        self.local_to_global[key] = global_id
        self.global_to_local[global_id].add(key)
        self.active_ids.add(global_id)
        
        return global_id
    
    def get_global_id(self, camera_id: int, track_id: int) -> int:
        """
        Get global ID for a local track (returns -1 if not found)
        
        Args:
            camera_id: Camera identifier
            track_id: Local track ID
            
        Returns:
            Global person ID or -1
        """
        key = (camera_id, track_id)
        return self.local_to_global.get(key, -1)
    
    def mark_inactive(self, camera_id: int, track_id: int):
        """
        Mark a local track as inactive
        
        Args:
            camera_id: Camera identifier
            track_id: Local track ID
        """
        key = (camera_id, track_id)
        if key in self.local_to_global:
            global_id = self.local_to_global[key]
            
            # Remove from active set if no other cameras tracking this person
            all_tracks = self.global_to_local[global_id]
            if len(all_tracks) == 1 and key in all_tracks:
                self.active_ids.discard(global_id)
    
    def get_active_count(self) -> int:
        """Get count of currently active global IDs"""
        return len(self.active_ids)
    
    def get_total_count(self) -> int:
        """Get total count of assigned global IDs"""
        return self.next_global_id - 1
    
    def reset(self):
        """Reset all identity mappings"""
        self.local_to_global.clear()
        self.global_to_local.clear()
        self.next_global_id = 1
        self.active_ids.clear()
    
    def merge_identities(self, global_id1: int, global_id2: int):
        """
        Merge two global identities (for future ReID)
        
        Args:
            global_id1: First global ID (will be kept)
            global_id2: Second global ID (will be merged into first)
        """
        if global_id2 not in self.global_to_local:
            return
        
        # Get all local tracks for second ID
        tracks_to_merge = self.global_to_local[global_id2]
        
        # Reassign to first ID
        for track_key in tracks_to_merge:
            self.local_to_global[track_key] = global_id1
            self.global_to_local[global_id1].add(track_key)
        
        # Remove second ID
        del self.global_to_local[global_id2]
        self.active_ids.discard(global_id2)
