from typing import List

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        # Edge case: no meetings at all -> 0 rooms needed
        n = len(intervals)
        if n == 0:
            return 0
        
        # Edge case: single meeting -> exactly 1 room needed
        if n == 1:
            return 1
        
        # -------------------------------------------------------------
        # Key insight:
        # Two meetings overlap if start_i <= end_j (touching counts as
        # overlap here, unlike the "classic" meeting rooms problem where
        # touching endpoints are allowed to share a room).
        #
        # So a room becomes FREE for reuse only when the new meeting's
        # start is STRICTLY GREATER than some existing meeting's end
        # (start > end  =>  no overlap).
        # If start <= end, they still overlap -> need a new room.
        # -------------------------------------------------------------
        
        # Separate and sort start times and end times independently.
        # Sorting them independently (not sorting the pairs) is the
        # classic trick that lets us use two pointers in O(N log N).
        starts = sorted(interval[0] for interval in intervals)
        ends = sorted(interval[1] for interval in intervals)
        
        rooms_in_use = 0      # current number of rooms occupied
        max_rooms = 0         # answer: peak rooms needed simultaneously
        end_ptr = 0            # pointer into sorted `ends`
        
        # Walk through meetings in order of their start time.
        for start_ptr in range(n):
            current_start = starts[start_ptr]
            
            # While the earliest-ending active meeting truly finished
            # before this meeting starts (strict inequality, because
            # start == end still counts as overlapping), free that room.
            #
            # Using a while loop (not if) correctly handles cases where
            # multiple meetings end before the current start, e.g.
            # several zero-length or back-to-back-but-non-touching
            # meetings clearing out at once.
            while end_ptr < n and ends[end_ptr] < current_start:
                rooms_in_use -= 1
                end_ptr += 1
            
            # Allocate a room for the current meeting.
            rooms_in_use += 1
            
            # Track the peak concurrent room usage — this is our answer.
            if rooms_in_use > max_rooms:
                max_rooms = rooms_in_use
        
        return max_rooms