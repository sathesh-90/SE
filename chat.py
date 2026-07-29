import heapq

def minimumMeetingGroups(intervals):
    # If there are no meetings, no rooms are required.
    if not intervals:
        return 0

    # Sort meetings by their start time.
    intervals.sort(key=lambda x: x[0])

    # Min-heap stores the end times of meetings currently occupying rooms.
    end_times = []

    for start, end in intervals:
        # A room can be reused only if the previous meeting ends
        # strictly before the current meeting starts.
        # Since overlap is defined as start <= end,
        # start == end is still considered overlapping.
        if end_times and start > end_times[0]:
            heapq.heappop(end_times)

        # Allocate the current meeting to a room
        # (either a reused room or a new one).
        heapq.heappush(end_times, end)

    # The heap size represents the maximum number of rooms
    # needed simultaneously.
    return len(end_times)