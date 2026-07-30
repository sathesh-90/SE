import heapq

"""
Algorithm:
1. Sort meetings according to start time.
2. Maintain a min heap containing ending times.
3. If current meeting starts strictly after the earliest ending meeting,
   reuse that room.
4. Otherwise allocate a new room.
5. Maximum heap size is the answer.

Time Complexity:
Sorting : O(N log N)
Heap Operations : O(N log N)
Overall : O(N log N)

Space Complexity:
O(N)
"""


def minimum_rooms(intervals):

    # Sort meetings based on start time
    intervals.sort(key=lambda x: x[0])

    # Min heap stores ending times
    min_heap = []

    answer = 0

    # Process meetings
    for start, end in intervals:

        # Room can be reused only when start > earliest end
        if min_heap and start > min_heap[0]:
            heapq.heappop(min_heap)

        # Allocate room
        heapq.heappush(min_heap, end)

        # Update maximum rooms used
        answer = max(answer, len(min_heap))

    return answer


def main():
    n = int(input())

    intervals = []

    for _ in range(n):
        start, end = map(int, input().split())
        intervals.append([start, end])

    print(minimum_rooms(intervals))


if __name__ == "__main__":
    main()