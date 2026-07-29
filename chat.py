from bisect import bisect_left
import sys

def main():
    input = sys.stdin.readline

    # Read number of jobs
    n_line = input().strip()
    if not n_line:
        return
    n = int(n_line)

    jobs = []
    for _ in range(n):
        start, end, profit = map(int, input().split())
        jobs.append((start, end, profit))

    # Sort jobs by start time.
    # This allows binary searching for the next compatible job.
    jobs.sort(key=lambda x: x[0])

    starts = [job[0] for job in jobs]

    # dp[i] = maximum profit obtainable starting from job i.
    # dp[n] = 0 (base case: no jobs left).
    dp = [0] * (n + 1)

    # Process from right to left so future states are already computed.
    for i in range(n - 1, -1, -1):
        start, end, profit = jobs[i]

        # Find the first job whose start time is >= current job's end time.
        # Such a job does not overlap with the current one.
        next_index = bisect_left(starts, end)

        # Option 1: Skip current job.
        skip = dp[i + 1]

        # Option 2: Take current job and continue from the next compatible job.
        take = profit + dp[next_index]

        # Store the best choice.
        dp[i] = max(skip, take)

    print(dp[0])


if __name__ == "__main__":
    main()