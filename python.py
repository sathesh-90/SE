"""
Algorithm:
1. Start from index 0.
2. For every candidate:
      a) Take it.
      b) Skip it.
3. If target becomes 0 -> save answer.
4. If target < 0 or index reaches end -> stop recursion.

Time Complexity:
Worst Case: O(2^(target/minCandidate))

Space Complexity:
O(target/minCandidate)
"""


def combinationSum(candidates, target):

    answer = []

    # Backtracking function
    def backtrack(index, target, current):

        # Valid combination found
        if target == 0:
            answer.append(current[:])
            return

        # Invalid case
        if index == len(candidates) or target < 0:
            return

        # -------------------
        # Take current element
        # -------------------
        current.append(candidates[index])

        # Stay on same index because reuse is allowed
        backtrack(index, target - candidates[index], current)

        # Remove last element
        current.pop()

        # -------------------
        # Skip current element
        # -------------------
        backtrack(index + 1, target, current)

    backtrack(0, target, [])

    return answer


def main():

    n = int(input())

    candidates = list(map(int, input().split()))

    target = int(input())

    print(combinationSum(candidates, target))


if __name__ == "__main__":
    main()