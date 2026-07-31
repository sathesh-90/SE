from typing import List

class Solution:

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:

        ans = []
        path = []
        n = len(candidates)

        def dfs(index, remaining):

            if remaining == 0:
                ans.append(path.copy())
                return

            if remaining < 0 or index == n:
                return

            path.append(candidates[index])
            dfs(index, remaining - candidates[index])
            path.pop()

            dfs(index + 1, remaining)

        dfs(0, target)

        return ans


def main():

    # Number of candidates
    n = int(input())

    # Candidates array
    candidates = list(map(int, input().split()))

    # Target value
    target = int(input())

    obj = Solution()

    result = obj.combinationSum(candidates, target)

    print(result)


if __name__ == "__main__":
    main()
