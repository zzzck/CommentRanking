from typing import List


class Solution:
    def findSubsequences(self, nums: List[int]) -> List[List[int]]:
        path = []
        result = []
        self.backtracking(nums, 0, path, result)
        return result

    def backtracking(self, nums, index, path, result):
        print(index)
        print(path)
        if len(path) >= 2:
            result.append(path[:])

        cur = 200
        for i in range(index, len(nums), 1):
            # print(i)
            # print(path)
            # print('------------')
            if nums[i] == cur:
                continue
            if len(path) == 0:
                path.append(nums[i])
                self.backtracking(nums, i + 1, path, result)
                cur = path.pop()
            elif nums[i] >= path[-1]:
                path.append(nums[i])
                self.backtracking(nums, i + 1, path, result)
                cur = path.pop()


if __name__ == '__main__':
    s = Solution()
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 1, 1, 1, 1]
    res = s.findSubsequences(nums)
    print(res)
