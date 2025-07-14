
Press CTRL F and find the question you are looking for with it's number.

# ✅ Leetcode 560 – Subarray Sum Equals K

---

## 🔍 Problem Statement

You are given an integer array `nums` and an integer `k`.

> Return the **total number of contiguous subarrays** whose **sum equals `k`**.

---

### 🧾 Example 1

`Input: nums = [1, 1, 1], k = 2   Output: 2`  

Explanation:

- Subarrays `[1,1]` (first two) and `[1,1]` (last two) both sum to 2.
    

---

### 🔒 Constraints:

- `1 <= nums.length <= 2 * 10⁴`
    
- `-1000 <= nums[i] <= 1000`
    
- Time limit demands: **O(n)** solution.
    

---

## 🪓 Brute Force Approach

### 💡 Idea:

Try every possible subarray, compute the sum, and check if it equals `k`.

code:

count = 0
for start in range(n):
    sum_ = 0
    for end in range(start, n):
        sum_ += nums[end]
        if sum_ == k:
            count += 1


### ⏱️ Time Complexity:

- Outer loop: O(n)
    
- Inner loop: O(n)
    
- Total: **O(n²)**
    

### ❌ Why It's Bad:

- Inefficient for large arrays (n up to 20,000)
    
- Recomputes sums repeatedly
    
- Fails on time limits for large inputs
    

---

## 🚀 Optimal Approach – Prefix Sum with Hash Map

### 🧠 Key Insight:

If you know the **prefix sum** up to two indices `i` and `j` (`j > i`),  
then the **sum of the subarray between them** is:



> If at index `j`, `prefix_sum = S`, and you’ve **seen a prefix sum** of `S - k` before,  
> then there is at least one subarray ending at `j` whose sum is `k`.

---

### 🔁 Algorithm Steps:

1. Initialize `prefix_sum = 0`
    
2. Use a hashmap `prefix_counts` to track how many times each prefix sum has occurred
    
3. For every number in `nums`:
    
    - Add it to `prefix_sum`
        
    - If `(prefix_sum - k)` exists in the map, it means there’s a subarray that sums to `k`
        
    - Add `prefix_counts[prefix_sum - k]` to the answer
        
    - Update `prefix_counts[prefix_sum] += 1`
        

---

### ✅ Python Code:

from collections import defaultdict

class Solution(object):
    def subarraySum(self, nums, k):
        prefix_counts = defaultdict(int)
        prefix_counts[0] = 1  # Handle exact match with prefix_sum == k

        current_sum = 0
        count = 0

        for num in nums:
            current_sum += num
            count += prefix_counts[current_sum - k]
            prefix_counts[current_sum] += 1

        return count

---

### ⏱️ Time and Space Complexity

|Metric|Value|
|---|---|
|Time|✅ **O(n)**|
|Space|✅ **O(n)** (for prefix_counts hash map)|

---

## 💡 Why This Works (Logically):

Imagine a **running sum** as you walk left to right.

You’re asking:

> “Have I ever seen a total such that if I subtract it from what I have now, I get `k`?”

That’s exactly what this checks at every step:

python

CopyEdit

`if prefix_sum - k exists:     that means some subarray ending here has sum k`

### 🧠 Example:

python

CopyEdit

`nums = [1, 2, 1, 2], k = 3  prefix_sum running → [1, 3, 4, 6]`

At each index, we:

- Check if `prefix_sum - k` exists in the hash map
    
- If yes, we **count how many times** that prefix sum appeared (multiple subarrays possible)
    
- Update our prefix sum count
    

---

## 🧠 Intuition Summary

|Element|Meaning|
|---|---|
|`prefix_sum`|Sum of all elements from 0 to current index|
|`prefix_sum - k`|The target sum we need to have seen earlier|
|`prefix_counts`|Map of all past prefix sums and their frequencies|