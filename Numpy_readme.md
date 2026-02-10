# NumPy for Data Science: Comprehensive Guide

This guide covers the essentials of **NumPy** (Numerical Python), the fundamental package for scientific computing in Python. It is the backbone of almost all Data Science libraries, including Pandas, Scikit-Learn, and TensorFlow.

---

## 🟢 Part 1: The Foundations

### 1. Why NumPy?

While Python lists are flexible, they are slow for large-scale math. NumPy arrays are:

* **Faster:** 50x to 100x faster than Python lists due to "Vectorization."
* **Memory Efficient:** They use significantly less RAM.
* **Optimized:** Written in C, allowing for high-performance numerical operations.

### 2. Installation & Setup

```bash
pip install numpy

```

Standard import convention:

```python
import numpy as np

```

### 3. Creating Arrays

| Method | Description | Example |
| --- | --- | --- |
| `np.array()` | Convert list to array | `np.array([1, 2, 3])` |
| `np.zeros()` | Array of zeros | `np.zeros((2, 3))` |
| `np.ones()` | Array of ones | `np.ones((3, 3))` |
| `np.arange()` | Sequence of numbers | `np.arange(0, 10, 2)` |
| `np.eye()` | Identity Matrix | `np.eye(3)` |

---

## 🟡 Part 2: Array Properties & Math

### 1. Inspecting Your Data

Before processing, check the "metadata" of your array:

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.shape)  # (2, 3) -> 2 rows, 3 columns
print(arr.ndim)   # 2 -> 2-dimensional
print(arr.dtype)  # int64

```

### 2. Vectorized Operations (No Loops!)

In NumPy, operations are applied **element-wise** automatically.

```python
arr = np.array([1, 2, 3])

print(arr + 10) # [11, 12, 13]
print(arr * 2)  # [2, 4, 6]
print(arr ** 2) # [1, 4, 9]

```

### 3. Aggregation Functions

```python
data = np.array([10, 20, 30, 40])

print(np.sum(data))  # 100
print(np.mean(data)) # 25.0
print(np.std(data))  # Standard Deviation

```

---

## 🔵 Part 3: Slicing, Filtering & Manipulation

### 1. Slicing and Reversing

```python
arr = np.array([10, 20, 30, 40, 50])

print(arr[1:4])   # [20, 30, 40] (Stop is exclusive)
print(arr[::-1])  # [50, 40, 30, 20, 10] (Reverse)

```

### 2. Boolean Masking (The "Filter")

This is the most powerful way to extract data:

```python
scores = np.array([85, 42, 90, 55, 78])

# Get all scores above 75
passing = scores[scores > 75] # [85, 90, 78]

```

### 3. Reshaping and Joining

* **Reshape:** `arr.reshape(rows, cols)` — Change structure without changing data.
* **Stacking:** * `np.vstack()`: Stack arrays vertically (add rows).
* `np.hstack()`: Stack arrays horizontally (add columns).



---

## 🟠 Part 4: Handling Missing Data & Special Values

In Data Science, "dirty" data is common. NumPy provides specific tools for this:

* **Detection:** You cannot use `==` to find missing values. Use `np.isnan(arr)`.
* **Cleaning:**
```python
# Replace all NaNs with 0
clean_arr = np.nan_to_num(arr, nan=0.0)

# Detect infinity (result of dividing by zero)
inf_mask = np.isinf(arr)

```



---

## 🛠 Real-World Workflow: Data Cleaning

When moving from NumPy to a real project, follow this sanity-check list:

1. **Inspect:** Use `.shape` and `np.isnan()` to find gaps.
2. **Impute:** Fill missing numerical values with the `np.mean()` or `np.median()`.
3. **Outliers:** Use Standard Deviation ( rule) to remove unrealistic data points (e.g., a "negative" salary or a 200-year-old employee).
4. **Vectorize:** Always prefer NumPy functions over Python `for` loops to keep your code fast.

---
