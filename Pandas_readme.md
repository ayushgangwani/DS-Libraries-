# Pandas for Data Analysis: Comprehensive Guide

Welcome to the comprehensive documentation for the **Pandas for Data Analysis** series. This guide covers everything from foundational concepts to advanced data manipulation techniques.

---

## 🟢 Part 1: The Basics

### 1. Introduction to Pandas

**Pandas** is the industry-standard Python library for **Data Manipulation** and **Data Analysis**.

* **History:** Developed by Wes McKinney in 2008 to address financial data challenges.
* **Key Advantage:** Built on **NumPy**, making it fast and compatible with visualization tools like Matplotlib.

### 2. Core Data Structures

1. **Series:** A 1D array (like a single column).
2. **DataFrame:** A 2D table (the entire spreadsheet).

### 3. Input and Output (I/O)

| Format | Read Command | Save Command |
| --- | --- | --- |
| **CSV** | `pd.read_csv('file.csv')` | `df.to_csv('out.csv', index=False)` |
| **Excel** | `pd.read_excel('file.xlsx')` | `df.to_excel('out.xlsx')` |

### 4. Selection & Filtering (Examples)

Using a sample DataFrame `df`:

| Name | Age | Department | Salary |
| --- | --- | --- | --- |
| Alice | 25 | IT | 50000 |
| Bob | 35 | HR | 60000 |
| Charlie | 40 | IT | 80000 |

* **Selecting Columns:**
```python
# Selecting one column (Series)
names = df['Name']

# Selecting multiple columns (DataFrame)
subset = df[['Name', 'Salary']]

```


* **Filtering with Logic:**
```python
# Employees in IT earning more than 60k
it_pros = df[(df['Department'] == 'IT') & (df['Salary'] > 60000)]

```



---

## 🔵 Part 2: Advanced Techniques

### 1. Modifying DataFrames

* **Adding a Column via Calculation:**
```python
# Adding a 10% bonus column
df['Bonus'] = df['Salary'] * 0.10

```


* **Updating a Specific Value:**
```python
# Change Alice's salary (Alice is at index 0)
df.loc[0, 'Salary'] = 55000

```



### 2. Handling Missing Data (NaN)

Imagine a dataset with missing ages:

```python
# Check how many values are missing in each column
print(df.isnull().sum())

# Fill missing Ages with the average age of the group
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Or, drop rows where 'Name' is missing
df.dropna(subset=['Name'], inplace=True)

```

### 3. Grouping and Aggregation

To find the total salary spend per department:

```python
# Group by 'Department' and sum the 'Salary'
dept_spend = df.groupby('Department')['Salary'].sum()

# Result:
# IT    130000
# HR     60000

```

### 4. Merging and Concatenating

**Merging (Joins):** Combining employee data with department locations.

```python
# df1 has Names and Dept; df2 has Dept and Location
merged_df = pd.merge(df1, df2, on='Department', how='inner')

```

**Concatenating (Stacking):** Combining 2023 data with 2024 data.

```python
# Stacking rows vertically
all_years = pd.concat([df_2023, df_2024], axis=0, ignore_index=True)

```

---

### Real-Life Applications

* **Finance:** Tracking stock volatility.
* **Retail:** Analyzing customer churn and top-performing products.
* **Healthcare:** Cleaning patient records for research.

---
