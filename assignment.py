import pandas as pd
import numpy as np


# # Task 1: NumPy Basics

# ### Array Creation
# #### 1. Create an array of 10 zeros but set the fifth element to 1.
# 

# In[14]:


# solution
arr = np.zeros(10, dtype=int)
arr[4] = 1
arr


# ### 2. Create a 4x4 matrix with values ranging from 0 to 15.

# In[ ]:


# solution

arr1 = np.arange(16).reshape(4, 4)
arr1


# ### Array Operations
# 
# #### 3. Create two NumPy arrays: Perform element-wise addition, subtraction, multiplication, and division.
# #### arr1: array([[41, 30, 10], [ 6, 35, 23], [23,  9, 10]])
# #### arr2: array([[4, 29,  2], [5, 13,  3], [24, 16,  7]])

# In[ ]:


# solution
# addition: arr1 + arr2

# subtraction: arr1 - arr2 

# division: arr1 / arr2

# multiplication: arr1 * arr2
Arr1 = np.array([[41, 30, 10], [6, 35, 23], [23, 9, 10]])
Arr2 = np.array([[4, 29, 2], [5, 13, 3], [24, 16, 7]])

addition = Arr1 + Arr2
subtraction = Arr1 - Arr2
multiplication = Arr1 * Arr2
division = Arr1 / Arr2

print(" Addition:\n", addition)
print("\n Subtraction:\n", subtraction)
print("\n Multiplication:\n", multiplication)
print("\n Division:\n", division)


# ### Array Slicing
# #### 4. Given the array: 
# #### arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# #### Extract elements from index 2 to 6.

# In[ ]:


# solution
Arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
sliced_array = Arr[2:6]
print("Sliced Array:", sliced_array)


# ### Using NumPy Methods
# #### 5. Convert the array below to a 3×3 NumPy array.
# #### array([[15, 18, 20], [15, 30, 8], [25, 30, 60]])
# #### Find the mean, sum, and maximum value.

# In[ ]:


# solution

# NB: round the mean to two decimal placesArr = np.array([[15, 18, 20], 
Arr = np.array([[15, 18, 20], [15, 30, 8], [25, 30, 60]])

mean_value = (np.mean(Arr), 2)  # Calculate mean
sum_value = np.sum(Arr)    # Calculate sum
max_value = np.max(Arr)

print("3x3 NumPy Array:\n", Arr)
print("\nMean Value:", mean_value)
print("Sum of Elements:", sum_value)
print("Maximum Value:", max_value)


# # Task 2: Pandas Basics

# ### Creating a Pandas Series
# #### 1. Create a Pandas Series from the list [10, 20, 30, 40, 50] with index labels 'a', 'b', 'c', 'd', 'e'.

# In[ ]:


# solution

data = [10,20,30,40,50]
index_labels = ['a', 'b', 'c', 'd', 'e']
series = pd.Series(data, index = index_labels)
print(series)


# ### Reading a CSV File
# #### 2. Read a CSV file named "data.csv" into a Pandas DataFrame. The CSV file has the following columns: "Name", "Age", "Salary".

# In[ ]:


# solution
<<<<<<< HEAD
df= pd.read_csv('data/data.csv')
=======
df = pd.read_csv('D:/python-numpy-project-nishantjoshi/python-numpy-pandas-group-a-nishantjoshi8/data/data.csv')

>>>>>>> 2d9357fbf46499909ac1cb507287ed68f906d7bc
df


# ### Slicing a DataFrame
# #### 3. From the DataFrame created in Exercise 2, select only the first three rows.

# In[ ]:


df.head(3)


# ### Manipulating a DataFrame - Adding a Column
# #### 4. Add a new column named "Tax" to the DataFrame from Exercise 2. The tax should be calculated as 10% of the "Salary" column.

# In[ ]:


# solution
# Calculate tax as 10% of Salary
df["Tax"] = df["Salary"] * 0.10

# Display updated DataFrame
df


# #### Filter the dataframe by the column "age" whereby the age is above 36 years

# In[ ]:


# solution
# Filter rows where Age is greater than 36
filtered_df = df[df["Age"] > 36]

# Display filtered DataFrame
filtered_df


# # Task 3: Data Analysis with Pandas

# ### Aggregation - Compute Total and Mean Sales
# #### 1. Given a dataset of sales transactions, compute the total and mean sales per region.

# In[ ]:


# Sample Data
sales_data = pd.DataFrame({
    "Region": ["North", "South", "North", "West", "South", "West"],
    "Sales": [1000, 1500, 1200, 1800, 1300, 1700]
})
sales_data.head()


# In[ ]:


# solution
# Compute total and mean sales per region
sales_summary = sales_data.groupby("Region")["Sales"].agg(["sum", "mean"])

# Display results
print(sales_summary)


# ### Merging - Combine Customer and Order Data
# #### 2. Merge customers and orders data based on CustomerID.

# In[ ]:


customers = pd.DataFrame({
    "CustomerID": [1, 2, 3, 4],
    "CustomerName": ["Alice", "Bob", "Charlie", "David"]
})

orders = pd.DataFrame({
    "OrderID": [101, 102, 103, 104],
    "CustomerID": [1, 2, 2, 4],
    "Amount": [250, 400, 600, 150]
})

print(customers.head(2), orders.head(2))


# In[ ]:


# solution
merged_data = customers.merge(orders, on="CustomerID", how="inner")
print(merged_data)


# ### Joining - Left Join Employee Data
# #### 4. Perform a left join between employees and departments on DeptID.

# In[ ]:


# NB: No test for this. Just an exercise for you to play around with
employees = pd.DataFrame({
    "EmpID": [101, 102, 103, 104],
    "EmpName": ["John", "Jane", "Alice", "Bob"],
    "DeptID": [1, 2, 2, 3]
})

departments = pd.DataFrame({
    "DeptID": [1, 2],
    "DeptName": ["HR", "Finance"]
})

print(employees.head(2), departments.head(2))


# In[ ]:


# solution
employees_departments = employees.merge(departments, on="DeptID", how="left")
print(employees_departments)


# ### Pivot Table - Summarize Sales Data
# #### 4. Create a pivot table showing total sales for each Region and Product.

# In[ ]:


sales_pivot_data = pd.DataFrame({
    "Region": ["North", "North", "South", "South", "West", "West"],
    "Product": ["A", "B", "A", "B", "A", "B"],
    "Sales": [200, 300, 150, 400, 250, 500]
})

sales_pivot_data.head(2)


# In[ ]:


# solution
pivot_table = sales_pivot_data.pivot_table(values="Sales", index="Region", columns="Product", aggfunc="sum")
print(pivot_table)