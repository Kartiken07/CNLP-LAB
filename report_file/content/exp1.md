# Experiment 1: Employee Data Analysis with Pandas and Matplotlib

## Abstract
This experiment analyses a 100-row employee information dataset using pandas and Matplotlib through ten tabulation and visualisation tasks. The file contains seven non-null columns: Employee_ID, Name, Department, Age, Gender, Salary and Experience_Years. Departmental aggregation shows Operations has the highest average salary (90,875.00) and HR the lowest (72,095.24), while HR is the largest department with 21 employees and Finance the smallest with 12. The workforce is 64% male and 36% female, the overall mean salary is 77,760, and 51 employees earn more than that mean. The dataset maximum is 119,000, reached twice in Marketing, and Sales has the highest average experience at 23.846154 years. Salary distribution, age distribution, a salary-versus-experience scatter plot and ranked bar charts were produced, with the scatter plot showing little visible relationship between experience and salary (correlation about 0.033).

## Description
The input file `employee_information_100.csv` contains 100 records indexed 0 to 99, and `df.info()` reports 100 non-null values in each of its seven columns, three `int64` columns and four string columns, with a memory usage of 5.6 KB. The notebook imports `pandas` and `matplotlib.pyplot`, reads the CSV with `pd.read_csv()`, and inspects the first five rows with `df.head()`, which begin with E001 Diana1 (HR, 39, Male, 58,000, 8 years) and end with E005 Nisha5 (IT, 50, Female, 30,000, 10 years). All seven string columns are read as plain strings in the executed notebook, and no rows are dropped or imputed.

The first five tasks build the core aggregations and charts. `groupby('Department')['Salary'].mean().sort_values(ascending=False)` gives Operations 90,875.00, Finance 82,416.67, Marketing 76,100.00, Sales 74,769.23, IT 73,611.11 and HR 72,095.24, drawn as a gold bar chart. `value_counts()` on Department returns HR 21, Marketing 20, IT 18, Operations 16, Sales 13 and Finance 12, drawn as a light-blue bar chart. Grouping Gender returns Female 36 and Male 64, displayed as a pie chart with `autopct='%1.1f%%'`, giving 64.0% male and 36.0% female. A 10-bin histogram describes salary (minimum 30,000, median 78,000, maximum 119,000, standard deviation about 25,769.81), and a purple alpha-0.5 scatter plot places salary on the x-axis and Experience_Years on the y-axis.

The remaining tasks rank, filter and compare the same columns. Sorting Salary descending and taking the first ten values gives 119,000 twice, 117,000 three times, then 116,000 and three values of 115,000, shown as an orange bar chart. `groupby('Department')['Salary'].max()` yields Finance 117,000, HR 113,000, IT 117,000, Marketing 119,000, Operations 117,000 and Sales 114,000, drawn as another light-blue bar chart. Comparing each salary with `df['Salary'].mean()` = 77,760 selects 51 employees; the first five shown are E002 Diana2 (105,000), E004 Neha4 (113,000), E008 Neha8 (98,000), E009 Mohan9 (110,000) and E010 Vikram10 (114,000). Average experience by department ranges from Finance 11.166667 to Sales 23.846154, and the final 10-bin histogram summarises ages from 22 to 59 with a mean of 41.93. Eight figures were saved from the notebook's plotting outputs.

## Code

### Loading the dataset and comparing departments, gender and headcount (Tasks 1.1-1.3)
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/kartiken/Downloads/CLNLP-LAB/EXP-1/employee_information_100.csv")
df.info()

dept_wise_avg_salary = df.groupby('Department')['Salary'].mean().sort_values(ascending=False)
dept_wise_avg_salary.plot(kind='bar', color='gold', edgecolor='black', figsize=(10, 6))
plt.title('Average Salary by Department')
plt.xlabel('Department')
plt.ylabel('Average Salary')
plt.show()

df.head()

count = df['Department'].value_counts()
count.plot(kind='bar', color='lightblue', edgecolor='black', figsize=(10, 6))
plt.title('Number of Employees by Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.show()

gender = df.groupby('Gender')["Gender"].count()
gender
gender.plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'], startangle=90, figsize=(8, 8))
```

### Distribution, ranking, filtering and age charts (Tasks 1.4-1.10)
```python
df["Salary"].plot(kind='hist', bins=10, color='lightgreen', edgecolor='black', figsize=(10, 6))

plt.scatter(df['Salary'], df['Experience_Years'], color='purple', alpha=0.5)
plt.title('Salary vs Experience Years')

df["Salary"].sort_values(ascending=False).head(10).plot(kind='bar', color='orange', edgecolor='black', figsize=(10, 6))

highest_salary_in_every_department = df.groupby("Department")["Salary"].max()
highest_salary_in_every_department.plot(kind='bar', color='lightblue', edgecolor='black', figsize=(10, 6))
plt.title('Highest Salary in Every Department')
plt.xlabel('Department')
plt.ylabel('Highest Salary')
plt.show()

meane = df["Salary"].mean()
df[df["Salary"] > meane].head()

df.groupby("Department")["Experience_Years"].mean()

ages = df["Age"]
ages.plot(kind='hist', bins=10, color='lightyellow', edgecolor='black', figsize=(10, 6))
```

## Tables

### Average salary by department (descending order)
| Department | Average Salary |
| --- | --- |
| Operations | 90875.00 |
| Finance | 82416.67 |
| Marketing | 76100.00 |
| Sales | 74769.23 |
| IT | 73611.11 |
| HR | 72095.24 |

### Employees per department and gender split
| Department | Employees |
| --- | --- |
| HR | 21 |
| Marketing | 20 |
| IT | 18 |
| Operations | 16 |
| Sales | 13 |
| Finance | 12 |

| Gender | Count | Percentage |
| --- | --- | --- |
| Male | 64 | 64.0% |
| Female | 36 | 36.0% |

### Highest salary and average experience by department
| Department | Highest Salary | Average Experience Years |
| --- | --- | --- |
| Finance | 117000 | 11.166667 |
| HR | 113000 | 19.809524 |
| IT | 117000 | 18.388889 |
| Marketing | 119000 | 21.900000 |
| Operations | 117000 | 16.875000 |
| Sales | 114000 | 23.846154 |

### Ten highest salaries in the dataset
| Rank | Employee_ID | Name | Department | Salary |
| --- | --- | --- | --- | --- |
| 1 | E086 | Omar86 | Marketing | 119000 |
| 2 | E088 | Diana88 | Marketing | 119000 |
| 3 | E023 | Rahul23 | Finance | 117000 |
| 4 | E059 | Neha59 | IT | 117000 |
| 5 | E015 | Ishan15 | Operations | 117000 |
| 6 | E049 | Charlie49 | IT | 116000 |
| 7 | E056 | Sneha56 | Operations | 115000 |
| 8 | E065 | Bob65 | Finance | 115000 |
| 9 | E032 | George32 | Operations | 115000 |
| 10 | E010 | Vikram10 | IT | 114000 |

### Summary statistics and first five above-average earners
| Metric | Value |
| --- | --- |
| Count | 100 |
| Mean salary | 77760.0 |
| Standard deviation | 25769.812423 |
| Minimum salary | 30000 |
| Median salary | 78000 |
| Maximum salary | 119000 |
| Employees above mean salary | 51 |
| Mean age | 41.93 |
| Mean experience | 18.99 |
| Salary-experience correlation | 0.0332739595 |

| Employee_ID | Name | Department | Age | Gender | Salary | Experience_Years |
| --- | --- | --- | --- | --- | --- | --- |
| E002 | Diana2 | Operations | 56 | Male | 105000 | 27 |
| E004 | Neha4 | HR | 57 | Male | 113000 | 34 |
| E008 | Neha8 | Finance | 24 | Female | 98000 | 7 |
| E009 | Mohan9 | HR | 57 | Female | 110000 | 23 |
| E010 | Vikram10 | IT | 26 | Male | 114000 | 14 |

## Figures

| path | caption |
| --- | --- |
| /Users/kartiken/Downloads/CLNLP-LAB/report_file/assets/exp1_01.png | Gold bar chart of average salary by department, sorted descending (Task 1.1) |
| /Users/kartiken/Downloads/CLNLP-LAB/report_file/assets/exp1_02.png | Light-blue bar chart of the number of employees in each department (Task 1.2) |
| /Users/kartiken/Downloads/CLNLP-LAB/report_file/assets/exp1_03.png | Pie chart showing the percentage of male and female employees (Task 1.3) |
| /Users/kartiken/Downloads/CLNLP-LAB/report_file/assets/exp1_04.png | Light-green 10-bin histogram of salary distribution (Task 1.4) |
| /Users/kartiken/Downloads/CLNLP-LAB/report_file/assets/exp1_05.png | Purple scatter plot of salary versus experience years (Task 1.5) |
| /Users/kartiken/Downloads/CLNLP-LAB/report_file/assets/exp1_06.png | Orange bar chart of the ten highest salaries (Task 1.6) |
| /Users/kartiken/Downloads/CLNLP-LAB/report_file/assets/exp1_07.png | Light-blue bar chart of the highest salary in every department (Task 1.7) |
| /Users/kartiken/Downloads/CLNLP-LAB/report_file/assets/exp1_08.png | Light-yellow 10-bin histogram of employee age distribution (Task 1.10) |
