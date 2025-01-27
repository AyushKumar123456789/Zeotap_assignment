

## 📄 README.md


# eCommerce Transactions Data Science Assignment

## 📌 Overview

This repository contains solutions to the Data Science Assignment focused on analyzing an eCommerce Transactions dataset. The project is divided into three main tasks:

1. **Exploratory Data Analysis (EDA) and Business Insights**
2. **Lookalike Model Development**
3. **Customer Segmentation using Clustering Techniques**

Each task is implemented using Python scripts and accompanied by detailed PDF reports. The deliverables are organized into separate folders for clarity and ease of access.

## 📁 Folder Structure

```
eCommerce_Assignment/
│
├── data/
│   ├── Customers.csv
│   ├── Products.csv
│   └── Transactions.csv
│
├── EDA/
│   ├── Ayush_Kumar_EDA.py
│   ├── Ayush_Kumar_EDA.pdf
│   ├── category_revenue_top5.png
│   ├── monthly_transactions.png
│   ├── price_distribution.png
│   └── correlation_heatmap.png
│
├── Lookalike/
│   ├── Ayush_Kumar_Lookalike.py
│   ├── Ayush_Kumar_Lookalike.csv
│   └── Ayush_Kumar_Lookalike_Explanation.pdf
│
├── Clustering/
│   ├── Ayush_Kumar_Clustering.py
│   ├── Ayush_Kumar_Clustering.pdf
│   └── ClusterPlot.png
│
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Ayush_Kumar/eCommerce_Assignment.git
cd eCommerce_Assignment
```

### 2. Set Up the Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Ensure you have `pip` installed. Then, install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 4. Data Preparation

Place the provided CSV files (`Customers.csv`, `Products.csv`, `Transactions.csv`) into the `data/` directory.

## 📊 Task 1: Exploratory Data Analysis (EDA) and Business Insights

### 🔍 Description

Perform EDA on the dataset to uncover patterns, trends, and insights that can drive business decisions.

### 🛠️ Files

- **Script:** `EDA/Ayush_Kumar_EDA.py`
- **Report:** `EDA/Ayush_Kumar_EDA.pdf`

### 📄 How to Run

Navigate to the EDA directory and execute the script:

```bash
cd EDA
python Ayush_Kumar_EDA.py
```

This will generate visualizations and a PDF report containing five key business insights.

## 🎯 Task 2: Lookalike Model

### 🔍 Description

Develop a Lookalike Model that recommends three similar customers based on a user's profile and transaction history. The model leverages both customer and product information to assign similarity scores.

### 🛠️ Files

- **Script:** `Lookalike/Ayush_Kumar_Lookalike.py`
- **Output CSV:** `Lookalike/Ayush_Kumar_Lookalike.csv`
- **Explanation Report:** `Lookalike/Ayush_Kumar_Lookalike_Explanation.pdf`

### 📄 How to Run

Navigate to the Lookalike directory and execute the script:

```bash
cd ../Lookalike
python Ayush_Kumar_Lookalike.py
```

This script performs the following:

1. Generates `Ayush_Kumar_Lookalike.csv` containing the top 3 lookalikes for the first 20 customers (`C0001` - `C0020`).
2. Produces a PDF report explaining the model development process.

## 🧩 Task 3: Customer Segmentation / Clustering

### 🔍 Description

Segment customers into distinct groups using clustering techniques based on their profiles and transaction histories. The goal is to identify actionable segments that can inform targeted marketing strategies.

### 🛠️ Files

- **Script:** `Clustering/Ayush_Kumar_Clustering.py`
- **Report:** `Clustering/Ayush_Kumar_Clustering.pdf`
- **Visualization:** `Clustering/ClusterPlot.png`

### 📄 How to Run

Navigate to the Clustering directory and execute the script:

```bash
cd ../Clustering
python Ayush_Kumar_Clustering.py
```

This script:

1. Determines the optimal number of clusters (K) between 2 and 10 using the Davies-Bouldin Index.
2. Applies K-Means clustering to segment the customers.
3. Generates a PDF report detailing:
   - Number of clusters formed.
   - DB Index value.
   - Cluster sizes.
   - PCA-based cluster visualization.

## 📝 Dependencies

All required Python libraries are listed in the `requirements.txt` file. Key dependencies include:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `fpdf`

Install them using:

```bash
pip install -r requirements.txt
```

## 📚 Additional Information

### 💡 Insights and Recommendations

- **EDA:** Provides foundational understanding of sales patterns, customer distribution, and product performance.
- **Lookalike Model:** Enables targeted marketing by identifying customers similar to high-value users.
- **Clustering:** Facilitates personalized marketing strategies by grouping customers with similar behaviors and preferences.

### 🔧 Code Enhancements

To ensure the scripts are robust and maintainable:

- **Modular Design:** Functions are used to encapsulate distinct tasks.
- **Error Handling:** Try-except blocks manage potential errors gracefully.
- **Documentation:** Comprehensive docstrings and inline comments explain the purpose and functionality of code segments.
- **Visualization Clarity:** All plots are labeled clearly with titles, axis labels, and legends for easy interpretation.

## 📫 Contact

For any queries or feedback, feel free to reach out at [john.doe@example.com](mailto:john.doe@example.com).

---

**Disclaimer:** This project is developed for academic purposes as part of a Data Science assignment. All datasets used are fictional and any resemblance to real entities is purely coincidental.




## 📝 Suggestions to Enhance Your Python Scripts

To make your Python scripts appear more polished and human-written, consider the following enhancements:

### 1. **Add Comprehensive Comments and Docstrings**

Provide clear explanations for each function and major code blocks. This aids readability and demonstrates a clear understanding of the processes involved.

**Example:**

```python
def load_and_merge_data(customers_path, products_path, transactions_path):
    """
    Loads and merges the Customers, Products, and Transactions datasets.

    Args:
        customers_path (str): Path to Customers.csv
        products_path (str): Path to Products.csv
        transactions_path (str): Path to Transactions.csv

    Returns:
        pd.DataFrame: Merged dataframe containing all necessary information.
        pd.DataFrame: Customers dataframe.
    """
    try:
        customers_df = pd.read_csv(customers_path)
        products_df = pd.read_csv(products_path)
        transactions_df = pd.read_csv(transactions_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Proceed with data cleaning and merging
    ...
```

### 2. **Use Meaningful Variable Names**

Ensure variable names are descriptive and convey their purpose. Avoid single-letter names unless used in a well-understood context (e.g., loop counters).

**Example:**

```python
# Less Descriptive
df1 = pd.read_csv(customers_path)

# More Descriptive
customers_df = pd.read_csv(customers_path)
```

### 3. **Implement Error Handling**

Use try-except blocks to handle potential errors, such as missing files or invalid inputs. This makes your scripts more robust and user-friendly.

**Example:**

```python
try:
    customers_df = pd.read_csv(customers_path)
except FileNotFoundError:
    print(f"Error: The file {customers_path} was not found.")
    sys.exit(1)
```

### 4. **Structure Code Logically with Functions**

Organize your code into functions that perform specific tasks. This enhances readability and reusability.

**Example:**

```python
def preprocess_data(merged_df):
    """
    Preprocesses the merged dataframe by handling missing values and encoding categorical variables.

    Args:
        merged_df (pd.DataFrame): The merged dataframe.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    # Handle missing values
    merged_df.fillna(0, inplace=True)

    # Encode categorical variables
    merged_df = pd.get_dummies(merged_df, columns=['Region'], prefix='Region')

    return merged_df
```

### 5. **Consistent Formatting and Style**

Adhere to PEP 8 guidelines for Python code to maintain consistency and readability. Tools like `flake8` or `black` can help enforce these standards.

**Example:**

```python
# PEP 8 Compliant Function Definition
def calculate_average_order_value(total_spent, num_transactions):
    """
    Calculates the average order value.

    Args:
        total_spent (float): Total amount spent by the customer.
        num_transactions (int): Number of transactions made by the customer.

    Returns:
        float: Average order value.
    """
    if num_transactions == 0:
        return 0
    return total_spent / num_transactions
```

### 6. **Use Logging Instead of Print Statements**

Implement logging to track the execution flow and debug issues more effectively. This is preferable over using multiple print statements.

**Example:**

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path):
    """
    Loads data from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    try:
        data = pd.read_csv(path)
        logging.info(f"Successfully loaded data from {path}")
        return data
    except FileNotFoundError:
        logging.error(f"The file {path} was not found.")
        sys.exit(1)
```

### 7. **Provide User-Friendly Outputs**

Ensure that outputs, especially in interactive scripts, are clear and informative. Use formatted strings for better readability.

**Example:**

```python
print(f"\nTop 3 Lookalikes for CustomerID {customer_id}:")
for idx, (look_id, score) in enumerate(lookalikes, 1):
    print(f"{idx}. CustomerID: {look_id}, Similarity Score: {score:.4f}")
```

### 8. **Include a Main Guard**

Use the `if __name__ == "__main__":` guard to ensure that certain parts of the script run only when the script is executed directly, not when imported as a module.

**Example:**

```python
def main():
    # Main execution flow
    ...

if __name__ == "__main__":
    main()
```

### 9. **Optimize Performance**

For large datasets, consider optimizing your code for performance. Utilize vectorized operations with pandas and numpy instead of iterative loops where possible.

**Example:**

```python
# Inefficient Loop
avg_order_values = []
for index, row in customer_spending.iterrows():
    avg_order = row['TotalSpent'] / row['NumTransactions'] if row['NumTransactions'] > 0 else 0
    avg_order_values.append(avg_order)

customer_spending['AvgOrderValue'] = avg_order_values

# Optimized Vectorized Operation
customer_spending['AvgOrderValue'] = customer_spending.apply(
    lambda row: row['TotalSpent'] / row['NumTransactions'] if row['NumTransactions'] > 0 else 0,
    axis=1
)
```

### 10. **Document Assumptions and Limitations**

Clearly state any assumptions made during the analysis and acknowledge potential limitations. This demonstrates critical thinking and transparency.

**Example:**

```python
# Assumption: Customers with zero transactions are considered inactive and have an AvgOrderValue of 0.
customer_spending['AvgOrderValue'] = customer_spending.apply(
    lambda row: row['TotalSpent'] / row['NumTransactions'] if row['NumTransactions'] > 0 else 0,
    axis=1
)
```

---

## ✅ Final Checklist for Full Score

To ensure you receive full marks based on the evaluation criteria, verify the following:

1. **EDA Task:**
   - Comprehensive exploratory analysis covering various aspects of the dataset.
   - At least five well-articulated business insights in the PDF report.
   - Clear and labeled visualizations included in the report.

2. **Lookalike Model Task:**
   - Accurate and logical model development leveraging both customer and product data.
   - `Lookalike.csv` correctly maps the first 20 customers to their top 3 lookalikes with similarity scores.
   - Interactive and user-friendly script with clear explanations.
   - High-quality recommendations with meaningful similarity scores.

3. **Customer Segmentation / Clustering Task:**
   - Appropriate choice of clustering algorithm with justification.
   - Optimal number of clusters determined using the Davies-Bouldin Index.
   - Calculation and presentation of clustering metrics.
   - Clear and insightful visualizations of clusters.
   - Comprehensive PDF report detailing clustering results and interpretations.

4. **General:**
   - Adherence to file naming conventions as specified.
   - Clean, efficient, and well-documented code.
   - Organized folder structure with all deliverables appropriately placed.
   - Successful execution of all scripts without errors.
   - Public GitHub repository with all files uploaded correctly.
   - Detailed and informative README.md guiding users through the project.
