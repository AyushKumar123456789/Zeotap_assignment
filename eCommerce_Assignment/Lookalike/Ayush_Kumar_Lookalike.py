
"""
Task 2: Lookalike Model

This script:
1. Reads Customer + Transaction + Product data
2. Engineers features for each customer
3. Computes similarity/distance metric
4. Outputs the top 3 lookalike customers for the first 20 customers
   in a CSV file: Ayush_Kumar_Lookalike.csv
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

# ================ STEP 1: READ & MERGE DATA ================
customers_df = pd.read_csv("../data/Customers.csv")
products_df = pd.read_csv("../data/Products.csv")
transactions_df = pd.read_csv("../data/Transactions.csv")

merged_df = transactions_df.merge(customers_df, on='CustomerID', how='left')
merged_df = merged_df.merge(products_df, on='ProductID', how='left')

# ================ STEP 2: FEATURE ENGINEERING ================
# Features:
# 1. TotalSpent
# 2. NumTransactions
# 3. AvgOrderValue
# 4. CategorySpend
# 5. DaysSinceSignup

merged_df['TransactionDate'] = pd.to_datetime(merged_df['TransactionDate'])
customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
current_date = merged_df['TransactionDate'].max()

# TotalSpent and NumTransactions
customer_spending = merged_df.groupby('CustomerID')['TotalValue'].agg(['sum', 'count']).reset_index()
customer_spending.columns = ['CustomerID', 'TotalSpent', 'NumTransactions']
customer_spending['AvgOrderValue'] = customer_spending['TotalSpent'] / customer_spending['NumTransactions']

# CategorySpend
category_spend = merged_df.groupby(['CustomerID','Category'])['TotalValue'].sum().unstack(fill_value=0)
category_spend.reset_index(inplace=True)

# DaysSinceSignup
customers_df['DaysSinceSignup'] = (current_date - customers_df['SignupDate']).dt.days

# Merge features
feature_df = pd.merge(customers_df[['CustomerID','Region','DaysSinceSignup']], 
                      customer_spending[['CustomerID','TotalSpent','NumTransactions','AvgOrderValue']], 
                      on='CustomerID', how='left')

feature_df = pd.merge(feature_df, category_spend, on='CustomerID', how='left')
feature_df.fillna(0, inplace=True)

# Convert categorical 'Region' into dummy variables
feature_df = pd.get_dummies(feature_df, columns=['Region'], prefix='Region')

# ================ STEP 3: NORMALIZE FEATURES & COMPUTE SIMILARITY ================
customer_ids = feature_df['CustomerID']
features_for_similarity = feature_df.drop('CustomerID', axis=1)

scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_for_similarity)

# Compute Cosine Similarity
similarity_matrix = cosine_similarity(features_scaled)

# ================ STEP 4: FIND TOP 3 LOOKALIKES FOR FIRST 20 CUSTOMERS ================
top_3_lookalikes_map = {}

# Map each CustomerID to its row index
id_to_index = {cid: idx for idx, cid in enumerate(customer_ids)}

for cust_id in customer_ids[:20]:  # first 20 customers
    idx = id_to_index[cust_id]
    # Similarities for this customer to all others
    sim_scores = similarity_matrix[idx]

    # Sort by similarity descending, skip self (where sim=1)
    sim_scores_argsort = np.argsort(sim_scores)[::-1]
    top_matches = []

    for candidate_idx in sim_scores_argsort:
        if candidate_idx == idx:
            continue  # skip self
        candidate_cust_id = customer_ids[candidate_idx]
        candidate_score = sim_scores[candidate_idx]
        top_matches.append((candidate_cust_id, round(candidate_score, 4)))
        if len(top_matches) == 3:
            break

    top_3_lookalikes_map[cust_id] = top_matches

# ================ STEP 5: OUTPUT LOOKALIKE CSV ================
# Format: CustomerID, Lookalikes
# Where Lookalikes is a list of tuples

output_rows = []
for cust_id, lookalike_list in top_3_lookalikes_map.items():
    output_rows.append({
        "CustomerID": cust_id,
        "Lookalikes": lookalike_list
    })

lookalike_df = pd.DataFrame(output_rows)
lookalike_df.to_csv("Ayush_Kumar_Lookalike.csv", index=False)

print("Lookalike CSV created successfully!")
