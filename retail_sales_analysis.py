import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Read data
df = pd.read_csv("Retail and wherehouse Sale.csv")

# --- Chart 1: Total retail sales by product type ---
retail_sales_by_type = df.groupby("ITEM TYPE")["RETAIL SALES"].sum().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
retail_sales_by_type.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Total retail sales by product type", fontsize=14)
plt.xlabel("Product Type", fontsize=12)
plt.ylabel("Retail sales", fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Chart 2: Monthly retail sales trend ---
monthly_sales = df.groupby('MONTH')['RETAIL SALES'].sum().sort_index()
plt.figure(figsize=(10, 6))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o', color='green')
plt.title('Monthly Retail Sales Trend')
plt.xlabel('Month')
plt.ylabel('Retail Sales')
plt.xticks(monthly_sales.index)
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Chart 3: Compare Retail and Warehouse Channel Revenues by Month ---
monthly_sales_dual = df.groupby('MONTH')[['RETAIL SALES', 'WAREHOUSE SALES']].sum().sort_index()
months = monthly_sales_dual.index
bar_width = 0.4
x = range(len(months))
plt.figure(figsize=(10, 6))
plt.bar([i - bar_width/2 for i in x], monthly_sales_dual['RETAIL SALES'], width=bar_width, label='Retail Sales')
plt.bar([i + bar_width/2 for i in x], monthly_sales_dual['WAREHOUSE SALES'], width=bar_width, label='Warehouse Sales')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.title('Monthly Retail vs Warehouse Sales')
plt.xticks(x, months)
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# --- Chart 4: Revenue Share by Product Type ---
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['Total Revenue'] = df['RETAIL SALES'] + df['WAREHOUSE SALES']
df = df[df['Total Revenue'] > 0]
type_sales = df.groupby('ITEM TYPE')['Total Revenue'].sum()
type_sales = type_sales[type_sales > 0]
plt.figure(figsize=(10, 8))
patches, texts = plt.pie(
    type_sales,
    labels=None,
    autopct=None,
    startangle=140,
    colors=plt.cm.Paired.colors
)
plt.legend(
    patches,
    type_sales.index,
    title="Product Type",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    borderaxespad=0.
)
plt.title('Revenue Share by Product Type')
plt.axis('equal')
plt.tight_layout()
plt.show()

# --- Chart 5: Actual and Predicted Retail Sales ---
X = df[['WAREHOUSE SALES']]
y = df['RETAIL SALES']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test.values, label='Actual', marker='o')
plt.plot(range(len(y_pred)), y_pred, label='Predicted', marker='x')
plt.title('Actual vs Predicted Retail Sales')
plt.xlabel('Test Sample Index')
plt.ylabel('Retail Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
