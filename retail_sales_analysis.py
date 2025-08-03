import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("Retail and wherehouse Sale.csv")

st.title("📊 Retail & Warehouse Sales Dashboard")

# --- Chart 1: Total retail sales by product type ---
st.subheader("1. Total retail sales by product type")
retail_sales_by_type = df.groupby("ITEM TYPE")["RETAIL SALES"].sum().sort_values(ascending=False)
fig1, ax1 = plt.subplots(figsize=(10, 6))
retail_sales_by_type.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
ax1.set_title("Total retail sales by product type")
ax1.set_xlabel("Product Type")
ax1.set_ylabel("Retail Sales")
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig1)

# --- Chart 2: Monthly retail sales trend ---
st.subheader("2. Monthly retail sales trend")
monthly_sales = df.groupby('MONTH')['RETAIL SALES'].sum().sort_index()
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(monthly_sales.index, monthly_sales.values, marker='o', color='green')
ax2.set_title("Monthly Retail Sales Trend")
ax2.set_xlabel("Month")
ax2.set_ylabel("Retail Sales")
ax2.grid(True)
st.pyplot(fig2)

# --- Chart 3: Retail vs Warehouse Revenue by Month ---
st.subheader("3. Compare Retail and Warehouse Channel Revenues by Month")
monthly_sales_dual = df.groupby('MONTH')[['RETAIL SALES', 'WAREHOUSE SALES']].sum().sort_index()
months = monthly_sales_dual.index
bar_width = 0.4
x = range(len(months))
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.bar([i - bar_width/2 for i in x], monthly_sales_dual['RETAIL SALES'], width=bar_width, label='Retail Sales')
ax3.bar([i + bar_width/2 for i in x], monthly_sales_dual['WAREHOUSE SALES'], width=bar_width, label='Warehouse Sales')
ax3.set_xlabel("Month")
ax3.set_ylabel("Total Revenue")
ax3.set_title("Monthly Retail vs Warehouse Sales")
ax3.set_xticks(list(x))
ax3.set_xticklabels(months)
ax3.legend()
ax3.grid(True, axis='y')
st.pyplot(fig3)

# --- Chart 4: Revenue Share by Product Type ---
st.subheader("4. Revenue Share by Product Type")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['Total Revenue'] = df['RETAIL SALES'] + df['WAREHOUSE SALES']
df = df[df['Total Revenue'] > 0]
type_sales = df.groupby('ITEM TYPE')['Total Revenue'].sum()
fig4, ax4 = plt.subplots(figsize=(10, 8))
patches, texts, autotexts = ax4.pie(
    type_sales,
    labels=type_sales.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Paired.colors
)
ax4.axis('equal')
ax4.set_title("Revenue Share by Product Type")
st.pyplot(fig4)

# --- Chart 5: Correlation Matrix between indicators ---
st.subheader("5. Correlation Matrix between Indicators")
corr_df = df[['RETAIL SALES', 'WAREHOUSE SALES', 'RETAIL TRANSFERS', 'Total Revenue']]
corr_matrix = corr_df.corr()
fig5, ax5 = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='YlGnBu',
    fmt=".2f",
    linewidths=0.5,
    ax=ax5
)
ax5.set_title('Correlation Matrix between Sales Indicators')
st.pyplot(fig5)


# --- Chart 6: Train Model ---
st.subheader("5. Actual vs Predicted Retail Sales")
X = df[['WAREHOUSE SALES']]
y = df['RETAIL SALES']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.plot(range(len(y_test)), y_test.values, label='Actual', marker='o')
ax5.plot(range(len(y_pred)), y_pred, label='Predicted', marker='x')
ax5.set_title('Actual vs Predicted Retail Sales')
ax5.set_xlabel('Test Sample Index')
ax5.set_ylabel('Retail Sales')
ax5.legend()
ax5.grid(True)
st.pyplot(fig6)
