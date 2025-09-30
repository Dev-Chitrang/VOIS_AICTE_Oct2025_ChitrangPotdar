# Streamlit Airbnb EDA Dashboard - Complete Version with All 7 Tabs
# Save as app.py and run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, anderson, kstest, levene, kruskal, mannwhitneyu, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Airbnb EDA Dashboard", initial_sidebar_state="expanded")

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 9

# Helper functions
@st.cache_data
def load_data(path: str):
    try:
        df = pd.read_excel(path)
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    except Exception:
        df = pd.read_csv(path)
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    return df

def summary_numbers(df):
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_values": int(df.isnull().sum().sum())
    }

# Load data
DATA_PATH = "clean_data.xlsx"
st.title("📊 Airbnb Booking — Comprehensive EDA Dashboard")
st.markdown("*This dashboard presents detailed exploratory data analysis of Airbnb booking dataset*")

with st.spinner("Loading data..."):
    df = load_data(DATA_PATH)

# Create derived columns
if 'price' in df.columns:
    percentiles = df['price'].quantile([0.25, 0.50, 0.75]).values
    df['price_segment'] = pd.cut(df['price'],
                                  bins=[0, percentiles[0], percentiles[1], percentiles[2], df['price'].max()],
                                  labels=['Budget', 'Mid-range', 'Premium', 'Luxury'])
    if 'service_fee' in df.columns:
        df['service_fee_pct'] = (df['service_fee'] / df['price']) * 100

if 'construction_year' in df.columns:
    current_year = 2024
    df['property_age'] = current_year - df['construction_year']
    df['decade'] = (df['construction_year'] // 10) * 10

# Sidebar
st.sidebar.header("📁 Dataset Information")
info = summary_numbers(df)
st.sidebar.metric("Total Rows", f"{info['rows']:,}")
st.sidebar.metric("Total Columns", info['cols'])
st.sidebar.metric("Missing Values", info['missing_values'])

if st.sidebar.checkbox("Show Raw Data Sample"):
    st.sidebar.write("First 100 rows:")
    st.dataframe(df.head(100), height=300)

st.sidebar.markdown("---")
st.sidebar.header("🔍 Global Filters")

# Country filter
country_filter = 'All'
if 'country' in df.columns:
    countries = ['All'] + sorted(df['country'].dropna().unique().tolist())
    country_filter = st.sidebar.selectbox("Country", countries, index=0)
    if country_filter != 'All':
        df = df[df['country'] == country_filter]

# Neighbourhood filter
neigh_filter = 'All'
if 'neighbourhood' in df.columns:
    neighs = ['All'] + sorted(df['neighbourhood'].dropna().unique().tolist())[:100]
    neigh_filter = st.sidebar.selectbox("Neighbourhood", neighs, index=0)
    if neigh_filter != 'All':
        df = df[df['neighbourhood'] == neigh_filter]

st.sidebar.info(f"Filtered dataset: **{len(df):,}** rows")

# Tabs
tabs = st.tabs([
    "🎯 Key Visualizations",
    "📊 Univariate",
    "📈 Bivariate",
    "🔗 Multivariate",
    "📉 Distribution & Tests",
    "⚡ Advanced Analysis",
    "📋 Summary"
])

# =====================================================
# TAB 1: KEY VISUALIZATIONS
# =====================================================
with tabs[0]:
    st.header("🎯 Key Visualizations - Executive Summary")
    st.markdown("*All important charts in one place for quick overview and screenshot*")

    # Row 1: Price Analysis
    st.subheader("1️⃣ Price Distribution & Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'price' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            price = df['price'].dropna()
            sns.histplot(price, bins=40, kde=True, ax=ax, color='skyblue')
            ax.axvline(price.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean: ${price.mean():.0f}')
            ax.axvline(price.median(), color='green', linestyle='--', linewidth=1.5, label=f'Median: ${price.median():.0f}')
            ax.set_title("Price Distribution", fontsize=11, fontweight='bold')
            ax.set_xlabel("Price ($)", fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        if 'room_type' in df.columns and 'price' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.boxplot(data=df, x='room_type', y='price', ax=ax, palette='Set2')
            ax.set_title("Price by Room Type", fontsize=11, fontweight='bold')
            ax.set_xlabel("Room Type", fontsize=9)
            ax.set_ylabel("Price ($)", fontsize=9)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col3:
        if 'price_segment' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            segment_counts = df['price_segment'].value_counts()
            colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
            ax.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90)
            ax.set_title("Price Segment Distribution", fontsize=11, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # Row 2: Location & Reviews
    st.subheader("2️⃣ Location & Review Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'neighbourhood_group' in df.columns and 'price' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ng_price = df.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False)
            bars = ax.barh(ng_price.index, ng_price.values, color='coral', alpha=0.8)
            ax.set_title("Avg Price by Neighbourhood Group", fontsize=11, fontweight='bold')
            ax.set_xlabel("Average Price ($)", fontsize=9)
            ax.set_ylabel("Neighbourhood Group", fontsize=9)
            ax.tick_params(labelsize=8)
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, f'${width:.0f}',
                       ha='left', va='center', fontsize=8)
            ax.grid(alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        if 'number_of_reviews' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            reviews = df['number_of_reviews'].dropna()
            sns.histplot(reviews, bins=40, kde=True, ax=ax, color='orange')
            ax.axvline(reviews.mean(), color='red', linestyle='--', linewidth=1.5,
                      label=f'Mean: {reviews.mean():.0f}')
            ax.set_title("Number of Reviews Distribution", fontsize=11, fontweight='bold')
            ax.set_xlabel("Number of Reviews", fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col3:
        if 'availability_365' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            avail = df['availability_365'].dropna()
            sns.histplot(avail, bins=40, kde=True, ax=ax, color='green')
            ax.axvline(avail.mean(), color='red', linestyle='--', linewidth=1.5,
                      label=f'Mean: {avail.mean():.0f}')
            ax.set_title("Availability Distribution", fontsize=11, fontweight='bold')
            ax.set_xlabel("Days Available per Year", fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # Row 3: Correlations & Geographic
    st.subheader("3️⃣ Correlations & Geographic Distribution")
    col1, col2 = st.columns(2)

    with col1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(6, 5))
            key_cols = [c for c in ['price', 'service_fee', 'number_of_reviews',
                                    'reviews_per_month', 'availability_365',
                                    'minimum_nights', 'review_rate_number']
                       if c in numeric_cols]
            if len(key_cols) > 1:
                corr = df[key_cols].corr()
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
                           center=0, ax=ax, linewidths=0.5, square=True)
                ax.set_title("Correlation Matrix (Key Variables)", fontsize=11, fontweight='bold')
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(rotation=0, fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    with col2:
        lat_col = 'lat' if 'lat' in df.columns else ('latitude' if 'latitude' in df.columns else None)
        lon_col = 'long' if 'long' in df.columns else ('longitude' if 'longitude' in df.columns else None)

        if lat_col and lon_col and 'price' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(df[lon_col], df[lat_col], c=df['price'],
                               cmap='YlOrRd', alpha=0.5, s=15)
            ax.set_title("Geographic Distribution (Price)", fontsize=11, fontweight='bold')
            ax.set_xlabel("Longitude", fontsize=9)
            ax.set_ylabel("Latitude", fontsize=9)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Price ($)', fontsize=8)
            ax.grid(alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # Row 4: Host Analysis
    st.subheader("4️⃣ Host Analysis & Key Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'calculated_host_listings_count' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            host_cats = pd.cut(df['calculated_host_listings_count'],
                              bins=[0, 1, 5, 10, 50, df['calculated_host_listings_count'].max()],
                              labels=['Single', 'Small\n(2-5)', 'Medium\n(6-10)',
                                     'Large\n(11-50)', 'Enterprise\n(50+)'])
            cat_counts = host_cats.value_counts()
            colors_host = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
            ax.bar(cat_counts.index, cat_counts.values, color=colors_host, alpha=0.8)
            ax.set_title("Host Portfolio Distribution", fontsize=11, fontweight='bold')
            ax.set_xlabel("Host Category", fontsize=9)
            ax.set_ylabel("Number of Hosts", fontsize=9)
            ax.tick_params(axis='x', rotation=0, labelsize=8)
            ax.grid(alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        if 'host_identity_verified' in df.columns and 'number_of_reviews' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.violinplot(data=df, x='host_identity_verified', y='number_of_reviews',
                          ax=ax, palette='pastel')
            ax.set_title("Reviews by Verification Status", fontsize=11, fontweight='bold')
            ax.set_xlabel("Host Verified", fontsize=9)
            ax.set_ylabel("Number of Reviews", fontsize=9)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3, axis='y')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col3:
        if 'cancellation_policy' in df.columns and 'price' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            policy_price = df.groupby('cancellation_policy')['price'].mean().sort_values()
            bars = ax.barh(policy_price.index, policy_price.values, color='teal', alpha=0.8)
            ax.set_title("Avg Price by Cancellation Policy", fontsize=11, fontweight='bold')
            ax.set_xlabel("Average Price ($)", fontsize=9)
            ax.set_ylabel("Policy", fontsize=9)
            ax.tick_params(labelsize=8)
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, f'${width:.0f}',
                       ha='left', va='center', fontsize=8)
            ax.grid(alpha=0.3, axis='x')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    # Key Metrics
    st.subheader("📊 Key Metrics Summary")
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

    with metric_col1:
        if 'price' in df.columns:
            avg_price = df['price'].mean()
            st.metric("Average Price", f"${avg_price:.2f}")

    with metric_col2:
        if 'number_of_reviews' in df.columns:
            avg_reviews = df['number_of_reviews'].mean()
            st.metric("Avg Reviews/Listing", f"{avg_reviews:.1f}")

    with metric_col3:
        if 'availability_365' in df.columns:
            avg_avail = df['availability_365'].mean()
            st.metric("Avg Availability", f"{avg_avail:.0f} days")

    with metric_col4:
        if 'host_identity_verified' in df.columns:
            verified_pct = (df['host_identity_verified'] == True).mean() * 100
            st.metric("Verified Hosts", f"{verified_pct:.1f}%")

    with metric_col5:
        if 'calculated_host_listings_count' in df.columns:
            avg_listings = df['calculated_host_listings_count'].mean()
            st.metric("Avg Listings/Host", f"{avg_listings:.1f}")

# =====================================================
# TAB 2: UNIVARIATE
# =====================================================
with tabs[1]:
    st.header("📊 Univariate Analysis")

    st.subheader("💰 Price Distribution")
    if 'price' in df.columns:
        col1, col2, col3 = st.columns(3)
        price = df['price'].dropna()

        with col1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.histplot(price, bins=50, kde=True, ax=ax, color='skyblue')
            ax.axvline(price.mean(), color='red', linestyle='--', label=f'Mean: ${price.mean():.2f}')
            ax.axvline(price.median(), color='green', linestyle='--', label=f'Median: ${price.median():.2f}')
            ax.set_title("Price Histogram + KDE", fontweight='bold')
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.boxplot(x=price, ax=ax, color='lightblue')
            ax.set_title("Price Box Plot", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col3:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.violinplot(y=price, ax=ax, color='coral')
            ax.set_title("Price Violin Plot", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.write("**Price Statistics:**")
        st.dataframe(price.describe().to_frame().T)

    st.markdown("---")

    st.subheader("📅 Availability & Review Patterns")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'availability_365' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            avail = df['availability_365'].dropna()
            sns.histplot(avail, bins=50, kde=True, ax=ax, color='lightgreen')
            ax.axvline(avail.mean(), color='red', linestyle='--', label=f'Mean: {avail.mean():.1f}')
            ax.set_title("Availability Distribution", fontweight='bold')
            ax.set_xlabel("Days Available/Year")
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        if 'number_of_reviews' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            reviews = df['number_of_reviews'].dropna()
            sns.histplot(reviews, bins=50, kde=True, ax=ax, color='orange')
            ax.axvline(reviews.mean(), color='red', linestyle='--', label=f'Mean: {reviews.mean():.1f}')
            ax.set_title("Number of Reviews", fontweight='bold')
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col3:
        if 'reviews_per_month' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            rpm = df['reviews_per_month'].dropna()
            sns.histplot(rpm, bins=50, kde=True, ax=ax, color='salmon')
            ax.axvline(rpm.mean(), color='red', linestyle='--', label=f'Mean: {rpm.mean():.2f}')
            ax.set_title("Reviews per Month", fontweight='bold')
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    st.subheader("🏠 Property Age & Host Portfolio")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'construction_year' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.histplot(df['construction_year'].dropna(), bins=30, ax=ax, color='steelblue')
            ax.set_title("Construction Year", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        if 'decade' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            decade_counts = df['decade'].value_counts().sort_index()
            ax.bar(decade_counts.index.astype(str), decade_counts.values, color='navy', alpha=0.7)
            ax.set_title("Listings by Decade", fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col3:
        if 'calculated_host_listings_count' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            host_counts = df['calculated_host_listings_count'].value_counts().sort_index()[:20]
            ax.bar(host_counts.index, host_counts.values, color='purple', alpha=0.7)
            ax.set_title("Host Listings Count (Top 20)", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# =====================================================
# TAB 3: BIVARIATE
# =====================================================
with tabs[2]:
    st.header("📈 Bivariate Analysis")

    st.subheader("💰 Price vs Location")
    col1, col2 = st.columns(2)

    with col1:
        if 'neighbourhood_group' in df.columns and 'price' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='neighbourhood_group', y='price', ax=ax, palette='Set2')
            ax.set_title("Price by Neighbourhood Group", fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        if 'neighbourhood' in df.columns and 'price' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            top_neigh = df.groupby('neighbourhood')['price'].mean().sort_values(ascending=False).head(10)
            ax.barh(top_neigh.index, top_neigh.values, color='coral', alpha=0.8)
            ax.set_title("Top 10 Neighbourhoods by Avg Price", fontweight='bold')
            ax.set_xlabel("Average Price")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    st.subheader("🛏️ Room Type Analysis")
    col1, col2, col3 = st.columns(3)

    with col1:
        if 'room_type' in df.columns and 'price' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.boxplot(data=df, x='room_type', y='price', ax=ax, palette='Set3')
            ax.set_title("Price by Room Type", fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        if 'room_type' in df.columns and 'price' in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.violinplot(data=df, x='room_type', y='price', ax=ax, palette='muted')
            ax.set_title("Price Distribution (Violin)", fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col3:
        if 'room_type' in df.columns:
            st.write("**Room Type Statistics:**")
            st.dataframe(df.groupby('room_type')['price'].agg(['mean', 'median', 'count']))

    st.markdown("---")

    st.subheader("✅ Verification & Booking Impact")
    col1, col2 = st.columns(2)

    with col1:
        if 'host_identity_verified' in df.columns and 'number_of_reviews' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df, x='host_identity_verified', y='number_of_reviews',
                       ax=ax, palette='pastel')
            ax.set_title("Reviews by Verification Status", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        if 'instant_bookable' in df.columns and 'price' in df.columns:
            df_instant = df[df['instant_bookable'].notna()]
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.violinplot(data=df_instant, x='instant_bookable', y='price',
                          ax=ax, palette='coolwarm')
            ax.set_title("Price by Instant Booking", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    st.subheader("🔗 Price Correlations")
    col1, col2 = st.columns(2)

    with col1:
        if 'price' in df.columns and 'service_fee' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df['price'], df['service_fee'], alpha=0.3)
            z = np.polyfit(df['price'].dropna(), df['service_fee'].dropna(), 1)
            p = np.poly1d(z)
            ax.plot(df['price'].sort_values(), p(df['price'].sort_values()),
                   "r--", linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
            ax.set_title("Price vs Service Fee", fontweight='bold')
            ax.set_xlabel("Price")
            ax.set_ylabel("Service Fee")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        if 'availability_365' in df.columns and 'number_of_reviews' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df['availability_365'], df['number_of_reviews'], alpha=0.3, color='purple')
            ax.set_title("Availability vs Reviews", fontweight='bold')
            ax.set_xlabel("Availability (days/year)")
            ax.set_ylabel("Number of Reviews")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# =====================================================
# TAB 4: MULTIVARIATE
# =====================================================
with tabs[3]:
    st.header("🔗 Multivariate Analysis")

    st.subheader("🗺️ Geographic Distribution")
    lat_col = 'lat' if 'lat' in df.columns else ('latitude' if 'latitude' in df.columns else None)
    lon_col = 'long' if 'long' in df.columns else ('longitude' if 'longitude' in df.columns else None)

    if lat_col and lon_col and 'price' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(df[lon_col], df[lat_col], c=df['price'],
                               cmap='YlOrRd', alpha=0.6, s=20)
            ax.set_title("Listings by Location (Color=Price)", fontweight='bold')
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.colorbar(scatter, ax=ax, label='Price ($)')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            sizes = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min() + 1e-9) * 100 + 10
            ax.scatter(df[lon_col], df[lat_col], s=sizes, alpha=0.4, c=df['price'], cmap='viridis')
            ax.set_title("Bubble Chart (Size & Color=Price)", fontweight='bold')
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    st.subheader("🏘️ Room Type Performance by Location")
    if 'neighbourhood_group' in df.columns and 'room_type' in df.columns and 'price' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            pivot = df.pivot_table(values='price', index='neighbourhood_group',
                                   columns='room_type', aggfunc='mean')
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap='RdYlGn', ax=ax)
            ax.set_title("Mean Price Heatmap", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            pivot_count = df.pivot_table(values='price', index='neighbourhood_group',
                                         columns='room_type', aggfunc='count')
            sns.heatmap(pivot_count, annot=True, fmt=".0f", cmap='Blues', ax=ax)
            ax.set_title("Listing Count Heatmap", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    st.subheader("📊 Multi-Factor Analysis")
    col1, col2 = st.columns(2)

    with col1:
        if 'cancellation_policy' in df.columns and 'price_segment' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 5))
            policy_segment = df.groupby(['cancellation_policy', 'price_segment']).size().unstack(fill_value=0)
            policy_segment.plot(kind='bar', stacked=False, ax=ax, colormap='viridis', alpha=0.8)
            ax.set_title("Cancellation Policy by Price Segment", fontweight='bold')
            ax.set_xlabel("Cancellation Policy")
            ax.set_ylabel("Count")
            ax.legend(title='Price Segment', fontsize=8)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        # FIXED: Using matplotlib instead of seaborn boxplot to avoid the hue bug
        # Multi-Factor Analysis → Boxplot Verification × Booking Type
        if 'host_identity_verified' in df.columns and 'instant_bookable' in df.columns and 'price' in df.columns:
            df_multi = df[df['instant_bookable'].notna()].copy()
            df_multi['booking_type'] = df_multi['instant_bookable'].map({0: 'Request', 1: 'Instant'})

            fig, ax = plt.subplots(figsize=(6, 5))

            verified_vals = df_multi['host_identity_verified'].unique()
            booking_vals = df_multi['booking_type'].unique()

            positions = []
            data_to_plot = []
            labels = []
            colors = []
            color_map = {'Request': '#66c2a5', 'Instant': '#fc8d62'}

            pos = 0
            for i, verified in enumerate(sorted(verified_vals)):
                group_added = False
                for j, booking in enumerate(sorted(booking_vals)):
                    subset = df_multi[
                        (df_multi['host_identity_verified'] == verified) &
                        (df_multi['booking_type'] == booking)
                    ]['price'].dropna()
                    if len(subset) > 0:
                        data_to_plot.append(subset)
                        positions.append(pos)
                        labels.append(f"{verified}\n{booking}")
                        colors.append(color_map.get(booking, '#999999'))
                        group_added = True
                        pos += 1
                if group_added:
                    pos += 0.5  # space between groups

            if len(data_to_plot) > 0:
                bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_xticks(positions)
                ax.set_xticklabels(labels, fontsize=8)
                ax.set_title("Price: Verification × Booking Type", fontweight='bold')
                ax.set_ylabel("Price ($)")
                ax.grid(alpha=0.3, axis='y')

                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=color_map['Request'], alpha=0.7, label='Request'),
                                Patch(facecolor=color_map['Instant'], alpha=0.7, label='Instant')]
                ax.legend(handles=legend_elements, fontsize=8)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("⚠️ Not enough data to plot Verification × Booking Type boxplot.")

    st.markdown("---")

    st.subheader("🔢 Comprehensive Correlation Matrix")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        key_cols = [c for c in ['price', 'service_fee', 'number_of_reviews',
                                'reviews_per_month', 'availability_365',
                                'minimum_nights', 'review_rate_number',
                                'calculated_host_listings_count']
                   if c in numeric_cols]

        if len(key_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df[key_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                       center=0, ax=ax, linewidths=1, square=True)
            ax.set_title("Correlation Matrix", fontweight='bold', fontsize=14)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# =====================================================
# TAB 5: DISTRIBUTION & TESTS
# =====================================================
with tabs[4]:
    st.header("📉 Distribution Analysis & Statistical Tests")

    st.subheader("📊 Normality Tests")
    if 'price' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            price_arr = df['price'].dropna()
            sample = price_arr.sample(min(5000, len(price_arr)), random_state=42)

            st.write("**Price Distribution Tests:**")
            try:
                sh_stat, sh_p = shapiro(sample)
                st.write(f"- Shapiro-Wilk: statistic={sh_stat:.6f}, p-value={sh_p:.6f}")
                st.write(f"  → {'Normal' if sh_p > 0.05 else 'Not Normal'} distribution")
            except:
                st.write("- Shapiro-Wilk: Could not compute")

            try:
                and_res = anderson(price_arr)
                st.write(f"- Anderson-Darling: statistic={and_res.statistic:.6f}")
            except:
                st.write("- Anderson-Darling: Could not compute")

            skew = price_arr.skew()
            kurt = price_arr.kurtosis()
            st.write(f"- Skewness: {skew:.4f}")
            st.write(f"- Kurtosis: {kurt:.4f}")

        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            stats.probplot(price_arr, dist="norm", plot=ax)
            ax.set_title("Q-Q Plot: Price", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    st.subheader("📈 Distribution Comparisons")
    col1, col2 = st.columns(2)

    with col1:
        if 'number_of_reviews' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            reviews = df['number_of_reviews'].dropna()
            sns.histplot(reviews, bins=50, kde=True, ax=ax, color='orange')
            ax.set_title("Reviews Distribution", fontweight='bold')
            ax.text(0.7, 0.9, f'Skew: {reviews.skew():.2f}', transform=ax.transAxes)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col2:
        if 'service_fee' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sf = df['service_fee'].dropna()
            stats.probplot(sf, dist="norm", plot=ax)
            ax.set_title("Q-Q Plot: Service Fee", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    st.subheader("🔬 Group Comparison Tests")
    if 'room_type' in df.columns and 'price' in df.columns:
        st.write("**Room Type Price Comparison:**")

        groups = [df[df['room_type']==rt]['price'].dropna() for rt in df['room_type'].unique()]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) >= 2:
            try:
                lev_s, lev_p = levene(*groups)
                st.write(f"- Levene's Test (Variance Equality): statistic={lev_s:.4f}, p-value={lev_p:.4f}")
                st.write(f"  → Variances are {'equal' if lev_p > 0.05 else 'not equal'}")
            except:
                st.write("- Levene's Test: Could not compute")

            try:
                kr_s, kr_p = kruskal(*groups)
                st.write(f"- Kruskal-Wallis Test: statistic={kr_s:.4f}, p-value={kr_p:.4f}")
                st.write(f"  → {'Significant' if kr_p < 0.05 else 'No significant'} difference between groups")
            except:
                st.write("- Kruskal-Wallis: Could not compute")

    if 'host_identity_verified' in df.columns and 'number_of_reviews' in df.columns:
        st.write("\n**Verification Effect on Reviews:**")
        try:
            verified = df[df['host_identity_verified'] == 'verified']['number_of_reviews'].dropna()
            unverified = df[df['host_identity_verified'] == 'unverified']['number_of_reviews'].dropna()

            if len(verified) > 0 and len(unverified) > 0:
                stat, p = mannwhitneyu(verified, unverified, alternative='two-sided')
                st.write(f"- Mann-Whitney U Test: statistic={stat:.4f}, p-value={p:.4f}")
                st.write(f"  → {'Significant' if p < 0.05 else 'No significant'} difference")
                st.write(f"  → Mean (Verified): {verified.mean():.2f}, Mean (Unverified): {unverified.mean():.2f}")
        except:
            st.write("- Mann-Whitney U Test: Could not compute")

# =====================================================
# TAB 6: ADVANCED ANALYSIS
# =====================================================
with tabs[5]:
    st.header("⚡ Advanced Analysis")

    st.subheader("💎 Price Segmentation & Profiling")
    if 'price_segment' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Segment Characteristics:**")
            agg_cols = {}
            if 'price' in df.columns:
                agg_cols['price'] = ['mean', 'min', 'max', 'count']
            if 'number_of_reviews' in df.columns:
                agg_cols['number_of_reviews'] = 'mean'
            if 'availability_365' in df.columns:
                agg_cols['availability_365'] = 'mean'

            if agg_cols:
                agg_df = df.groupby('price_segment').agg(agg_cols).round(2)
                st.dataframe(agg_df)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            for seg in df['price_segment'].dropna().unique():
                sns.kdeplot(df[df['price_segment']==seg]['price'].dropna(),
                           label=str(seg), ax=ax, linewidth=2)
            ax.set_title("Price Distribution by Segment", fontweight='bold')
            ax.set_xlabel("Price")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    st.subheader("🎯 Outlier Detection")
    col1, col2 = st.columns(2)

    with col1:
        if 'review_rate_number' in df.columns:
            rrn = df['review_rate_number'].dropna()
            q1 = rrn.quantile(0.25)
            q3 = rrn.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5*iqr
            upper = q3 + 1.5*iqr
            outliers = ((rrn < lower) | (rrn > upper)).sum()

            st.write("**Review Rate Number Outliers:**")
            st.write(f"- Total Outliers: {outliers} ({outliers/len(rrn)*100:.2f}%)")
            st.write(f"- Lower Bound: {lower:.2f}")
            st.write(f"- Upper Bound: {upper:.2f}")
            st.write(f"- IQR: {iqr:.2f}")

    with col2:
        if 'review_rate_number' in df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df['review_rate_number'], ax=ax, color='lightcoral')
            ax.set_title("Review Rate Outliers", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")

    st.subheader("📅 Time-based Review Patterns")
    if 'last_review' in df.columns:
        df_time = df[df['last_review'].notna()].copy()

        if len(df_time) > 0:
            df_time['review_month'] = df_time['last_review'].dt.month
            df_time['review_year'] = df_time['last_review'].dt.year

            col1, col2 = st.columns(2)

            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                monthly = df_time.groupby(df_time['last_review'].dt.to_period('M')).size()
                monthly.index = monthly.index.to_timestamp()
                monthly.plot(ax=ax, marker='o', linewidth=2, color='blue')
                ax.set_title("Reviews Over Time", fontweight='bold')
                ax.set_xlabel("Date")
                ax.set_ylabel("Number of Reviews")
                ax.grid(alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                heat_data = df_time.groupby(['review_year','review_month']).size().unstack(fill_value=0)
                if len(heat_data) > 0:
                    sns.heatmap(heat_data, annot=False, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})
                    ax.set_title("Review Heatmap (Year × Month)", fontweight='bold')
                    ax.set_xlabel("Month")
                    ax.set_ylabel("Year")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    st.markdown("---")

    st.subheader("👤 Host Performance Analysis")
    if 'calculated_host_listings_count' in df.columns:
        col1, col2 = st.columns(2)

        with col1:
            df['host_type'] = df['calculated_host_listings_count'].apply(
                lambda x: 'Multi-Host' if x > 1 else 'Single-Host'
            )

            if 'price' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=df, x='host_type', y='price', ax=ax, palette='Set1')
                ax.set_title("Price: Multi-Host vs Single-Host", fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        with col2:
            if 'number_of_reviews' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.violinplot(data=df, x='host_type', y='number_of_reviews', ax=ax, palette='Set2')
                ax.set_title("Reviews: Multi-Host vs Single-Host", fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# =====================================================
# TAB 7: SUMMARY
# =====================================================
with tabs[6]:
    st.header("📋 Summary Statistics & Insights")

    st.subheader("📊 Descriptive Statistics")
    try:
        desc = df.describe().T
        desc['missing'] = df.isnull().sum()
        st.dataframe(desc.style.background_gradient(cmap='YlOrRd', subset=['mean', 'std']))
    except:
        st.write("Could not generate summary statistics")

    st.markdown("---")

    st.subheader("💡 Key Insights")

    insights = []

    if 'price' in df.columns:
        avg_price = df['price'].mean()
        med_price = df['price'].median()
        insights.append(f"**Average Price:** ${avg_price:.2f} (Median: ${med_price:.2f})")
        insights.append(f"**Price Range:** ${df['price'].min():.2f} - ${df['price'].max():.2f}")

    if 'price_segment' in df.columns:
        top_segment = df['price_segment'].mode()[0]
        insights.append(f"**Most Common Segment:** {top_segment}")

    if 'room_type' in df.columns:
        top_room = df['room_type'].mode()[0]
        insights.append(f"**Most Common Room Type:** {top_room}")

    if 'availability_365' in df.columns:
        avg_avail = df['availability_365'].mean()
        full_avail = (df['availability_365'] == 365).sum()
        insights.append(f"**Average Availability:** {avg_avail:.0f} days/year")
        insights.append(f"**Fully Available Listings:** {full_avail} ({full_avail/len(df)*100:.1f}%)")

    if 'number_of_reviews' in df.columns:
        avg_reviews = df['number_of_reviews'].mean()
        no_reviews = (df['number_of_reviews'] == 0).sum()
        insights.append(f"**Average Reviews per Listing:** {avg_reviews:.1f}")
        insights.append(f"**Listings with No Reviews:** {no_reviews} ({no_reviews/len(df)*100:.1f}%)")

    if 'host_identity_verified' in df.columns:
        verified = (df['host_identity_verified'] == 'verified').sum()
        insights.append(f"**Verified Hosts:** {verified} ({verified/len(df)*100:.1f}%)")

    if 'calculated_host_listings_count' in df.columns:
        avg_listings = df['calculated_host_listings_count'].mean()
        multi_host = (df['calculated_host_listings_count'] > 1).sum()
        insights.append(f"**Average Listings per Host:** {avg_listings:.1f}")
        insights.append(f"**Multi-Property Hosts:** {multi_host} ({multi_host/len(df)*100:.1f}%)")

    if 'cancellation_policy' in df.columns:
        top_policy = df['cancellation_policy'].mode()[0]
        insights.append(f"**Most Common Cancellation Policy:** {top_policy}")

    col1, col2 = st.columns(2)
    mid = len(insights) // 2

    with col1:
        for insight in insights[:mid]:
            st.markdown(f"✓ {insight}")

    with col2:
        for insight in insights[mid:]:
            st.markdown(f"✓ {insight}")

    st.markdown("---")

    st.subheader("🎯 Business Recommendations")

    recommendations = [
        "**Pricing Strategy:** Focus on the most popular price segments identified in the analysis",
        "**Host Verification:** Encourage more hosts to verify their identity for better trust and reviews",
        "**Availability Optimization:** Properties with higher availability tend to get more bookings",
        "**Room Type Focus:** Invest in the most popular room types in high-demand neighborhoods",
        "**Review Management:** Listings with more reviews attract more bookings - encourage guest reviews",
        "**Geographic Targeting:** Premium locations command higher prices - focus marketing efforts there",
        "**Cancellation Policy:** Flexible policies may attract more bookings but balance with risk",
        "**Multi-listing Hosts:** Support hosts with multiple properties as they form a significant portion"
    ]

    for i, rec in enumerate(recommendations, 1):
        st.write(f"{i}. {rec}")

    st.markdown("---")

    st.subheader("📝 Data Quality Notes")

    missing_summary = df.isnull().sum().sort_values(ascending=False)
    missing_summary = missing_summary[missing_summary > 0]

    if len(missing_summary) > 0:
        st.write("**Columns with Missing Values:**")
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(missing_summary.head(10))
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            missing_summary.head(10).plot(kind='barh', ax=ax, color='salmon')
            ax.set_title("Top 10 Columns with Missing Values", fontweight='bold')
            ax.set_xlabel("Missing Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.success("✅ No missing values in the dataset!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📊 Airbnb EDA Dashboard | Built with Streamlit & Python</p>
    <p>For detailed analysis, refer to the Jupyter notebook: Airbnb_booking_analysis.ipynb</p>
</div>
""", unsafe_allow_html=True)
