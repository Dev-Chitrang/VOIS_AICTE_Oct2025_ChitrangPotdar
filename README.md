# 🏨 Airbnb Hotel Booking Data Analysis

## 📌 Problem Statement
The hospitality industry has undergone significant transformation due to the rise of online platforms like **Airbnb**. While Airbnb provides flexible lodging options, both hosts and guests face challenges:

- **Hosts** struggle to set competitive pricing, maximize occupancy, and maintain positive reviews.
- **Guests** aim to find affordable, safe, and quality stays.

### 🎯 Project Goal
This project analyzes Airbnb booking data to:
- Identify factors influencing successful bookings.
- Understand guest behavior and preferences.
- Explore host performance metrics.
- Provide **data-driven insights** for improved decision-making.

---

## 🔍 Objectives of the Analysis

### **Category 1: Univariate Analysis**
- Price distribution with 4 visualization types
- Availability patterns analysis
- Review frequency distributions
- Listing age analysis by decade
- Host portfolio size analysis
- Minimum nights requirements

### **Category 2: Bivariate Analysis**
- Price vs Location (neighbourhoods)
- Room type pricing strategies
- Verification impact on reviews (*Mann-Whitney U test*)
- Instant booking vs pricing (*t-test*)
- Cancellation policy impact
- Service fee relationships (*Pearson/Spearman correlation*)
- Reviews vs availability analysis
- Property age vs price
- Minimum nights vs reviews
- Host performance metrics

### **Category 3: Multivariate Analysis**
- Geographical price mapping (*2D heatmaps*)
- Room type performance by location
- Verification + Instant booking + Pricing (*ANOVA*)
- Policy-Price-Reviews nexus (*3D visualizations*)
- Temporal-Spatial-Price analysis
- Host performance dashboard
- Availability-Price-Reviews relationships
- Country-wise comprehensive comparisons

### **Category 4: Distribution & Skewness**
- Price normality tests (*Shapiro-Wilk, Anderson-Darling*)
- Review metrics skewness analysis
- Service fee distribution tests (*Kolmogorov-Smirnov*)
- Comparative distribution analysis (*Levene's, Kruskal-Wallis*)
- Q-Q plots for all major metrics

### **Category 5: Advanced Statistical Analysis**
- Price segmentation & profiling
- Time-based review patterns (*seasonal analysis*)
- Complete correlation matrix with multicollinearity detection
- Pair plots for key variables

---

## 📂 Project Structure
- **`Airbnb_booking_analysis.ipynb`** → Contains the **detailed data analysis**, statistical testing, and visualizations.
- **`app.py`** → Streamlit-based **analytical dashboard** for interactive exploration of the dataset.
- **`requirements.txt`** → Dependencies required to run the project.
- **`images/`** → Dashboard preview screenshots and visual outputs.

---

## 📊 Analytical Dashboard (Add-On)
An **interactive dashboard** was developed using **Streamlit** as an extension of the notebook analysis.

With the dashboard, users can:
- Explore Airbnb datasets visually.
- Perform univariate, bivariate, and multivariate analysis dynamically.
- Access statistical insights interactively.

---

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, SciPy, Statsmodels, Scikit-learn)
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Statistical Testing**: Shapiro-Wilk, ANOVA, Mann-Whitney U, T-tests, Correlations
- **Dashboard**: Streamlit

---

## 🚀 How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/airbnb-data-analysis.git
   cd airbnb-data-analysis

2. Create virtual environment and install dependencies
    ```bash
        python -m venv .venv
        source .venv/bin/activate    # On Windows: .venv\Scripts\activate
        pip install -r requirements.txt

3. Run streamlit dashboard
    ```bash
        streamlit run app.py

## Key Insights

- Pricing varies significantly by neighborhood and room type.

- Verification status and instant booking impact guest reviews and pricing.

- Cancellation policies show strong relationships with guest satisfaction.

- Seasonality trends reveal peak booking times and review surges.

- Skewness and distribution tests highlight the non-normal nature of pricing and review metrics.

Sample Dashboard Preview
![Test Image 1](images\screencapture-localhost-8501-2025-09-30-18_01_18.png)

## Conclusion

- The core analysis is performed in Airbnb_booking_analysis.ipynb, covering in-depth statistical and exploratory data analysis.

- The Streamlit dashboard is an add-on feature that allows interactive exploration and visualization of Airbnb booking trends.

Together, they provide a comprehensive, data-driven view of Airbnb performance, supporting:

    - Optimized pricing strategies.

    - Improved guest satisfaction.

    - Enhanced overall market competitiveness.

## Contributing

Contributions are welcome! Feel free to fork this repository, open issues, and submit pull requests.

## License

This project is licensed under the MIT License.
