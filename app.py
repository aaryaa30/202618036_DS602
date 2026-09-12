import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Statistical Analysis",
    page_icon="📊",
    layout="wide"
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    return df


df = load_data()


# ---------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------
df["sex"] = df["sex"].astype(str)
df["smoker"] = df["smoker"].astype(str)
df["region"] = df["region"].astype(str)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------
st.title("📊 Medical Insurance Cost Statistical Analysis")
st.write(
    "Interactive statistical analysis, hypothesis testing, "
    "OLS regression, prediction and diagnostic analysis."
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("Filters")

min_age = int(df["age"].min())
max_age = int(df["age"].max())

age_range = st.sidebar.slider(
    "Age Range",
    min_age,
    max_age,
    (min_age, max_age)
)

selected_sex = st.sidebar.multiselect(
    "Sex",
    options=df["sex"].unique(),
    default=list(df["sex"].unique())
)

selected_smoker = st.sidebar.multiselect(
    "Smoker",
    options=df["smoker"].unique(),
    default=list(df["smoker"].unique())
)

selected_region = st.sidebar.multiselect(
    "Region",
    options=df["region"].unique(),
    default=list(df["region"].unique())
)


filtered_df = df[
    (df["age"].between(age_range[0], age_range[1])) &
    (df["sex"].isin(selected_sex)) &
    (df["smoker"].isin(selected_smoker)) &
    (df["region"].isin(selected_region))
]


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📈 Data Exploration",
    "🧪 Hypothesis Testing Lab",
    "🤖 Prediction & Diagnostics"
])


# =========================================================
# TAB 1 - DATA EXPLORATION
# =========================================================
with tab1:

    st.header("Data Exploration")

    # Dataset summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", len(df))

    with col2:
        st.metric("Filtered Records", len(filtered_df))

    with col3:
        st.metric("Features", df.shape[1])

    with col4:
        st.metric("Average Charges", f"${df['charges'].mean():,.2f}")

    st.subheader("Dataset Preview")

    # Only show first 10 rows
    st.dataframe(filtered_df.head(10), use_container_width=True)

    # -----------------------------------------------------
    # DESCRIPTIVE STATISTICS
    # -----------------------------------------------------
    st.subheader("Descriptive Statistics")

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    descriptive = pd.DataFrame({
        "Mean": df[numerical_columns].mean(),
        "Median": df[numerical_columns].median(),
        "Std Dev": df[numerical_columns].std(),
        "IQR": df[numerical_columns].quantile(0.75)
                - df[numerical_columns].quantile(0.25),
        "Skewness": df[numerical_columns].skew(),
        "Kurtosis": df[numerical_columns].kurt()
    })

    st.dataframe(
        descriptive.round(3),
        use_container_width=True
    )

    # -----------------------------------------------------
    # DISTRIBUTION PLOT
    # -----------------------------------------------------
    st.subheader("Distribution Plot")

    selected_numeric = st.selectbox(
        "Select numerical variable",
        numerical_columns
    )

    fig_hist = px.histogram(
        filtered_df,
        x=selected_numeric,
        marginal="box",
        nbins=30,
        title=f"Distribution of {selected_numeric}"
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

    # -----------------------------------------------------
    # SCATTER PLOT
    # -----------------------------------------------------
    st.subheader("Bivariate Analysis")

    col1, col2 = st.columns(2)

    with col1:
        x_variable = st.selectbox(
            "X Variable",
            numerical_columns,
            index=0
        )

    with col2:
        y_variable = st.selectbox(
            "Y Variable",
            numerical_columns,
            index=min(3, len(numerical_columns) - 1)
        )

    fig_scatter = px.scatter(
        filtered_df,
        x=x_variable,
        y=y_variable,
        color="smoker",
        hover_data=["age", "bmi", "charges"],
        title=f"{x_variable} vs {y_variable}"
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

    # -----------------------------------------------------
    # CORRELATION MATRIX
    # -----------------------------------------------------
    st.subheader("Correlation Matrix")

    correlation = filtered_df[numerical_columns].corr()

    fig_corr = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )


# =========================================================
# TAB 2 - HYPOTHESIS TESTING
# =========================================================
with tab2:

    st.header("Hypothesis Testing Lab")

    st.write("Significance level: α = 0.05")

    # -----------------------------------------------------
    # TEST 1 - TWO GROUP COMPARISON
    # -----------------------------------------------------
    st.subheader("Hypothesis Test 1: Compare Two Groups")

    st.write(
        "Select a categorical variable having two groups and "
        "a numerical variable to compare."
    )

    categorical_two_group = st.selectbox(
        "Select Categorical Factor",
        ["sex", "smoker"],
        key="two_group_category"
    )

    numerical_two_group = st.selectbox(
        "Select Numerical Metric",
        ["age", "bmi", "children", "charges"],
        index=3,
        key="two_group_numeric"
    )

    group_values = df[categorical_two_group].dropna().unique()

    if len(group_values) == 2:

        group1 = df[
            df[categorical_two_group] == group_values[0]
        ][numerical_two_group].dropna()

        group2 = df[
            df[categorical_two_group] == group_values[1]
        ][numerical_two_group].dropna()

        st.write(
            f"**H₀:** There is no significant difference in "
            f"{numerical_two_group} between the two groups."
        )

        st.write(
            f"**H₁:** There is a significant difference in "
            f"{numerical_two_group} between the two groups."
        )

        # Shapiro-Wilk normality test
        sample1 = group1.sample(
            min(500, len(group1)),
            random_state=42
        )

        sample2 = group2.sample(
            min(500, len(group2)),
            random_state=42
        )

        shapiro1 = stats.shapiro(sample1)
        shapiro2 = stats.shapiro(sample2)

        # Levene's equal variance test
        levene_result = stats.levene(
            group1,
            group2
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                f"Shapiro p-value ({group_values[0]})",
                f"{shapiro1.pvalue:.6f}"
            )

        with col2:
            st.metric(
                f"Shapiro p-value ({group_values[1]})",
                f"{shapiro2.pvalue:.6f}"
            )

        with col3:
            st.metric(
                "Levene p-value",
                f"{levene_result.pvalue:.6f}"
            )

        # Decide whether data is normally distributed
        normal_data = (
            shapiro1.pvalue > 0.05
            and shapiro2.pvalue > 0.05
        )

        equal_variance = levene_result.pvalue > 0.05

        # Automatically select statistical test
        if normal_data:

            test_result = stats.ttest_ind(
                group1,
                group2,
                equal_var=equal_variance
            )

            selected_test = (
                "Two-Sample Independent t-test"
            )

        else:

            test_result = stats.mannwhitneyu(
                group1,
                group2,
                alternative="two-sided"
            )

            selected_test = "Mann-Whitney U Test"

        st.write(
            f"### Automatically Selected Test: {selected_test}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Test Statistic",
                f"{test_result.statistic:.4f}"
            )

        with col2:
            st.metric(
                "p-value",
                f"{test_result.pvalue:.8f}"
            )

        # Final conclusion
        if test_result.pvalue < 0.05:

            st.error(
                "Reject H₀: There is a statistically significant "
                f"difference in {numerical_two_group} between "
                "the selected groups."
            )

        else:

            st.success(
                "Fail to Reject H₀: There is not enough evidence "
                f"to conclude that {numerical_two_group} differs "
                "significantly between the selected groups."
            )

    else:

        st.warning(
            "Please select a categorical variable containing "
            "exactly two groups."
        )


    # -----------------------------------------------------
    # TEST 2 - ONE-WAY ANOVA
    # -----------------------------------------------------
    st.divider()

    st.subheader("Hypothesis Test 2: One-Way ANOVA")

    st.write(
        "Select a categorical factor and a numerical metric. "
        "ANOVA checks whether the numerical metric differs "
        "across three or more groups."
    )

    categorical_anova = st.selectbox(
        "Select Categorical Factor",
        ["region"],
        key="anova_category"
    )

    numerical_anova = st.selectbox(
        "Select Numerical Metric",
        ["age", "bmi", "children", "charges"],
        index=3,
        key="anova_numeric"
    )

    anova_groups = [
        group[numerical_anova].dropna().values
        for _, group in df.groupby(categorical_anova)
    ]

    anova_group_names = df[
        categorical_anova
    ].dropna().unique()

    if len(anova_groups) >= 3:

        st.write(
            f"**H₀:** The mean {numerical_anova} is equal "
            f"across all {categorical_anova} groups."
        )

        st.write(
            f"**H₁:** At least one {categorical_anova} group "
            f"has a different mean {numerical_anova}."
        )

        anova_result = stats.f_oneway(
            *anova_groups
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "F-statistic",
                f"{anova_result.statistic:.4f}"
            )

        with col2:
            st.metric(
                "p-value",
                f"{anova_result.pvalue:.8f}"
            )

        if anova_result.pvalue < 0.05:

            st.error(
                "Reject H₀: There is a statistically significant "
                f"difference in {numerical_anova} across the "
                f"{categorical_anova} groups."
            )

        else:

            st.success(
                "Fail to Reject H₀: There is not enough evidence "
                f"of a significant difference in {numerical_anova} "
                f"across the {categorical_anova} groups."
            )

        # ANOVA visualization
        fig_box = px.box(
            df,
            x=categorical_anova,
            y=numerical_anova,
            color=categorical_anova,
            title=(
                f"{numerical_anova} by "
                f"{categorical_anova}"
            )
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        )

    else:

        st.warning(
            "ANOVA requires at least three groups."
        )


# =========================================================
# TAB 3 - PREDICTION & DIAGNOSTICS
# =========================================================
with tab3:

    st.header("Live Prediction & Diagnostics")

    # -----------------------------------------------------
    # OLS MODEL
    # -----------------------------------------------------
    st.subheader("Multiple Linear Regression")

    st.write(
        "Model: charges ~ age + bmi + children + sex + smoker + region"
    )

    model = smf.ols(
        "charges ~ age + bmi + children + C(sex) + C(smoker) + C(region)",
        data=df
    ).fit()

    # Model summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "R²",
            f"{model.rsquared:.4f}"
        )

    with col2:
        st.metric(
            "Adjusted R²",
            f"{model.rsquared_adj:.4f}"
        )

    with col3:
        st.metric(
            "F-statistic",
            f"{model.fvalue:.2f}"
        )

    with col4:
        st.metric(
            "Model p-value",
            f"{model.f_pvalue:.6g}"
        )

    # -----------------------------------------------------
    # COEFFICIENT TABLE
    # -----------------------------------------------------
    st.subheader("Model Coefficients")

    coefficient_table = pd.DataFrame({
        "Coefficient": model.params,
        "P-value": model.pvalues,
        "CI Lower 95%": model.conf_int()[0],
        "CI Upper 95%": model.conf_int()[1]
    })

    st.dataframe(
        coefficient_table.round(4),
        use_container_width=True
    )

    # -----------------------------------------------------
    # LIVE PREDICTION
    # -----------------------------------------------------
    st.subheader("Live Medical Charges Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        input_age = st.number_input(
            "Age",
            min_value=int(df["age"].min()),
            max_value=int(df["age"].max()),
            value=30
        )

        input_bmi = st.number_input(
            "BMI",
            min_value=float(df["bmi"].min()),
            max_value=float(df["bmi"].max()),
            value=30.0
        )

    with col2:
        input_children = st.number_input(
            "Children",
            min_value=int(df["children"].min()),
            max_value=int(df["children"].max()),
            value=0
        )

        input_sex = st.selectbox(
            "Sex",
            df["sex"].unique()
        )

    with col3:
        input_smoker = st.selectbox(
            "Smoker",
            df["smoker"].unique()
        )

        input_region = st.selectbox(
            "Region",
            df["region"].unique()
        )

    prediction_data = pd.DataFrame({
        "age": [input_age],
        "bmi": [input_bmi],
        "children": [input_children],
        "sex": [input_sex],
        "smoker": [input_smoker],
        "region": [input_region]
    })

    prediction = model.get_prediction(
        prediction_data
    )

    prediction_summary = prediction.summary_frame(
        alpha=0.05
    )

    predicted_charge = prediction_summary[
        "mean"
    ].iloc[0]

    mean_ci_lower = prediction_summary[
        "mean_ci_lower"
    ].iloc[0]

    mean_ci_upper = prediction_summary[
        "mean_ci_upper"
    ].iloc[0]

    obs_ci_lower = prediction_summary[
        "obs_ci_lower"
    ].iloc[0]

    obs_ci_upper = prediction_summary[
        "obs_ci_upper"
    ].iloc[0]

    st.success(
        f"Predicted Medical Charge: **${predicted_charge:,.2f}**"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"95% Confidence Interval: "
            f"${mean_ci_lower:,.2f} to ${mean_ci_upper:,.2f}"
        )

    with col2:
        st.info(
            f"95% Prediction Interval: "
            f"${obs_ci_lower:,.2f} to ${obs_ci_upper:,.2f}"
        )

    # -----------------------------------------------------
    # RESIDUAL DIAGNOSTICS
    # -----------------------------------------------------
    st.subheader("Residual Diagnostics")

    fitted_values = model.fittedvalues
    residuals = model.resid

    # Residual vs fitted
    fig_residual = px.scatter(
        x=fitted_values,
        y=residuals,
        labels={
            "x": "Fitted Values",
            "y": "Residuals"
        },
        title="Residuals vs Fitted Values"
    )

    fig_residual.add_hline(
        y=0,
        line_dash="dash"
    )

    st.plotly_chart(
        fig_residual,
        use_container_width=True
    )

    # -----------------------------------------------------
    # Q-Q PLOT
    # -----------------------------------------------------
    st.subheader("Q-Q Plot")

    qq = sm.qqplot(
        residuals,
        line="45",
        fit=True
    )

    st.pyplot(qq.figure)

    # -----------------------------------------------------
    # JARQUE-BERA TEST
    # -----------------------------------------------------
    jb_test = stats.jarque_bera(
        residuals
    )

    st.write("### Normality Test")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Jarque-Bera Statistic",
            f"{jb_test.statistic:.4f}"
        )

    with col2:
        st.metric(
            "Jarque-Bera p-value",
            f"{jb_test.pvalue:.8f}"
        )

    if jb_test.pvalue > 0.05:
        st.success(
            "Fail to Reject H₀: Residuals are reasonably "
            "consistent with a normal distribution."
        )
    else:
        st.warning(
            "Reject H₀: Residuals show evidence of "
            "departure from normality."
        )

    # -----------------------------------------------------
    # HOMOSCEDASTICITY - BREUSCH PAGAN
    # -----------------------------------------------------
    st.subheader("Homoscedasticity Test")

    bp_test = het_breuschpagan(
        residuals,
        model.model.exog
    )

    bp_pvalue = bp_test[1]

    st.metric(
        "Breusch-Pagan p-value",
        f"{bp_pvalue:.8f}"
    )

    if bp_pvalue > 0.05:
        st.success(
            "Fail to Reject H₀: No significant evidence "
            "of heteroscedasticity."
        )
    else:
        st.warning(
            "Reject H₀: Evidence of heteroscedasticity exists."
        )

    # -----------------------------------------------------
    # VIF
    # -----------------------------------------------------
    st.subheader("Multicollinearity - VIF")

    continuous_predictors = [
        "age",
        "bmi",
        "children"
    ]

    vif_data = pd.DataFrame()

    vif_data["Variable"] = continuous_predictors

    vif_data["VIF"] = [
        variance_inflation_factor(
            df[continuous_predictors].values,
            i
        )
        for i in range(len(continuous_predictors))
    ]

    st.dataframe(
        vif_data.round(4),
        use_container_width=True
    )

    st.caption(
        "Generally, VIF values below 5 indicate low "
        "multicollinearity."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "Statistics Lab 04 | Medical Insurance Cost Analysis | "
    "Streamlit + Statsmodels"
)