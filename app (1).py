import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re
import math

st.set_page_config(page_title="Electoral Analytics & Predictive Intelligence Platform", layout="wide")

st.title("Electoral Analytics & Predictive Intelligence Platform")
st.caption("Strategic Demographics, Spatial Intelligence, and Probabilistic Victory Modeling")

sns.set_theme(style="whitegrid")

def normal_cdf(x, mean, std):
    if std <= 0:
        return 1.0 if x >= mean else 0.0
    return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2))))

@st.cache_data
def load_electoral_data():
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    dfs = []

    for f in csv_files:
        try:
            t = pd.read_csv(f, dtype=str, engine='python', on_bad_lines='skip', encoding='utf-8-sig')
            t.columns = t.columns.str.strip().str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)

            years_found = re.findall(r'\d{4}', f)
            if years_found:
                t['YEAR'] = years_found[0]
            elif 'YEAR' in t.columns:
                t['YEAR'] = t['YEAR'].astype(str).str.strip()
            else:
                t['YEAR'] = 'Current'

            dfs.append(t)
        except Exception:
            continue

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    df['VOTER_COUNT'] = 1

    # Gender Standardization
    gender_col = next((c for c in df.columns if 'GEN' in c or 'SEX' in c), None)
    if gender_col:
        df['GENDER_CLEAN'] = df[gender_col].astype(str).str.strip().str.upper().str[0]
        df['GENDER_CLEAN'] = df['GENDER_CLEAN'].map({'M': 'Male', 'F': 'Female', 'T': 'Third Gender'}).fillna('Unknown')
    else:
        df['GENDER_CLEAN'] = 'Unknown'

    # Age Parsing
    age_col = next((c for c in df.columns if 'AGE' in c), None)
    if age_col:
        df['AGE_CLEAN'] = df[age_col].astype(str).str.extract(r'(\d+)')[0]
        df['AGE_CLEAN'] = pd.to_numeric(df['AGE_CLEAN'], errors='coerce').fillna(35)
    else:
        df['AGE_CLEAN'] = 35

    bins = [17, 25, 40, 60, 150]
    labels = ['Youth (18-25)', 'Young Adult (26-40)', 'Middle Age (41-60)', 'Senior (60+)']
    df['AGE_GROUP'] = pd.cut(df['AGE_CLEAN'], bins=bins, labels=labels).astype(str)

    # Categorical Standardizations
    for target in ['ASSEMBLY', 'CATEGORY', 'LOCALITY']:
        actual_col = next((c for c in df.columns if target in c), None)
        if actual_col:
            df[target] = df[actual_col].fillna('Unknown').astype(str).str.strip().str.title()
        else:
            df[target] = 'Unknown'

    return df

df = load_electoral_data()

if df is not None:
    st.sidebar.header("Filter Controls")

    # Assembly Segment Filter
    all_constituencies = ["All Segments"] + sorted([x for x in df['ASSEMBLY'].unique() if x != 'Unknown'])
    selected_constituency = st.sidebar.selectbox("Assembly Segment", all_constituencies)
    df_const = df[df['ASSEMBLY'] == selected_constituency] if selected_constituency != "All Segments" else df.copy()

    # Election Year Filter
    years = sorted([x for x in df_const["YEAR"].unique() if x != 'Current'])
    year_options = ["All Years"] + years if years else ["Current"]
    selected_year = st.sidebar.selectbox("Election Year Focus", year_options)

    if selected_year != "All Years":
        df_year = df_const[df_const["YEAR"] == selected_year]
    else:
        df_year = df_const

    # Metric Cards (Filtered strictly by selected year)
    total_voters = len(df_year)
    males = len(df_year[df_year["GENDER_CLEAN"] == 'Male'])
    females = len(df_year[df_year["GENDER_CLEAN"] == 'Female'])
    gender_ratio = round((females / males * 1000), 0) if males > 0 else 0
    female_share = round((females / total_voters * 100), 1) if total_voters > 0 else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sample Electorate", f"{total_voters:,}")
    m2.metric("Male Voters", f"{males:,}")
    m3.metric("Female Voters", f"{females:,}")
    m4.metric("Gender Ratio (F/1000M)", f"{gender_ratio:.0f}")
    m5.metric("Female Share", f"{female_share}%")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Multi-Year Dynamics",
        "Community & Category Analysis",
        "Comprehensive Gender & Age Demographics",
        "Probabilistic Victory Simulator & Candidate Engine"
    ])

    # TAB 1: Multi-Year Trends
    with tab1:
        st.subheader("Multi-Year Electorate Dynamics")
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.markdown("##### Total Voter Growth Across Election Years")
            if len(years) > 0:
                year_counts = df_const.groupby("YEAR")["VOTER_COUNT"].sum().reset_index()
                fig_y, ax_y = plt.subplots(figsize=(6, 3.5))
                sns.barplot(data=year_counts, x="YEAR", y="VOTER_COUNT", palette="Blues_d", ax=ax_y)
                ax_y.set_ylabel("Voters")
                st.pyplot(fig_y)
                plt.close(fig_y)

                st.markdown("**Yearly Voter Breakdown**")
                st.dataframe(year_counts.rename(columns={"YEAR": "Year", "VOTER_COUNT": "Total Voters"}), use_container_width=True)

        with col_t2:
            st.markdown("##### Community & Category Shift Across Years")
            if len(years) > 0 and not df_const[df_const['CATEGORY'] != 'Unknown'].empty:
                cat_trend = df_const[df_const['CATEGORY'] != 'Unknown'].groupby(["YEAR", "CATEGORY"])["VOTER_COUNT"].sum().reset_index()
                fig_ct, ax_ct = plt.subplots(figsize=(6, 3.5))
                sns.lineplot(data=cat_trend, x="YEAR", y="VOTER_COUNT", hue="CATEGORY", marker="o", ax=ax_ct)
                ax_ct.set_ylabel("Voters")
                st.pyplot(fig_ct)
                plt.close(fig_ct)

                st.markdown("**Category Dynamics Matrix**")
                cat_pivot = cat_trend.pivot(index='YEAR', columns='CATEGORY', values='VOTER_COUNT').fillna(0).astype(int)
                st.dataframe(cat_pivot, use_container_width=True)

    # TAB 2: Community & Category Analysis
    with tab2:
        st.subheader(f"Community and Category Breakdown ({selected_year})")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### Category Density in Top Localities")
            clean_loc = df_year[(df_year['LOCALITY'] != 'Unknown') & (df_year['CATEGORY'] != 'Unknown')]
            if not clean_loc.empty:
                top_locs = clean_loc['LOCALITY'].value_counts().head(10).index
                pivot_cat = pd.crosstab(clean_loc[clean_loc['LOCALITY'].isin(top_locs)]['LOCALITY'], clean_loc['CATEGORY'])
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.heatmap(pivot_cat, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
                ax.set_ylabel("Locality")
                st.pyplot(fig)
                plt.close(fig)

                st.markdown("**Locality vs Category Data Table**")
                st.dataframe(pivot_cat, use_container_width=True)

        with col_b:
            st.markdown("##### Category Share Distribution")
            clean_cat = df_year[df_year['CATEGORY'] != 'Unknown']
            if not clean_cat.empty:
                cat_df = clean_cat['CATEGORY'].value_counts().reset_index()
                cat_df.columns = ["Category", "Voters"]
                fig2, ax2 = plt.subplots(figsize=(7, 4))
                sns.barplot(data=cat_df, x="Category", y="Voters", palette="mako", ax=ax2)
                st.pyplot(fig2)
                plt.close(fig2)

                st.markdown("**Category Voter Summary**")
                st.dataframe(cat_df, use_container_width=True)

    # TAB 3: Comprehensive Gender & Age Demographics
    with tab3:
        st.subheader(f"Comprehensive Gender & Age Demographics ({selected_year})")

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.markdown("##### Age Distribution Across Electorate")
            age_dist = df_year["AGE_GROUP"].value_counts().reset_index()
            age_dist.columns = ["Age Group", "Voters"]
            if age_dist["Voters"].sum() > 0:
                fig3, ax3 = plt.subplots(figsize=(6, 3.5))
                order_list = ['Youth (18-25)', 'Young Adult (26-40)', 'Middle Age (41-60)', 'Senior (60+)']
                sns.barplot(data=age_dist, x="Age Group", y="Voters", palette="viridis", ax=ax3, order=order_list)
                plt.xticks(rotation=15, ha='right')
                st.pyplot(fig3)
                plt.close(fig3)

                st.markdown("**Age Group Breakdown**")
                st.dataframe(age_dist, use_container_width=True)

        with row1_col2:
            st.markdown("##### Gender Split by Age Group")
            clean_gender = df_year[df_year['GENDER_CLEAN'] != 'Unknown']
            if not clean_gender.empty:
                age_gender = pd.crosstab(clean_gender["AGE_GROUP"], clean_gender["GENDER_CLEAN"])
                valid_idx = [i for i in ['Youth (18-25)', 'Young Adult (26-40)', 'Middle Age (41-60)', 'Senior (60+)'] if i in age_gender.index]
                age_gender = age_gender.loc[valid_idx]
                fig4, ax4 = plt.subplots(figsize=(6, 3.5))
                age_gender.plot(kind="bar", stacked=True, ax=ax4, color=["#e377c2", "#1f77b4", "#2ca02c"])
                plt.xticks(rotation=15, ha='right')
                plt.ylabel("Voters")
                st.pyplot(fig4)
                plt.close(fig4)

                st.markdown("**Gender x Age Group Matrix**")
                st.dataframe(age_gender, use_container_width=True)

        st.markdown("---")
        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            st.markdown("##### Gender Ratio (Females per 1000 Males) by Age Group")
            if not clean_gender.empty and 'Male' in age_gender.columns and 'Female' in age_gender.columns:
                ratio_by_age = (age_gender['Female'] / age_gender['Male'] * 1000).round(0).reset_index()
                ratio_by_age.columns = ['Age Group', 'Gender Ratio']
                fig5, ax5 = plt.subplots(figsize=(6, 3.5))
                sns.barplot(data=ratio_by_age, x='Age Group', y='Gender Ratio', palette="magma", ax=ax5)
                ax5.axhline(1000, color='red', linestyle='--', label='Parity (1000)')
                plt.xticks(rotation=15, ha='right')
                plt.legend()
                st.pyplot(fig5)
                plt.close(fig5)

                st.markdown("**Age Group Gender Ratios**")
                st.dataframe(ratio_by_age, use_container_width=True)

        with row2_col2:
            st.markdown("##### Top 10 Localities by Female Electorate Share (%)")
            clean_gen_loc = df_year[(df_year['GENDER_CLEAN'] != 'Unknown') & (df_year['LOCALITY'] != 'Unknown')]
            if not clean_gen_loc.empty:
                loc_gender = pd.crosstab(clean_gen_loc["LOCALITY"], clean_gen_loc["GENDER_CLEAN"])
                loc_gender['Total'] = loc_gender.sum(axis=1)
                loc_gender = loc_gender[loc_gender['Total'] >= 50]
                if 'Female' in loc_gender.columns:
                    loc_gender['Female Share (%)'] = (loc_gender['Female'] / loc_gender['Total'] * 100).round(1)
                    top_female_locs = loc_gender.sort_values(by='Female Share (%)', ascending=False).head(10).reset_index()

                    fig6, ax6 = plt.subplots(figsize=(6, 3.5))
                    sns.barplot(data=top_female_locs, y='LOCALITY', x='Female Share (%)', palette="rocket", ax=ax6)
                    ax6.set_xlabel("Female Share (%)")
                    ax6.set_ylabel("Locality")
                    st.pyplot(fig6)
                    plt.close(fig6)

                    st.markdown("**Top Female Density Localities**")
                    st.dataframe(top_female_locs[['LOCALITY', 'Total', 'Female', 'Female Share (%)']], use_container_width=True)

    # TAB 4: Probabilistic Victory Simulator & Candidate Engine
    with tab4:
        st.subheader("Probabilistic Victory Simulator & Candidate Strategy Engine")
        st.markdown("Simulate electoral victory probability over a standardized **2 Lakh (200,000 voter)** constituency using Binomial / Normal distribution approximations.")

        target_categories = sorted([x for x in df_year["CATEGORY"].unique() if x != 'Unknown'])

        st.markdown("##### 1. Constituency & Candidate Profile Parameters")
        cand_col1, cand_col2, cand_col3, cand_col4 = st.columns(4)
        with cand_col1:
            constituency_scale = st.number_input("Constituency Scale (Voters)", min_value=50000, max_value=1000000, value=200000, step=10000)
        with cand_col2:
            cand_gender = st.selectbox("Candidate Gender", ["Female", "Male", "Other"])
        with cand_col3:
            cand_age_group = st.selectbox("Candidate Age Bracket", ["Middle Age (41-60)", "Young Adult (26-40)", "Senior (60+)", "Youth (18-25)"])
        with cand_col4:
            cand_locality_status = st.selectbox("Candidate Standing", ["Local Resident", "External / Non-Local"])

        st.markdown("##### 2. Turnout & Support Baseline Models")
        turn_col1, turn_col2, turn_col3 = st.columns(3)
        with turn_col1:
            expected_turnout = st.slider("Polling Day Turnout (%)", min_value=30, max_value=90, value=65, step=1)
        with turn_col2:
            target_support_base = st.slider("Target Coalition Support Rate (%)", min_value=20, max_value=95, value=70, step=1)
        with turn_col3:
            non_target_support_base = st.slider("Baseline Non-Target Support Rate (%)", min_value=0, max_value=40, value=15, step=1)

        st.markdown("##### 3. Coalition Target Selection")
        coal_col1, coal_col2 = st.columns([2, 1])
        with coal_col1:
            selected_targets = st.multiselect("Select Target Coalition Communities", target_categories, default=target_categories[:2] if len(target_categories) > 1 else target_categories)
            filter_youth = st.checkbox("Focus Target Coalition on Youth Voters (18-25)", value=False)
            filter_female = st.checkbox("Focus Target Coalition on Female Voters", value=False)

        # Dynamic Candidate Synergy Calculations
        synergy_boost = 0.0
        synergy_reasons = []

        if cand_gender == "Female" and filter_female:
            synergy_boost += 6.0
            synergy_reasons.append("+6.0% Female Candidate Alignment with Female Voters")
        elif cand_gender == "Female":
            synergy_boost += 3.0
            synergy_reasons.append("+3.0% Organic Female Candidate Voter Preference")

        if cand_age_group in ["Youth (18-25)", "Young Adult (26-40)"] and filter_youth:
            synergy_boost += 5.0
            synergy_reasons.append("+5.0% Youth Candidate Alignment with Young Voters")

        if cand_locality_status == "Local Resident":
            synergy_boost += 3.0
            synergy_reasons.append("+3.0% Local Roots Advantage")
        else:
            synergy_boost -= 2.0
            synergy_reasons.append("-2.0% External Candidate Penalty")

        effective_target_support = min(0.95, max(0.05, (target_support_base + synergy_boost) / 100.0))
        effective_non_target_support = non_target_support_base / 100.0
        turnout_rate = expected_turnout / 100.0

        # Sub-Demographic Mathematical Modeling
        sim_df = df_year[df_year["CATEGORY"].isin(selected_targets)]
        if filter_youth:
            sim_df = sim_df[sim_df["AGE_GROUP"] == 'Youth (18-25)']
        if filter_female:
            sim_df = sim_df[sim_df["GENDER_CLEAN"] == 'Female']

        target_sample_size = len(sim_df)
        total_sample_size = len(df_year) if len(df_year) > 0 else 1

        target_share_of_electorate = target_sample_size / total_sample_size

        # Scale to 200,000 Constituency
        scaled_target_pool = int(constituency_scale * target_share_of_electorate)
        scaled_non_target_pool = constituency_scale - scaled_target_pool

        # Expected Votes & Variance Calculations
        target_cast_votes = scaled_target_pool * turnout_rate
        non_target_cast_votes = scaled_non_target_pool * turnout_rate
        total_cast_votes = target_cast_votes + non_target_cast_votes

        expected_target_votes = target_cast_votes * effective_target_support
        expected_non_target_votes = non_target_cast_votes * effective_non_target_support
        expected_total_votes = expected_target_votes + expected_non_target_votes

        win_threshold = total_cast_votes * 0.45 # 45% plurality target

        # Binomial / Normal Distribution Uncertainty Modeling
        var_target = target_cast_votes * effective_target_support * (1 - effective_target_support)
        var_non_target = non_target_cast_votes * effective_non_target_support * (1 - effective_non_target_support)
        std_error = math.sqrt(var_target + var_non_target)

        expected_margin = expected_total_votes - win_threshold
        win_probability = normal_cdf(expected_total_votes, win_threshold, std_error) * 100

        # Approximated Margin Ranges (95% CI rounded to nearest 100)
        margin_lower = int(round((expected_margin - 1.96 * std_error) / 100.0) * 100)
        margin_upper = int(round((expected_margin + 1.96 * std_error) / 100.0) * 100)

        st.markdown("---")
        st.markdown("##### 4. Probabilistic Forecast & Win Margin Analysis")

        res_m1, res_m2, res_m3, res_m4 = st.columns(4)
        res_m1.metric("Target Share of Electorate", f"{target_share_of_electorate*100:.1f}% ({scaled_target_pool:,} voters)")
        res_m2.metric("Projected Total Cast Votes", f"{int(total_cast_votes):,}")
        res_m3.metric("Win Plurality Threshold (45%)", f"{int(win_threshold):,}")
        res_m4.metric("Win Probability", f"{win_probability:.1f}%")

        st.markdown("##### Approximated Outcome Range & Directives")

        if synergy_reasons:
            st.info("Candidate Synergy Impacts Applied: " + " | ".join(synergy_reasons))

        if margin_lower > 0:
            st.success(f"Victory Highly Probable: Estimated surplus of approximately {abs(margin_lower):,} to {abs(margin_upper):,} votes above the victory threshold.")
        elif margin_upper > 0:
            st.warning(f"Competitive Race (Margin of Error): Outcome ranges from a potential deficit of {abs(margin_lower):,} votes to a surplus of {abs(margin_upper):,} votes.")
        else:
            st.error(f"Deficit Warning: Projected deficit of approximately {abs(margin_upper):,} to {abs(margin_lower):,} votes below the required 45% mark.")

        st.markdown("##### Executive Directives")
        directives = []

        if cand_gender == "Female":
            directives.append("Female Candidate Directives: Mobilize female ward captains across high female density localities. Focus messaging on community welfare, household economics, and local safety.")

        if cand_locality_status == "External / Non-Local":
            directives.append("Non-Local Counter-Strategy: Appoint prominent local community leaders as co-campaign leads to neutralize non-local perception.")

        if filter_youth or cand_age_group in ["Youth (18-25)", "Young Adult (26-40)"]:
            directives.append("Youth Engagement Strategy: Focus digital campaigns on recruitment drives, educational access, and regional employment opportunities.")

        if expected_margin < 0:
            needed_support_increase = round((abs(expected_margin) / (target_cast_votes if target_cast_votes > 0 else 1)) * 100, 1)
            directives.append(f"Coalition Expansion Required: Expand target coalition categories or increase support capture rate by approximately {needed_support_increase}% to achieve victory.")

        for d in directives:
            st.write(f"- {d}")

        st.markdown("##### Targeted Demographic Base Matrix")
        sim_summary = sim_df.groupby(['CATEGORY', 'GENDER_CLEAN'])['VOTER_COUNT'].sum().unstack(fill_value=0)
        st.dataframe(sim_summary, use_container_width=True)

else:
    st.error("No CSV files found in directory. Place constituency CSV files in the project folder to proceed.")
