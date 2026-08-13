%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error

st.set_page_config(page_title="Electoral Intelligence Portal", layout="wide", initial_sidebar_state="expanded")
st.markdown("### Electoral Intelligence Portal")
st.sidebar.header("Parameters")

@st.cache_data
def load_default_data():
    records = [
        # --- 2024 Election Cycle ---
        {'Year': 2024, 'Segment': 'Rohaniya', 'Party': 'BJP', 'Electors': 430500, 'Votes_Secured': 127508, 'Campaign_Spend_Lakhs': 85, 'Rally_Count': 12, 'OBC_Share': 40.0, 'SC_Share': 19.0, 'Muslim_Share': 12.0, 'General_Share': 24.0, 'Male_Turnout': 58.5, 'Female_Turnout': 55.4, 'Booth_Count': 432},
        {'Year': 2024, 'Segment': 'Rohaniya', 'Party': 'INC', 'Electors': 430500, 'Votes_Secured': 101225, 'Campaign_Spend_Lakhs': 55, 'Rally_Count': 8, 'OBC_Share': 40.0, 'SC_Share': 19.0, 'Muslim_Share': 12.0, 'General_Share': 24.0, 'Male_Turnout': 58.5, 'Female_Turnout': 55.4, 'Booth_Count': 432},
        {'Year': 2024, 'Segment': 'Rohaniya', 'Party': 'BSP', 'Electors': 430500, 'Votes_Secured': 10527, 'Campaign_Spend_Lakhs': 15, 'Rally_Count': 2, 'OBC_Share': 40.0, 'SC_Share': 19.0, 'Muslim_Share': 12.0, 'General_Share': 24.0, 'Male_Turnout': 58.5, 'Female_Turnout': 55.4, 'Booth_Count': 432},

        {'Year': 2024, 'Segment': 'Varanasi North', 'Party': 'BJP', 'Electors': 442100, 'Votes_Secured': 131241, 'Campaign_Spend_Lakhs': 90, 'Rally_Count': 15, 'OBC_Share': 23.0, 'SC_Share': 14.0, 'Muslim_Share': 25.0, 'General_Share': 34.0, 'Male_Turnout': 55.2, 'Female_Turnout': 53.1, 'Booth_Count': 445},
        {'Year': 2024, 'Segment': 'Varanasi North', 'Party': 'INC', 'Electors': 442100, 'Votes_Secured': 101731, 'Campaign_Spend_Lakhs': 60, 'Rally_Count': 10, 'OBC_Share': 23.0, 'SC_Share': 14.0, 'Muslim_Share': 25.0, 'General_Share': 34.0, 'Male_Turnout': 55.2, 'Female_Turnout': 53.1, 'Booth_Count': 445},
        {'Year': 2024, 'Segment': 'Varanasi North', 'Party': 'BSP', 'Electors': 442100, 'Votes_Secured': 4173, 'Campaign_Spend_Lakhs': 12, 'Rally_Count': 1, 'OBC_Share': 23.0, 'SC_Share': 14.0, 'Muslim_Share': 25.0, 'General_Share': 34.0, 'Male_Turnout': 55.2, 'Female_Turnout': 53.1, 'Booth_Count': 445},

        {'Year': 2024, 'Segment': 'Varanasi South', 'Party': 'BJP', 'Electors': 355000, 'Votes_Secured': 97878, 'Campaign_Spend_Lakhs': 70, 'Rally_Count': 10, 'OBC_Share': 21.0, 'SC_Share': 12.0, 'Muslim_Share': 22.0, 'General_Share': 41.0, 'Male_Turnout': 52.1, 'Female_Turnout': 50.4, 'Booth_Count': 358},
        {'Year': 2024, 'Segment': 'Varanasi South', 'Party': 'INC', 'Electors': 355000, 'Votes_Secured': 81732, 'Campaign_Spend_Lakhs': 50, 'Rally_Count': 7, 'OBC_Share': 21.0, 'SC_Share': 12.0, 'Muslim_Share': 22.0, 'General_Share': 41.0, 'Male_Turnout': 52.1, 'Female_Turnout': 50.4, 'Booth_Count': 358},
        {'Year': 2024, 'Segment': 'Varanasi South', 'Party': 'BSP', 'Electors': 355000, 'Votes_Secured': 1032, 'Campaign_Spend_Lakhs': 5, 'Rally_Count': 0, 'OBC_Share': 21.0, 'SC_Share': 12.0, 'Muslim_Share': 22.0, 'General_Share': 41.0, 'Male_Turnout': 52.1, 'Female_Turnout': 50.4, 'Booth_Count': 358},

        {'Year': 2024, 'Segment': 'Varanasi Cantt.', 'Party': 'BJP', 'Electors': 440200, 'Votes_Secured': 145922, 'Campaign_Spend_Lakhs': 95, 'Rally_Count': 14, 'OBC_Share': 26.0, 'SC_Share': 14.0, 'Muslim_Share': 17.0, 'General_Share': 39.0, 'Male_Turnout': 55.8, 'Female_Turnout': 53.0, 'Booth_Count': 441},
        {'Year': 2024, 'Segment': 'Varanasi Cantt.', 'Party': 'INC', 'Electors': 440200, 'Votes_Secured': 95000, 'Campaign_Spend_Lakhs': 65, 'Rally_Count': 9, 'OBC_Share': 26.0, 'SC_Share': 14.0, 'Muslim_Share': 17.0, 'General_Share': 39.0, 'Male_Turnout': 55.8, 'Female_Turnout': 53.0, 'Booth_Count': 441},
        {'Year': 2024, 'Segment': 'Varanasi Cantt.', 'Party': 'BSP', 'Electors': 440200, 'Votes_Secured': 8500, 'Campaign_Spend_Lakhs': 10, 'Rally_Count': 1, 'OBC_Share': 26.0, 'SC_Share': 14.0, 'Muslim_Share': 17.0, 'General_Share': 39.0, 'Male_Turnout': 55.8, 'Female_Turnout': 53.0, 'Booth_Count': 441},

        {'Year': 2024, 'Segment': 'Sevapuri', 'Party': 'BJP', 'Electors': 361500, 'Votes_Secured': 108890, 'Campaign_Spend_Lakhs': 75, 'Rally_Count': 11, 'OBC_Share': 42.0, 'SC_Share': 22.0, 'Muslim_Share': 10.0, 'General_Share': 21.0, 'Male_Turnout': 61.2, 'Female_Turnout': 58.9, 'Booth_Count': 365},
        {'Year': 2024, 'Segment': 'Sevapuri', 'Party': 'INC', 'Electors': 361500, 'Votes_Secured': 80000, 'Campaign_Spend_Lakhs': 55, 'Rally_Count': 8, 'OBC_Share': 42.0, 'SC_Share': 22.0, 'Muslim_Share': 10.0, 'General_Share': 21.0, 'Male_Turnout': 61.2, 'Female_Turnout': 58.9, 'Booth_Count': 365},
        {'Year': 2024, 'Segment': 'Sevapuri', 'Party': 'BSP', 'Electors': 361500, 'Votes_Secured': 9500, 'Campaign_Spend_Lakhs': 15, 'Rally_Count': 2, 'OBC_Share': 42.0, 'SC_Share': 22.0, 'Muslim_Share': 10.0, 'General_Share': 21.0, 'Male_Turnout': 61.2, 'Female_Turnout': 58.9, 'Booth_Count': 365},

        # --- 2019 Election Cycle ---
        {'Year': 2019, 'Segment': 'Rohaniya', 'Party': 'BJP', 'Electors': 405200, 'Votes_Secured': 135400, 'Campaign_Spend_Lakhs': 80, 'Rally_Count': 10, 'OBC_Share': 40.0, 'SC_Share': 19.0, 'Muslim_Share': 12.0, 'General_Share': 24.0, 'Male_Turnout': 58.0, 'Female_Turnout': 54.0, 'Booth_Count': 408},
        {'Year': 2019, 'Segment': 'Rohaniya', 'Party': 'INC', 'Electors': 405200, 'Votes_Secured': 50000, 'Campaign_Spend_Lakhs': 40, 'Rally_Count': 5, 'OBC_Share': 40.0, 'SC_Share': 19.0, 'Muslim_Share': 12.0, 'General_Share': 24.0, 'Male_Turnout': 58.0, 'Female_Turnout': 54.0, 'Booth_Count': 408},
        {'Year': 2019, 'Segment': 'Rohaniya', 'Party': 'BSP', 'Electors': 405200, 'Votes_Secured': 60000, 'Campaign_Spend_Lakhs': 45, 'Rally_Count': 6, 'OBC_Share': 40.0, 'SC_Share': 19.0, 'Muslim_Share': 12.0, 'General_Share': 24.0, 'Male_Turnout': 58.0, 'Female_Turnout': 54.0, 'Booth_Count': 408},

        {'Year': 2019, 'Segment': 'Varanasi North', 'Party': 'BJP', 'Electors': 415000, 'Votes_Secured': 141200, 'Campaign_Spend_Lakhs': 85, 'Rally_Count': 12, 'OBC_Share': 23.0, 'SC_Share': 14.0, 'Muslim_Share': 25.0, 'General_Share': 34.0, 'Male_Turnout': 54.5, 'Female_Turnout': 51.5, 'Booth_Count': 419},
        {'Year': 2019, 'Segment': 'Varanasi North', 'Party': 'INC', 'Electors': 415000, 'Votes_Secured': 65000, 'Campaign_Spend_Lakhs': 50, 'Rally_Count': 7, 'OBC_Share': 23.0, 'SC_Share': 14.0, 'Muslim_Share': 25.0, 'General_Share': 34.0, 'Male_Turnout': 54.5, 'Female_Turnout': 51.5, 'Booth_Count': 419},
        {'Year': 2019, 'Segment': 'Varanasi North', 'Party': 'BSP', 'Electors': 415000, 'Votes_Secured': 55000, 'Campaign_Spend_Lakhs': 40, 'Rally_Count': 5, 'OBC_Share': 23.0, 'SC_Share': 14.0, 'Muslim_Share': 25.0, 'General_Share': 34.0, 'Male_Turnout': 54.5, 'Female_Turnout': 51.5, 'Booth_Count': 419},

        # --- 2014 Election Cycle ---
        {'Year': 2014, 'Segment': 'Rohaniya', 'Party': 'BJP', 'Electors': 380000, 'Votes_Secured': 120000, 'Campaign_Spend_Lakhs': 75, 'Rally_Count': 9, 'OBC_Share': 40.0, 'SC_Share': 19.0, 'Muslim_Share': 12.0, 'General_Share': 24.0, 'Male_Turnout': 55.0, 'Female_Turnout': 50.0, 'Booth_Count': 390},
        {'Year': 2014, 'Segment': 'Rohaniya', 'Party': 'INC', 'Electors': 380000, 'Votes_Secured': 45000, 'Campaign_Spend_Lakhs': 35, 'Rally_Count': 4, 'OBC_Share': 40.0, 'SC_Share': 19.0, 'Muslim_Share': 12.0, 'General_Share': 24.0, 'Male_Turnout': 55.0, 'Female_Turnout': 50.0, 'Booth_Count': 390},
        {'Year': 2014, 'Segment': 'Rohaniya', 'Party': 'BSP', 'Electors': 380000, 'Votes_Secured': 52000, 'Campaign_Spend_Lakhs': 40, 'Rally_Count': 5, 'OBC_Share': 40.0, 'SC_Share': 19.0, 'Muslim_Share': 12.0, 'General_Share': 24.0, 'Male_Turnout': 55.0, 'Female_Turnout': 50.0, 'Booth_Count': 390},

        {'Year': 2014, 'Segment': 'Varanasi North', 'Party': 'BJP', 'Electors': 390000, 'Votes_Secured': 125000, 'Campaign_Spend_Lakhs': 78, 'Rally_Count': 10, 'OBC_Share': 23.0, 'SC_Share': 14.0, 'Muslim_Share': 25.0, 'General_Share': 34.0, 'Male_Turnout': 52.0, 'Female_Turnout': 48.0, 'Booth_Count': 400},
        {'Year': 2014, 'Segment': 'Varanasi North', 'Party': 'INC', 'Electors': 390000, 'Votes_Secured': 48000, 'Campaign_Spend_Lakhs': 38, 'Rally_Count': 5, 'OBC_Share': 23.0, 'SC_Share': 14.0, 'Muslim_Share': 25.0, 'General_Share': 34.0, 'Male_Turnout': 52.0, 'Female_Turnout': 48.0, 'Booth_Count': 400},
        {'Year': 2014, 'Segment': 'Varanasi North', 'Party': 'BSP', 'Electors': 390000, 'Votes_Secured': 49000, 'Campaign_Spend_Lakhs': 38, 'Rally_Count': 5, 'OBC_Share': 23.0, 'SC_Share': 14.0, 'Muslim_Share': 25.0, 'General_Share': 34.0, 'Male_Turnout': 52.0, 'Female_Turnout': 48.0, 'Booth_Count': 400}
    ]
    return pd.DataFrame(records)

df_master = load_default_data()

analytical_view = st.sidebar.radio("Module", [
    "Multi-Year Demographic & Vote Analysis",
    "War Room Strategy & Simulation"
])

@st.cache_resource
def train_predictive_pipeline(data):
    train_df = data.copy()
    
    categorical_features = ['Party', 'Segment']
    numeric_features = ['Year', 'Electors', 'Campaign_Spend_Lakhs', 'Rally_Count', 'OBC_Share', 'SC_Share', 'Muslim_Share', 'General_Share', 'Female_Turnout', 'Booth_Count']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1.0))
    ])
    
    X = train_df[categorical_features + numeric_features]
    y = train_df['Votes_Secured']
    
    pipeline.fit(X, y)
    return pipeline

model_pipeline = train_predictive_pipeline(df_master)

if analytical_view == "Multi-Year Demographic & Vote Analysis":
    st.subheader("Historical Multi-Year Comparison & Demographic Breakdown")

    target_segment = st.sidebar.selectbox("Select Target Segment", df_master['Segment'].unique())
    selected_history_year = st.sidebar.selectbox("Select Historical Election Year", sorted(df_master['Year'].unique(), reverse=True))

    segment_data = df_master[df_master['Segment'] == target_segment]
    year_filtered_data = segment_data[segment_data['Year'] == selected_history_year]

    col_metric1, col_metric2, col_metric3 = st.columns(3)
    total_electors = segment_data['Electors'].iloc[0]
    booths = segment_data['Booth_Count'].iloc[0]
    voters_per_booth = int(total_electors / booths) if booths > 0 else 0

    col_metric1.metric("Total Electorate Base", f"{total_electors:,}")
    col_metric2.metric("Allocated Polling Booths", f"{booths}")
    col_metric3.metric("Voters Per Booth Density", f"{voters_per_booth:,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"#### Party Performance in {selected_history_year} ({target_segment})")
        if not year_filtered_data.empty:
            st.dataframe(year_filtered_data[['Party', 'Votes_Secured', 'Campaign_Spend_Lakhs', 'Rally_Count']], hide_index=True, use_container_width=True)

            bar_fig = px.bar(year_filtered_data, x='Party', y='Votes_Secured', color='Party', title=f"Votes Secured by Party ({selected_history_year})")
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.warning("No record found for this specific year and segment combination.")

    with col_right:
        st.markdown(f"#### Caste and Community Distribution ({target_segment})")
        demo_df = segment_data[['OBC_Share', 'SC_Share', 'Muslim_Share', 'General_Share']].drop_duplicates()
        if not demo_df.empty:
            demo_melted = demo_df.melt(var_name='Community', value_name='Share (%)')
            color_map = {'OBC_Share': '#F59E0B', 'SC_Share': '#3B82F6', 'Muslim_Share': '#10B981', 'General_Share': '#6366F1'}

            pie_fig = px.pie(demo_melted, values='Share (%)', names='Community', color='Community', color_discrete_map=color_map, hole=0.4)
            pie_fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Multi-Year Trend Analysis Across Cycles (2014 - 2024)")
    trend_fig = px.line(segment_data, x='Year', y='Votes_Secured', color='Party', markers=True, title=f"Vote Trajectory Across Years in {target_segment}")
    st.plotly_chart(trend_fig, use_container_width=True)

elif analytical_view == "War Room Strategy & Simulation":
    st.subheader("Advanced Political Consultant War Room")

    forecast_year = st.sidebar.selectbox("Target Forecast Year", [2029, 2034, 2039])
    consulting_party = st.sidebar.selectbox("Select Client Party", df_master['Party'].unique())
    target_segment = st.sidebar.selectbox("Select Target Segment", df_master['Segment'].unique())

    st.markdown(f"**Advising Portfolio: {consulting_party} | Target Region: {target_segment} | Projection Cycle: {forecast_year}**")

    st.markdown("#### Strategic Caste & Community Mobilization Focus Checkboxes")
    col_cb1, col_cb2, col_cb3, col_cb4 = st.columns(4)
    focus_obc = col_cb1.checkbox("Focus OBC Block", value=True)
    focus_sc = col_cb2.checkbox("Focus SC Block", value=False)
    focus_muslim = col_cb3.checkbox("Focus Muslim Block", value=False)
    focus_general = col_cb4.checkbox("Focus General Block", value=False)

    st.markdown("---")
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        new_spend = st.slider("Campaign Budget Allocation (Lakhs)", min_value=10, max_value=250, value=85)
    with col_sim2:
        new_rallies = st.slider("Targeted Ground Rallies", min_value=1, max_value=120, value=12)
    with col_sim3:
        female_focus_boost = st.slider("Women Voter Outreach Intensity (%)", min_value=1, max_value=50, value=15)

    if st.button("Execute War Room Simulation"):
        sim_records = []
        seg_data_recent = df_master[(df_master['Segment'] == target_segment) & (df_master['Year'] == 2024)]
        growth_multiplier = 1.05 if forecast_year == 2029 else (1.10 if forecast_year == 2034 else 1.15)

        for _, row in seg_data_recent.iterrows():
            rec = row.to_dict()
            party_name = row['Party']
            rec['Year'] = forecast_year
            rec['Electors'] = int(row['Electors'] * growth_multiplier)

            if party_name == consulting_party:
                rec['Campaign_Spend_Lakhs'] = new_spend
                rec['Rally_Count'] = new_rallies
                rec['Female_Turnout'] = min(75.0, row['Female_Turnout'] + (female_focus_boost * 0.2))

            # Prepare feature input DataFrame
            input_features = pd.DataFrame([{
                'Party': party_name,
                'Segment': target_segment,
                'Year': forecast_year,
                'Electors': rec['Electors'],
                'Campaign_Spend_Lakhs': rec['Campaign_Spend_Lakhs'],
                'Rally_Count': rec['Rally_Count'],
                'OBC_Share': rec['OBC_Share'],
                'SC_Share': rec['SC_Share'],
                'Muslim_Share': rec['Muslim_Share'],
                'General_Share': rec['General_Share'],
                'Female_Turnout': rec['Female_Turnout'],
                'Booth_Count': rec['Booth_Count']
            }])

            # Get raw ML prediction from pipeline
            raw_ml_prediction = model_pipeline.predict(input_features)[0]

            if party_name == consulting_party:
                caste_multiplier = 1.0
                if focus_obc: caste_multiplier += (row['OBC_Share'] / 100.0) * 0.06
                if focus_sc: caste_multiplier += (row['SC_Share'] / 100.0) * 0.06
                if focus_muslim: caste_multiplier += (row['Muslim_Share'] / 100.0) * 0.06
                if focus_general: caste_multiplier += (row['General_Share'] / 100.0) * 0.06

                final_votes = raw_ml_prediction * caste_multiplier
                rec['Simulated_Votes'] = int(max(1000, final_votes))
            else:
                rec['Simulated_Votes'] = int(max(1000, raw_ml_prediction))

            sim_records.append(rec)

        df_sim = pd.DataFrame(sim_records)
        df_sim = df_sim.sort_values(by='Simulated_Votes', ascending=False)

        st.markdown("---")
        st.markdown(f"#### Simulated Electoral Outcome for {forecast_year}")
        display_columns = ['Party', 'Simulated_Votes', 'Electors', 'Campaign_Spend_Lakhs', 'Rally_Count', 'Female_Turnout', 'OBC_Share', 'SC_Share', 'Muslim_Share', 'General_Share']
        st.dataframe(df_sim[display_columns], hide_index=True, use_container_width=True)

        chart = alt.Chart(df_sim).mark_bar().encode(
            x=alt.X('Party:N', sort='-y'),
            y='Simulated_Votes:Q',
            color=alt.condition(alt.datum.Party == consulting_party, alt.value('#F59E0B'), alt.value('#374151'))
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
