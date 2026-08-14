import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# App initialization
st.set_page_config(
    page_title="Varanasi Strategic Intelligence Advisory",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Core styling
st.markdown("""
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    h1, h2, h3, h4 { font-family: 'Segoe UI', Roboto, Helvetica, sans-serif; font-weight: 600; }
    .verdict-box {
        padding: 18px 24px;
        border-radius: 6px;
        margin-bottom: 20px;
        border: 1px solid #CBD5E1;
    }
    .verdict-win { background-color: #F0FDF4; border-left: 6px solid #16A34A; }
    .verdict-loss { background-color: #FEF2F2; border-left: 6px solid #DC2626; }
    .verdict-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
    .verdict-subtitle { font-size: 0.95rem; color: #475569; }
    .chart-caption { font-size: 0.85rem; color: #64748B; margin-bottom: 8px; font-style: italic; }
    </style>
""", unsafe_allow_html=True)

# Load resources
@st.cache_resource
def load_assets(data_dir):
    df_2024 = pd.read_csv(os.path.join(data_dir, "Varanasi_Election_2024_Actuals.csv"))
    df_2019 = pd.read_csv(os.path.join(data_dir, "Varanasi_Election_2019_Backcast.csv"))
    model_payload = joblib.load(os.path.join(data_dir, "trained_election_models.joblib"))
    return df_2024, df_2019, model_payload

# Set path to local directory
DATA_PATH = "./"

try:
    df_2024, df_2019, model_payload = load_assets(DATA_PATH)
    turnout_model = model_payload['turnout_model']
    party_models = model_payload['party_models']
    train_features = model_payload['train_features']
    modeled_parties = [p for p in model_payload['all_parties'] if p != 'Other']
except Exception as e:
    st.error(f"Asset Ingestion Error: {e}")
    st.stop()

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

SCALE_FACTOR = 5.15

# Sidebar parameters
st.sidebar.title("Strategic Navigation")
target_party = st.sidebar.selectbox("Select Target / Client Party", modeled_parties)

panel_selection = st.sidebar.radio(
    "Analytical Modules",
    ["Panel 1: Strategic Baseline & Demographics", "Panel 2: 2029 Predictive Scenario Engine"]
)
st.sidebar.markdown("---")

# Baseline calculations
total_electors_base = int(df_2024['Total_Electors_2024'].sum())
total_votes_base = int(df_2024['Total Votes'].sum() * SCALE_FACTOR)
constituency_turnout_base = (total_votes_base / total_electors_base) * 100 if total_electors_base > 0 else 0

opponent_parties = [p for p in modeled_parties if p != target_party]

target_votes_base = int(df_2024[target_party].sum() * SCALE_FACTOR)
target_share_base = (df_2024[target_party].sum() / df_2024['Total Votes'].sum()) * 100

opp_sums = {p: int(df_2024[p].sum() * SCALE_FACTOR) for p in opponent_parties}
lead_opponent = max(opp_sums, key=opp_sums.get)
lead_opp_votes_base = opp_sums[lead_opponent]
lead_opp_share_base = (df_2024[lead_opponent].sum() / df_2024['Total Votes'].sum()) * 100

net_margin_base = target_votes_base - lead_opp_votes_base

df_2024['Base_Winner'] = df_2024[modeled_parties].idxmax(axis=1)
booths_won_base = int((df_2024['Base_Winner'] == target_party).sum())
total_booths = len(df_2024)

df_2024['Max_Opponent_Votes'] = df_2024[opponent_parties].max(axis=1)
df_2024['Margin_Votes'] = df_2024[target_party] - df_2024['Max_Opponent_Votes']
df_2024['Margin_Pct'] = (df_2024['Margin_Votes'] / df_2024['Total Votes']) * 100

df_2024['Booth_Category'] = np.where(
    df_2024[f'{target_party}_Vote_Share'] >= 0.50, 'Stronghold (>50%)',
    np.where(
        df_2024['Margin_Pct'] >= 0,
        np.where(df_2024['Margin_Pct'] <= 5.0, 'Lean Positive (Vulnerable)', 'Safe Positive'),
        np.where(df_2024['Margin_Pct'] >= -5.0, 'Lean Negative (Targetable)', 'Weakness / Opponent Safe')
    )
)

# ---------------------------------------------------------
# Panel 1 Output
# ---------------------------------------------------------
if panel_selection == "Panel 1: Strategic Baseline & Demographics":
    is_winning_base = net_margin_base > 0
    swing_needed_base = abs(net_margin_base) / total_votes_base * 100 if total_votes_base > 0 else 0

    if is_winning_base:
        st.markdown(f"""
        <div class="verdict-box verdict-win">
            <div class="verdict-title" style="color: #15803D;">2024 BASELINE STATUS: VICTORY ({target_party})</div>
            <div class="verdict-subtitle">
                Lead Margin: <b>{net_margin_base:,} votes</b> (+{abs(target_share_base - lead_opp_share_base):.2f}% over {lead_opponent}) |
                Booths Secured: <b>{booths_won_base} / {total_booths}</b> ({(booths_won_base/total_booths)*100:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-box verdict-loss">
            <div class="verdict-title" style="color: #B91C1C;">2024 BASELINE STATUS: DEFEAT ({target_party})</div>
            <div class="verdict-subtitle">
                Deficit: <b>{abs(net_margin_base):,} votes</b> behind {lead_opponent} ({lead_opp_share_base:.2f}% vs {target_share_base:.2f}%) |
                Swing Required to Win: <b>+{swing_needed_base:.2f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader(f"Baseline Assessment: {target_party} in Varanasi (2024 Actuals)")
    st.markdown("---")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Electorate", f"{total_electors_base:,}")
    k2.metric("Total Votes Polled", f"{total_votes_base:,}")
    k3.metric("Constituency Turnout", f"{constituency_turnout_base:.2f}%")
    k4.metric(f"{target_party} Vote Share", f"{target_share_base:.2f}%")
    k5.metric(f"Net Margin vs {lead_opponent}", f"{net_margin_base:+,} votes")

    st.markdown("---")
    st.subheader("1. Assembly Segment Performance Matrix")

    seg_stats = df_2024.groupby('Assembly_Segment').agg(
        Booths=('Booth_No', 'count'),
        Electors=('Total_Electors_2024', 'sum'),
        Votes_Polled=('Total Votes', lambda x: int(x.sum() * SCALE_FACTOR)),
        Target_Votes=(target_party, lambda x: int(x.sum() * SCALE_FACTOR)),
        Opponent_Votes=(lead_opponent, lambda x: int(x.sum() * SCALE_FACTOR)),
        Margin_Votes=('Margin_Votes', lambda x: int(x.sum() * SCALE_FACTOR))
    ).reset_index()

    seg_stats['Turnout (%)'] = (seg_stats['Votes_Polled'] / seg_stats['Electors']) * 100
    seg_stats[f'{target_party} (%)'] = seg_stats.apply(lambda row: (row['Target_Votes'] / row['Votes_Polled']) * 100 if row['Votes_Polled'] > 0 else 0, axis=1)
    seg_stats[f'{lead_opponent} (%)'] = seg_stats.apply(lambda row: (row['Opponent_Votes'] / row['Votes_Polled']) * 100 if row['Votes_Polled'] > 0 else 0, axis=1)

    col_t1, col_t2 = st.columns([1, 1])
    with col_t1:
        st.markdown('<div class="chart-caption">Compares vote shares of the target party vs leading opponent across each assembly zone.</div>', unsafe_allow_html=True)
        fig_seg_bar = px.bar(
            seg_stats, x='Assembly_Segment', y=[f'{target_party} (%)', f'{lead_opponent} (%)'],
            barmode='group', title=f"Vote Share Comparison ({target_party} vs {lead_opponent})",
            color_discrete_sequence=['#2563EB', '#EA580C']
        )
        st.plotly_chart(fig_seg_bar, use_container_width=True)

    with col_t2:
        st.markdown('<div class="chart-caption">Displays absolute net vote margin (positive or negative) per assembly constituency.</div>', unsafe_allow_html=True)
        fig_margin_bar = px.bar(
            seg_stats, x='Assembly_Segment', y='Margin_Votes',
            title=f"Net Vote Margin by Assembly Segment",
            color='Margin_Votes', color_continuous_scale=['#DC2626', '#E2E8F0', '#16A34A']
        )
        st.plotly_chart(fig_margin_bar, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Constituency Demographics & Caste Spread")

    col_d1, col_d2 = st.columns([1, 1])
    caste_cols = [c for c in df_2024.columns if c.startswith('Caste_') and c.endswith('_2024')]
    caste_summary = df_2024[caste_cols].mean().reset_index()
    caste_summary.columns = ['Demographic_Group', 'Proportion']
    caste_summary['Group'] = caste_summary['Demographic_Group'].str.replace('Caste_', '').str.replace('_mean_2024', '')
    caste_summary['Share (%)'] = caste_summary['Proportion'] * 100

    with col_d1:
        st.markdown('<div class="chart-caption">Breakdown of demographic and caste proportions across the entire voter roll.</div>', unsafe_allow_html=True)
        fig_caste = px.pie(
            caste_summary, values='Share (%)', names='Group', hole=0.45,
            title="Electorate Caste Breakdown", color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig_caste, use_container_width=True)

    with col_d2:
        seg_caste = df_2024.groupby('Assembly_Segment')[caste_cols].mean().reset_index()
        seg_caste.columns = [c.replace('Caste_', '').replace('_mean_2024', '') for c in seg_caste.columns]
        seg_caste_melted = seg_caste.melt(id_vars='Assembly_Segment', var_name='Community', value_name='Weight')
        seg_caste_melted['Percentage'] = seg_caste_melted['Weight'] * 100

        st.markdown('<div class="chart-caption">Heatmap showing concentration weights of each community per assembly zone.</div>', unsafe_allow_html=True)
        fig_heat = px.density_heatmap(
            seg_caste_melted, x='Assembly_Segment', y='Community', z='Percentage',
            color_continuous_scale='Blues', title="Community Density Matrix (% per Segment)"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")
    st.subheader("3. Booth Classification & Strategy Targeting")

    f_col1, f_col2 = st.columns(2)
    with f_col1:
        sel_ac = st.selectbox("Assembly Segment Filter", ["All"] + list(df_2024['Assembly_Segment'].unique()))
    with f_col2:
        sel_cat = st.selectbox("Booth Category Filter", ["All"] + list(df_2024['Booth_Category'].unique()))

    df_view = df_2024.copy()
    if sel_ac != "All": df_view = df_view[df_view['Assembly_Segment'] == sel_ac]
    if sel_cat != "All": df_view = df_view[df_view['Booth_Category'] == sel_cat]

    display_cols = ['Assembly_Segment', 'Booth_No', 'Total Votes', 'Turnout_Percentage', f'{target_party}_Vote_Share', 'Margin_Pct', 'Booth_Category']
    st.dataframe(df_view[display_cols].sort_values(by='Margin_Pct', ascending=True), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Panel 2 Output
# ---------------------------------------------------------
elif panel_selection == "Panel 2: 2029 Predictive Scenario Engine":
    st.sidebar.subheader("1. Macro Turnout Adjustments")
    sim_turnout_delta = st.sidebar.slider("Uniform Turnout Shift (%)", -15.0, 15.0, 0.0, 0.5) / 100.0

    st.sidebar.subheader("2. Differential Demographic Mobilization")
    yadav_factor = st.sidebar.slider("OBC Yadav Mobilization", 0.60, 1.40, 1.0, 0.05)
    patel_factor = st.sidebar.slider("OBC Patel/Kurmi Mobilization", 0.60, 1.40, 1.0, 0.05)
    muslim_factor = st.sidebar.slider("Muslim Mobilization", 0.60, 1.40, 1.0, 0.05)
    brahmin_factor = st.sidebar.slider("Brahmin Mobilization", 0.60, 1.40, 1.0, 0.05)
    dalit_factor = st.sidebar.slider("SC Dalit Mobilization", 0.60, 1.40, 1.0, 0.05)

    st.sidebar.subheader("3. Direct Partisan Swings")
    target_swing = st.sidebar.slider(f"{target_party} Swing (%)", -15.0, 15.0, 0.0, 0.5) / 100.0
    opp_swing = st.sidebar.slider(f"{lead_opponent} Swing (%)", -15.0, 15.0, 0.0, 0.5) / 100.0

    X_sim = df_2024[train_features].copy()
    for col in X_sim.columns:
        if 'OBC_Yadav' in col: X_sim[col] = (X_sim[col] * yadav_factor).clip(0, 1)
        elif 'OBC_Patel' in col: X_sim[col] = (X_sim[col] * patel_factor).clip(0, 1)
        elif 'Muslim' in col: X_sim[col] = (X_sim[col] * muslim_factor).clip(0, 1)
        elif 'Brahmin' in col: X_sim[col] = (X_sim[col] * brahmin_factor).clip(0, 1)
        elif 'SC_Dalit' in col: X_sim[col] = (X_sim[col] * dalit_factor).clip(0, 1)

    c_cols = [c for c in X_sim.columns if 'Caste_' in c]
    X_sim[c_cols] = X_sim[c_cols].div(X_sim[c_cols].sum(axis=1), axis=0)

    sim_turnout = turnout_model.predict(X_sim) + sim_turnout_delta
    sim_turnout = np.clip(sim_turnout, 0.05, 0.95)

    raw_preds = np.zeros((len(X_sim), len(model_payload['all_parties'])))
    for idx, party in enumerate(model_payload['all_parties']):
        pred = party_models[party].predict(X_sim)
        if party == target_party: pred += target_swing
        elif party == lead_opponent: pred += opp_swing
        raw_preds[:, idx] = pred

    sim_shares = softmax(raw_preds)

    df_sim = df_2024[['Assembly_Segment', 'Booth_No', 'Total_Electors_2024']].copy()
    df_sim['Sim_Total_Votes'] = (df_sim['Total_Electors_2024'] * sim_turnout * SCALE_FACTOR).round().astype(int)

    for idx, party in enumerate(model_payload['all_parties']):
        p_name = model_payload['all_parties'][idx]
        if p_name in modeled_parties:
            df_sim[f'Sim_{p_name}_Share'] = sim_shares[:, idx]

    share_cols_sim = [f'Sim_{p}_Share' for p in modeled_parties]
    sum_shares = df_sim[share_cols_sim].sum(axis=1)
    for col in share_cols_sim:
        df_sim[col] = df_sim[col] / sum_shares

    for party in modeled_parties:
        df_sim[f'Sim_{party}_Votes'] = (df_sim['Sim_Total_Votes'] * df_sim[f'Sim_{party}_Share']).round().astype(int)

    total_sim_polled = df_sim['Sim_Total_Votes'].sum()
    sim_target_votes = df_sim[f'Sim_{target_party}_Votes'].sum()
    sim_target_share = (sim_target_votes / total_sim_polled) * 100 if total_sim_polled > 0 else 0

    sim_opp_votes = df_sim[f'Sim_{lead_opponent}_Votes'].sum()
    sim_opp_share = (sim_opp_votes / total_sim_polled) * 100 if total_sim_polled > 0 else 0
    sim_net_margin = sim_target_votes - sim_opp_votes

    sim_party_cols = [f'Sim_{p}_Votes' for p in modeled_parties]
    df_sim['Sim_Winner'] = df_sim[sim_party_cols].idxmax(axis=1).str.replace('Sim_', '').str.replace('_Votes', '')
    sim_booths_won = int((df_sim['Sim_Winner'] == target_party).sum())

    is_winning_sim = sim_net_margin > 0
    if is_winning_sim:
        st.markdown(f"""
        <div class="verdict-box verdict-win">
            <div class="verdict-title" style="color: #15803D;">2029 PROJECTION: VICTORY PROJECTED ({target_party})</div>
            <div class="verdict-subtitle">
                Projected Margin: <b>+{sim_net_margin:,} votes</b> over {lead_opponent} |
                Projected Vote Share: <b>{sim_target_share:.2f}%</b> |
                Booths Won: <b>{sim_booths_won} / {total_booths}</b> ({(sim_booths_won/total_booths)*100:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-box verdict-loss">
            <div class="verdict-title" style="color: #B91C1C;">2029 PROJECTION: DEFEAT PROJECTED ({target_party})</div>
            <div class="verdict-subtitle">
                Projected Deficit: <b>{sim_net_margin:,} votes</b> behind {lead_opponent} |
                Projected Vote Share: <b>{sim_target_share:.2f}%</b> |
                Booths Won: <b>{sim_booths_won} / {total_booths}</b> ({(sim_booths_won/total_booths)*100:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader(f"2029 Lok Sabha Predictive Simulation: {target_party}")
    st.markdown("---")

    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("2029 Projected Polled Votes", f"{total_sim_polled:,}", f"{total_sim_polled - total_votes_base:+,} vs 2024")
    fc2.metric(f"Projected {target_party} Share", f"{sim_target_share:.2f}%", f"{sim_target_share - target_share_base:+,.2f}%")
    fc3.metric(f"Projected Margin vs {lead_opponent}", f"{sim_net_margin:+,} votes", f"{sim_net_margin - net_margin_base:+,}")
    fc4.metric("Simulated Booths Won", f"{sim_booths_won} / {total_booths}", f"{sim_booths_won - booths_won_base:+} booths")

    st.markdown("---")

    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        st.markdown('<div class="chart-caption">Compares actual 2024 constituency vote shares against the projected 2029 simulation outcomes.</div>', unsafe_allow_html=True)
        comp_summary = pd.DataFrame({
            'Party': [target_party, lead_opponent],
            'Baseline 2024 (%)': [target_share_base, lead_opp_share_base],
            'Projected 2029 Scenario (%)': [sim_target_share, sim_opp_share]
        })
        fig_comp = go.Figure(data=[
            go.Bar(name='Baseline 2024', x=comp_summary['Party'], y=comp_summary['Baseline 2024 (%)'], marker_color='#94A3B8'),
            go.Bar(name='Projected 2029 Scenario', x=comp_summary['Party'], y=comp_summary['Projected 2029 Scenario (%)'], marker_color='#2563EB')
        ])
        fig_comp.update_layout(barmode='group', yaxis_title="Vote Share (%)")
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_sc2:
        st.markdown('<div class="chart-caption">Shows projected vote margins across all assembly segments for the 2029 election.</div>', unsafe_allow_html=True)
        seg_sim = df_sim.groupby('Assembly_Segment')[[f'Sim_{target_party}_Votes', f'Sim_{lead_opponent}_Votes']].sum().reset_index()
        seg_sim['Sim_Margin'] = seg_sim[f'Sim_{target_party}_Votes'] - seg_sim[f'Sim_{lead_opponent}_Votes']

        fig_sim_seg = px.bar(
            seg_sim, x='Assembly_Segment', y='Sim_Margin',
            title=f"Projected 2029 Net Margin ({target_party} vs {lead_opponent})",
            color='Sim_Margin', color_continuous_scale=['#DC2626', '#E2E8F0', '#16A34A']
        )
        st.plotly_chart(fig_sim_seg, use_container_width=True)

    st.markdown("---")
    st.subheader(f"Booth Flipping & Targetability Matrix ({target_party})")

    df_sim['Base_Winner'] = df_2024['Base_Winner']
    df_sim['Target_Gain'] = (df_sim['Base_Winner'] != target_party) & (df_sim['Sim_Winner'] == target_party)
    df_sim['Target_Loss'] = (df_sim['Base_Winner'] == target_party) & (df_sim['Sim_Winner'] != target_party)

    col_fl1, col_fl2 = st.columns(2)
    with col_fl1:
        st.markdown(f"**Booths Gained by {target_party} (`{df_sim['Target_Gain'].sum()}` booths)**")
        st.dataframe(df_sim[df_sim['Target_Gain']][['Assembly_Segment', 'Booth_No', 'Base_Winner', 'Sim_Winner']], hide_index=True, use_container_width=True)
    with col_fl2:
        st.markdown(f"**Booths Lost by {target_party} (`{df_sim['Target_Loss'].sum()}` booths)**")
        st.dataframe(df_sim[df_sim['Target_Loss']][['Assembly_Segment', 'Booth_No', 'Base_Winner', 'Sim_Winner']], hide_index=True, use_container_width=True)
