import streamlit as st
import openai
import os
import pandas as pd
import re
from io import StringIO
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load rate card and site list
excel_path = "adzymic_rate_card.xlsx"
site_list_path = "APX Sitelist - Regional.xlsx"
impact_df = pd.read_excel(excel_path, sheet_name='APX - Impact ')
reach_df = pd.read_excel(excel_path, sheet_name='APX - Reach')
site_xls = pd.ExcelFile(site_list_path)

impact_clean = impact_df.iloc[1:].copy()
impact_clean.columns = impact_df.iloc[0]
impact_clean = impact_clean.rename(columns={impact_clean.columns[0]: "Format"})
supported_markets = ['SG', 'MY', 'TH', 'ID']
cpm_table = impact_clean[["Format"] + supported_markets].dropna()

st.title("🧠 AI Media Plan Generator")

with st.form("media_form"):
    brand_name = st.text_input("Brand Name")
    budget = st.number_input("Budget (in USD)", min_value=1000, value=50000, step=1000)
    campaign_period = st.text_input("Campaign Period", value="May–July 2025")
    objective = st.selectbox("Objective", ["Brand Awareness", "Engagement", "Performance", "Conversions"])
    market = st.selectbox("Market", supported_markets)
    planning_mode = st.radio("Planning Mode", ["Automated Format Selection", "Manual Format Selection"])
    selected_formats = []
    if planning_mode == "Manual Format Selection":
        available_formats = sorted(set(impact_clean["Format"].dropna().str.title().tolist() + ["Reach Media"]))
        selected_formats = st.multiselect("Select formats to include", available_formats)
    submitted = st.form_submit_button("Generate Media Plan")

if submitted and (planning_mode == "Automated Format Selection" or selected_formats):
    with st.spinner("Generating your plan..."):
        market_cpm_impact = cpm_table[["Format", market]].dropna()
        market_cpm_impact_dict = dict(zip(market_cpm_impact["Format"].str.lower(), market_cpm_impact[market]))

        reach_cpm_row = reach_df[reach_df['Market'].str.strip().str.upper() == market.upper()]
        if not reach_cpm_row.empty:
            if budget > 50000:
                reach_cpm_value = reach_cpm_row.iloc[0][">50K"]
            elif budget > 25000:
                reach_cpm_value = reach_cpm_row.iloc[0][">25K"]
            else:
                reach_cpm_value = reach_cpm_row.iloc[0][">10K"]
            market_cpm_impact_dict["reach media"] = reach_cpm_value

        format_filter = f"\nOnly include the following formats in the plan: {', '.join(selected_formats)}." if planning_mode == "Manual Format Selection" else ""
        rate_card_string = "\n".join([f"- {fmt.title()}: ${cpm:.2f} CPM" for fmt, cpm in market_cpm_impact_dict.items()])
        site_df = site_xls.parse(market)
        site_df.columns = site_df.columns.str.strip().str.lower()
        site_map = site_df.groupby("format")["site"].apply(lambda x: ", ".join(x.astype(str))).to_dict() if "format" in site_df.columns and "site" in site_df.columns else {}

        site_list_string = "\n".join([f"- {fmt.title()}: {sites}" for fmt, sites in site_map.items()])
        site_block = f"\nRecommended Sites for Formats:{site_list_string}" if site_map else ""

        system_prompt = f"""
                You are a senior media planner. Based on the campaign brief and the following rate card for the {market} market, generate two alternative media plans that include the following:
                - Suggested formats: Reach media, APX Impact, DOOH, CTV
                - Budget allocation per format
                - Estimated impressions or reach (using provided CPMs)
                - Recommended site placements based on format
                - Rationale or short explanation for choices
                {format_filter}

                Each version should be clearly labeled (e.g., Option 1, Option 2).

                Rate Card:
                {rate_card_string}

                {site_block}
                """

        user_prompt = f"""
                Campaign Brief:
                - Brand: {brand_name}
                - Budget: ${budget}
                - Campaign Period: {campaign_period}
                - Objective: {objective}
                - Market: {market}

                Please present each plan in a clean markdown table format (| format | budget | cpm | reach | recommended sites | notes |), followed by a brief rationale.
                """

    try:
        client = openai.Client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        result_text = response.choices[0].message.content
        st.markdown(result_text)

        lines = result_text.strip().splitlines()
        all_tables, table = [], []
        for line in lines:
            if line.strip().startswith('|') and '---' not in line:
                table.append(line)
            elif table:
                all_tables.append("\n".join(table))
                table = []
        if table:
            all_tables.append("\n".join(table))

        seen = set()
        unique_tables = []
        for t in all_tables:
            if t not in seen:
                unique_tables.append(t)
                seen.add(t)

        site_df = site_xls.parse(market)
        site_df.columns = site_df.columns.str.strip().str.lower()
        site_map = site_df.groupby("format")["site"].apply(lambda x: ", ".join(x.astype(str))).to_dict() if "format" in site_df.columns and "site" in site_df.columns else {}

        for i, markdown_table in enumerate(unique_tables):
            try:
                df = pd.read_csv(StringIO(markdown_table), sep="|", engine="python")
                df = df.dropna(axis=1, how="all")
                df.columns = [col.strip().lower() for col in df.columns]
                if "format" in df.columns and "budget" in df.columns:
                    df["format"] = df["format"].str.lower()
                    df["cpm"] = df["format"].map(market_cpm_impact_dict)
                    df["impressions"] = (df["budget"] / df["cpm"] * 1000).round().astype('Int64')
                    df["reach"] = df["impressions"].astype(str) + " est"
                    df["recommended_sites"] = df["format"].str.title().map(site_map).fillna("-")

                if set(["format", "budget", "cpm", "reach"]).issubset(df.columns):
                    display_cols = ["format", "budget", "cpm", "reach", "recommended_sites"] + [col for col in df.columns if col not in ["format", "budget", "cpm", "reach", "recommended_sites"]]
                    df = df[display_cols]


                st.download_button(
                    f"Download Option {i+1} as CSV",
                    df.to_csv(index=False).encode('utf-8', 'ignore'),
                    f"media_plan_option_{i+1}.csv",
                    "text/csv"
                )
            except Exception as e:
                st.warning(f"Couldn't parse Option {i+1} to CSV: {e}")
    except Exception as e:
        st.error(f"Error generating media plan: {e}")
