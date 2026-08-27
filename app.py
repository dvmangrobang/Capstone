import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Capstone Dashboard", layout="wide")
st.title("Sales Dashboard")

# ── Data loaders ──────────────────────────────────────────────
@st.cache_data
def load_weekly():
    return pd.read_csv("data/df_weekly.csv", parse_dates=["Week_Start"])

@st.cache_data
def load_predictions():
    df = pd.read_csv("data/df_predictions.csv", parse_dates=["Week_Start"])
    df = df.rename(columns={"Item (2)": "Item"})
    return df

@st.cache_data
def load_heatmap():
    return pd.read_csv("data/df_heatmap.csv")

@st.cache_data
def load_bundles():
    df = pd.read_csv("data/df_bundles.csv")
    df = df.rename(columns={
        "Product Bundles": "Product_Bundle",
        "Confidence (Likelihood)": "Confidence",
    })
    return df

df_weekly = load_weekly()
df_predictions = load_predictions()
df_heatmap = load_heatmap()
df_bundles = load_bundles()


# ── Helper: bridge actual/forecast lines so they connect visually ──
def add_bridge_point(data, group_col, sort_col):
    """Duplicates the last 'Actual' row into the 'Forecast' group so the
    line connects continuously instead of showing a gap at the transition."""
    data = data.sort_values(sort_col)
    actual = data[data[group_col] == "Actual"]
    forecast = data[data[group_col] == "Forecast"]
    if not actual.empty and not forecast.empty:
        bridge = actual.iloc[[-1]].copy()
        bridge[group_col] = "Forecast"
        forecast = pd.concat([bridge, forecast], ignore_index=True)
    return pd.concat([actual, forecast], ignore_index=True)


# ── Layout: 2x2 grid ──────────────────────────────────────────
top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)

# ── Top left: Sales vs. Time ───────────────────────────────────
with top_left:
    st.subheader("Sales vs. Time")
    view = st.selectbox("View", ["Monthly", "Weekly"], key="sales_view")

    if view == "Monthly":
        monthly_df = df_weekly.copy()
        monthly_df["month"] = monthly_df["Week_Start"].dt.to_period("M")
        monthly_df["row_type"] = monthly_df["Event"].apply(
            lambda x: "Forecast" if x == "Forecast" else "Actual"
        )

        # if any week in a month is a forecast, treat the whole month as Forecast
        # (avoids splitting one month into two separate line points)
        month_type = (
            monthly_df.groupby("month")["row_type"]
                      .apply(lambda s: "Forecast" if (s == "Forecast").any() else "Actual")
        )

        monthly = (
            monthly_df.groupby("month")["Total_Quantity_Sold"]
                      .sum()
                      .reset_index()
        )
        monthly["Type"] = monthly["month"].map(month_type)
        monthly["month_label"] = monthly["month"].dt.strftime("%b %Y")
        monthly = add_bridge_point(monthly, "Type", "month")

        fig1 = px.line(
            monthly, x="month_label", y="Total_Quantity_Sold", color="Type",
            markers=True,
            color_discrete_map={"Actual": "#1f77b4", "Forecast": "#ff7f0e"},
            labels={"month_label": "Month", "Total_Quantity_Sold": "Total Sales"},
        )
        fig1.update_xaxes(type="category")

    else:  # Weekly
        available_months = (
            df_weekly["Week_Start"].dt.to_period("M")
            .drop_duplicates().sort_values().astype(str)
        )
        selected_month = st.selectbox("Select month", available_months, key="sales_month")

        month_df = df_weekly[
            df_weekly["Week_Start"].dt.to_period("M").astype(str) == selected_month
        ].copy()
        month_df["week_label"] = month_df["Week_Start"].dt.strftime("Week of %b %d")
        month_df["Type"] = month_df["Event"].apply(
            lambda x: "Forecast" if x == "Forecast" else "Actual"
        )
        month_df = add_bridge_point(month_df, "Type", "Week_Start")

        fig1 = px.line(
            month_df, x="week_label", y="Total_Quantity_Sold", color="Type",
            markers=True,
            color_discrete_map={"Actual": "#1f77b4", "Forecast": "#ff7f0e"},
            labels={"week_label": "Week", "Total_Quantity_Sold": "Total Sales"},
        )
        fig1.update_xaxes(type="category")

    st.plotly_chart(fig1, width='stretch')

# ── Top right: Top Items by Sales ──────────────────────────────
with top_right:
    st.subheader("Top Items by Sales")
    bar_view = st.selectbox("View", ["Monthly Totals", "Top 5 Weekly"], key="topitems_view")

    if bar_view == "Monthly Totals":
        pred_month = df_predictions.copy()
        pred_month["month"] = pred_month["Week_Start"].dt.to_period("M")

        available_months2 = (
            pred_month["month"].drop_duplicates().sort_values().astype(str)
        )
        selected_month2 = st.selectbox("Select month", available_months2, key="topitems_month")

        month_data = pred_month[pred_month["month"].astype(str) == selected_month2].copy()
        month_data["Type"] = month_data["Event"].apply(
            lambda x: "Forecast" if x == "Forecast" else "Actual"
        )
        monthly_by_item = (
            month_data.groupby(["Item", "Type"])["Quantity"]
                      .sum()
                      .reset_index()
                      .sort_values("Quantity", ascending=False)
        )

        fig2 = px.bar(
            monthly_by_item, x="Item", y="Quantity", color="Type",
            labels={"Item": "Product", "Quantity": "Sales"},
            color_discrete_map={"Actual": "#1f77b4", "Forecast": "#ff7f0e"},
            category_orders={"Type": ["Actual", "Forecast"]},
        )
        fig2.update_xaxes(categoryorder="total descending")

    else:  # Top 5 Weekly
        available_weeks = df_predictions["Week_Start"].drop_duplicates().sort_values()
        week_labels = available_weeks.dt.strftime("Week of %b %d, %Y")
        week_map = dict(zip(week_labels, available_weeks))

        selected_label = st.selectbox("Select week", week_labels, key="topitems_week")
        selected_week = week_map[selected_label]

        week_data = df_predictions[df_predictions["Week_Start"] == selected_week].copy()
        week_data["Type"] = week_data["Event"].apply(
            lambda x: "Forecast" if x == "Forecast" else "Actual"
        )
        top5 = week_data.sort_values("Quantity", ascending=False).head(5)

        fig2 = px.bar(
            top5, x="Item", y="Quantity", color="Type",
            labels={"Item": "Product", "Quantity": "Sales"},
            color_discrete_map={"Actual": "#1f77b4", "Forecast": "#ff7f0e"},
            category_orders={"Type": ["Actual", "Forecast"]},
        )
        fig2.update_xaxes(categoryorder="total descending")

    st.plotly_chart(fig2, width='stretch')

# ── Bottom left: Sales Volume Heatmap ──────────────────────────
with bottom_left:
    st.subheader("Sales Volume Heatmap")

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    time_labels = ["10AM-12PM", "12PM-2PM", "2PM-4PM", "4PM-6PM", "6PM-8PM", "8PM-10PM"]

    df_heat = df_heatmap.copy()
    bins = [10, 12, 14, 16, 18, 20, 22]
    df_heat["time_block"] = pd.cut(df_heat["Hour"], bins=bins, labels=time_labels, right=False)

    pivot = df_heat.pivot_table(index="Day", columns="time_block", values="Quantity", aggfunc="sum")
    pivot = pivot.reindex(index=day_order, columns=time_labels)

    fig3 = px.imshow(
        pivot,
        labels=dict(x="Time Block", y="Day of Week", color="Avg Sales"),
        aspect="auto",
        color_continuous_scale="YlOrRd",
        text_auto=".0f",
    )
    st.plotly_chart(fig3, width='stretch')

# ── Bottom right: Product Bundle Confidence ─────────────────────
with bottom_right:
    st.subheader("Product Bundles")

    bundles = df_bundles.drop_duplicates(subset="Product_Bundle").copy()
    # Confidence may already contain a "%" as text (e.g. "9%") — strip it before converting
    bundles["Confidence"] = (
        bundles["Confidence"].astype(str).str.replace("%", "", regex=False).astype(float)
    )
    bundles = bundles.sort_values("Confidence", ascending=False)
    bundles["Confidence"] = bundles["Confidence"].astype(int).astype(str) + "%"

    display_df = bundles.rename(columns={"Product_Bundle": "Product Bundle"})

    st.dataframe(display_df, width='stretch', hide_index=True)