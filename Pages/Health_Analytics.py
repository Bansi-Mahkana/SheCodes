st.subheader("🔮 What-If Scenario Analysis")

metric = st.selectbox(
    "Select Metric",
    ["respiratory_cases", "heat_related_cases"]
)

impact = st.selectbox(
    "Scenario Type",
    ["Increase", "Decrease"]
)

percent = st.slider("Impact (%)", 0, 50, 10)

multiplier = 1 + percent/100 if impact == "Increase" else 1 - percent/100

filtered_df["simulated"] = filtered_df[metric] * multiplier

fig = px.line(
    filtered_df,
    x="date",
    y=[metric, "simulated"],
    title="What-If Impact Simulation"
)

st.plotly_chart(fig, use_container_width=True)
