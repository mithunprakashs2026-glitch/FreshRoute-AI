import streamlit as st
import pandas as pd
import plotly.express as px
from dataclasses import dataclass
from typing import Dict, List, Tuple

st.set_page_config(page_title="FreshRoute AI", page_icon="🥬", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
.hero {padding: 1.25rem 1.5rem; border-radius: 16px; background: linear-gradient(120deg,#123b32,#1e6b4f); color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; color: white;}
.hero p {margin: .35rem 0 0 0; opacity: .9;}
.small-note {font-size: .85rem; color: #777;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>🥬 FreshRoute AI</h1><p>Perishable produce allocation • explainable dispatch • what-if replanning</p></div>', unsafe_allow_html=True)
st.caption("Prototype using simulated data. Results are estimates from the assumptions shown here—not real-world performance claims.")

# -------------------- Simulation data --------------------
PRODUCE = [
    ("Tomatoes", 180, 3.0, 0.45),
    ("Bananas", 150, 2.4, 0.35),
    ("Spinach", 90, 4.0, 0.55),
    ("Mangoes", 120, 5.0, 0.65),
    ("Potatoes", 220, 1.8, 0.25),
    ("Strawberries", 70, 7.0, 0.80),
]

DESTINATIONS = [
    ("City Market", 180, 0.30, 0.0),
    ("Campus Canteens", 130, 0.22, 0.0),
    ("Retail Hub", 160, 0.25, 0.0),
    ("Rural Outlet", 100, 0.15, 0.0),
]

ROUTES = {
    "City Market": {"hours": 3, "cost_per_unit": 0.28, "capacity": 180},
    "Campus Canteens": {"hours": 2, "cost_per_unit": 0.20, "capacity": 130},
    "Retail Hub": {"hours": 5, "cost_per_unit": 0.34, "capacity": 160},
    "Rural Outlet": {"hours": 7, "cost_per_unit": 0.42, "capacity": 100},
}

def make_data(demand_multiplier: float, delay_hours: int, capacity_multiplier: float):
    batches = []
    for i, (name, qty, price, unit_cost) in enumerate(PRODUCE):
        shelf = [2, 4, 1, 5, 8, 2][i]
        batches.append({
            "batch_id": f"B{i+1:02d}", "commodity": name, "quantity": qty,
            "shelf_life_h": shelf, "market_price": price, "unit_cost": unit_cost
        })
    demand = {}
    for dest, qty, factor, _ in DESTINATIONS:
        for name, base_qty, _, _ in PRODUCE:
            demand[(dest, name)] = max(0, int(base_qty * factor * demand_multiplier))
    routes = {d: {"hours": v["hours"] + delay_hours,
                  "cost_per_unit": v["cost_per_unit"],
                  "capacity": int(v["capacity"] * capacity_multiplier)} for d, v in ROUTES.items()}
    return batches, demand, routes

def allocate_fifo(batches, demand, routes):
    """Baseline: allocate in fixed destination order; use earliest-expiring batches first."""
    remaining_demand = dict(demand)
    remaining_capacity = {d: r["capacity"] for d, r in routes.items()}
    allocations = []
    # Fixed order is intentionally simple, but still respects route shelf-life feasibility and capacity.
    for batch in sorted(batches, key=lambda b: b["shelf_life_h"]):
        qty_left = batch["quantity"]
        for dest in ["City Market", "Campus Canteens", "Retail Hub", "Rural Outlet"]:
            key = (dest, batch["commodity"])
            feasible = batch["shelf_life_h"] >= routes[dest]["hours"]
            if not feasible or qty_left <= 0:
                continue
            take = min(qty_left, remaining_demand.get(key, 0), remaining_capacity[dest])
            if take > 0:
                allocations.append({"batch_id": batch["batch_id"], "commodity": batch["commodity"],
                    "destination": dest, "quantity": take, "arrival_h": routes[dest]["hours"],
                    "shelf_life_h": batch["shelf_life_h"], "market_price": batch["market_price"],
                    "unit_cost": batch["unit_cost"], "transport_cost": routes[dest]["cost_per_unit"],
                    "reason": "FIFO: earliest-expiring feasible stock, fixed destination order"})
                qty_left -= take
                remaining_demand[key] -= take
                remaining_capacity[dest] -= take
    return allocations


def allocate_freshroute(batches, demand, routes):
    """Optimize feasible allocation using mixed-integer linear programming."""
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import csr_matrix

    candidates = []
    for b in batches:
        for dest, route in routes.items():
            key = (dest, b["commodity"])
            if b["shelf_life_h"] < route["hours"]:
                continue
            if demand.get(key, 0) <= 0 or route["capacity"] <= 0:
                continue

            net = b["market_price"] - b["unit_cost"] - route["cost_per_unit"]
            candidates.append((b, dest, route, net))

    if not candidates:
        return []

    n = len(candidates)
    constraint_rows = []
    limits = []

    # Batch inventory limits
    for b in batches:
        constraint_rows.append([
            1 if c[0]["batch_id"] == b["batch_id"] else 0
            for c in candidates
        ])
        limits.append(b["quantity"])

    # Destination capacity limits
    for dest, route in routes.items():
        constraint_rows.append([
            1 if c[1] == dest else 0 for c in candidates
        ])
        limits.append(route["capacity"])

    # Demand limits for each destination and commodity
    for key, max_demand in demand.items():
        constraint_rows.append([
            1 if (c[1], c[0]["commodity"]) == key else 0
            for c in candidates
        ])
        limits.append(max_demand)

    result = milp(
        c=np.array([-c[3] for c in candidates]),
        integrality=np.ones(n),
        bounds=Bounds(
            np.zeros(n),
            np.array([c[0]["quantity"] for c in candidates])
        ),
        constraints=LinearConstraint(
            csr_matrix(np.array(constraint_rows, dtype=float)),
            -np.inf,
            np.array(limits, dtype=float)
        ),
        options={"time_limit": 10}
    )

    if not result.success or result.x is None:
        raise RuntimeError(f"MILP optimization failed: {result.message}")

    allocations = []
    for i, value in enumerate(result.x):
        quantity = int(round(value))
        if quantity <= 0:
            continue

        b, dest, route, net = candidates[i]
        allocations.append({
            "batch_id": b["batch_id"],
            "commodity": b["commodity"],
            "destination": dest,
            "quantity": quantity,
            "arrival_h": route["hours"],
            "shelf_life_h": b["shelf_life_h"],
            "market_price": b["market_price"],
            "unit_cost": b["unit_cost"],
            "transport_cost": route["cost_per_unit"],
            "reason": (
                f"MILP allocation; estimated net/unit ₹{net:.2f}; "
                f"{b['shelf_life_h'] - route['hours']}h shelf-life buffer"
            )
        })

    return allocations

def summarize(allocations, batches):
    alloc = pd.DataFrame(allocations)
    total_stock = sum(b["quantity"] for b in batches)
    total_purchase_cost = float(sum(b["quantity"] * b["unit_cost"] for b in batches))
    if alloc.empty:
        return {"alloc": alloc, "stock": total_stock, "shipped": 0, "spoilage": total_stock,
                "fulfillment": 0.0, "revenue": 0.0, "transport": 0.0,
                "purchase": total_purchase_cost, "net": -total_purchase_cost, "discounted": 0}
    alloc["gross_revenue"] = alloc["quantity"] * alloc["market_price"]
    alloc["purchase_cost"] = alloc["quantity"] * alloc["unit_cost"]
    alloc["transport_total"] = alloc["quantity"] * alloc["transport_cost"]
    shipped = int(alloc["quantity"].sum())
    # Revenue assumes sold units receive listed market price; unsold stock is counted as spoilage.
    revenue = float(alloc["gross_revenue"].sum())
    transport = float(alloc["transport_total"].sum())
    purchase = total_purchase_cost
    # Fulfilment denominator is total generated demand, not inventory.
    return {"alloc": alloc, "stock": total_stock, "shipped": shipped,
            "spoilage": max(0, total_stock - shipped), "revenue": revenue,
            "transport": transport, "purchase": purchase,
            "net": revenue - transport - purchase,
            "discounted": 0}


def recommend_discounts(batches, demand, routes):
    recommendations = []

    for batch in batches:
        commodity = batch["commodity"]
        stock = batch["quantity"]
        shelf_life = batch["shelf_life_h"]
        price = batch["market_price"]

        total_demand = sum(
            demand.get((destination, commodity), 0)
            for destination in routes
        )

        # Estimate the portion of stock that may remain unsold.
        at_risk = max(0, stock - total_demand)

        if at_risk <= 0:
            continue

        # Transparent rule: shorter remaining shelf life means
        # a stronger discount recommendation.
        if shelf_life <= 2:
            discount_rate = 0.20
        elif shelf_life <= 4:
            discount_rate = 0.10
        else:
            discount_rate = 0.05

        recommendations.append({
            "Commodity": commodity,
            "At-risk units (estimated)": at_risk,
            "Shelf life (hours)": shelf_life,
            "Market price (₹/unit)": price,
            "Suggested discount (%)": discount_rate * 100,
            "Discounted price (₹/unit)": round(
                price * (1 - discount_rate), 2
            ),
            "Potential revenue if all at-risk units sell (₹)": round(
                at_risk * price * (1 - discount_rate), 2
            ),
            "Reason": (
                f"{shelf_life} hours of shelf life remaining; "
                f"{discount_rate:.0%} rule-based discount"
            ),
        })

    return pd.DataFrame(recommendations)


def get_demand_total(demand):
    return sum(demand.values())

# -------------------- Controls --------------------
with st.sidebar:
    st.header("Scenario controls")
    demand_multiplier = st.slider("Demand level", 0.5, 1.5, 1.0, 0.1)
    delay_hours = st.slider("Transport delay (hours)", 0, 4, 0, 1)
    capacity_multiplier = st.slider("Transport capacity", 0.5, 1.2, 1.0, 0.1)
    st.markdown("---")
    st.markdown("**Scenario inputs**")
    st.write(f"Demand multiplier: **{demand_multiplier:.1f}×**")
    st.write(f"Added route delay: **{delay_hours} h**")
    st.write(f"Capacity multiplier: **{capacity_multiplier:.1f}×**")
    run = st.button("🔄 Recalculate plans", type="primary", use_container_width=True)

# Streamlit reruns on every control change; button provides an explicit user action cue.
batches, demand, routes = make_data(demand_multiplier, delay_hours, capacity_multiplier)
fifo = summarize(allocate_fifo(batches, demand, routes), batches)
smart = summarize(allocate_freshroute(batches, demand, routes), batches)
demand_total = get_demand_total(demand)
for result in (fifo, smart):
    result["fulfillment"] = (result["shipped"] / demand_total * 100) if demand_total else 0
    result["spoilage_pct"] = result["spoilage"] / result["stock"] * 100 if result["stock"] else 0

# -------------------- Results --------------------
st.subheader("Decision scorecard")
cols = st.columns(4)
cols[0].metric("FreshRoute net value", f"₹{smart['net']:,.0f}",
               delta=f"₹{smart['net'] - fifo['net']:,.0f} vs baseline")
cols[1].metric("Stock shipped", f"{smart['shipped']:,} units",
               delta=f"{smart['shipped'] - fifo['shipped']:+,} vs baseline")
cols[2].metric("Estimated unsold/spoiled", f"{smart['spoilage']:,} units",
               delta=f"{smart['spoilage'] - fifo['spoilage']:+,} vs baseline", delta_color="inverse")
cols[3].metric("Demand fulfilled", f"{smart['fulfillment']:.1f}%",
               delta=f"{smart['fulfillment'] - fifo['fulfillment']:+.1f} pp vs baseline")

st.markdown("### Baseline vs FreshRoute")
compare = pd.DataFrame([
    {"Strategy":"FIFO baseline", "Net value (₹)":fifo["net"], "Shipped units":fifo["shipped"], "Spoilage (units)":fifo["spoilage"], "Demand fulfilled (%)":fifo["fulfillment"]},
    {"Strategy":"FreshRoute", "Net value (₹)":smart["net"], "Shipped units":smart["shipped"], "Spoilage (units)":smart["spoilage"], "Demand fulfilled (%)":smart["fulfillment"]},
])
left, right = st.columns([1.15, 1])
with left:
    metric = st.selectbox("Chart metric", ["Net value (₹)", "Shipped units", "Spoilage (units)", "Demand fulfilled (%)"])
    fig = px.bar(compare, x="Strategy", y=metric, color="Strategy", text_auto=".1f", title=f"{metric}: same scenario, two strategies")
    fig.update_layout(showlegend=False, margin=dict(l=10,r=10,t=55,b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Compare the underlying financial totals"):
        financials = pd.DataFrame([
            {"Strategy": "FIFO baseline", "Sales revenue (₹)": fifo["revenue"],
             "Transport cost (₹)": fifo["transport"], "Starting inventory cost (₹)": fifo["purchase"],
             "Estimated net value (₹)": fifo["net"]},
            {"Strategy": "FreshRoute MILP", "Sales revenue (₹)": smart["revenue"],
             "Transport cost (₹)": smart["transport"], "Starting inventory cost (₹)": smart["purchase"],
             "Estimated net value (₹)": smart["net"]},
        ])
        st.dataframe(financials.round(2), use_container_width=True, hide_index=True)
        st.caption("Net value = shipped-unit sales revenue − transport cost on shipped units − purchase cost of all starting inventory.")

with right:
    st.markdown("**How to read this**")
    st.write("Both strategies receive the same simulated inventory, demand, route times, and capacities. The baseline uses fixed destination order; FreshRoute optimizes feasible allocations using estimated contribution and shelf-life constraints.")
    st.info("FreshRoute uses a mixed-integer linear programming (MILP) optimizer to maximize estimated contribution under the modeled constraints. Results are simulated estimates; discount-driven demand changes and spoilage penalties are not directly optimized.")
    st.markdown("**Constraint checks**")
    checks = [
        ("Destination capacity", all(smart["alloc"].groupby("destination")["quantity"].sum().get(d, 0) <= r["capacity"] for d, r in routes.items()) if not smart["alloc"].empty else True),
        ("Arrival within shelf life", bool((smart["alloc"]["arrival_h"] <= smart["alloc"]["shelf_life_h"]).all()) if not smart["alloc"].empty else True),
        ("No destination over-demand", all(smart["alloc"].groupby(["destination", "commodity"])["quantity"].sum().get(k, 0) <= v for k, v in demand.items()) if not smart["alloc"].empty else True),
    ]
    for name, ok in checks:
        st.write(("✅" if ok else "❌") + " " + name)


st.markdown("### FreshRoute dispatch plan")
if not smart["alloc"].empty:
    display = smart["alloc"][["batch_id","commodity","destination","quantity","arrival_h","shelf_life_h","reason"]].copy()
    display.columns = ["Batch","Produce","Destination","Units","Travel (h)","Shelf life (h)","Decision explanation"]
    st.dataframe(display, use_container_width=True, hide_index=True)
else:
    st.warning("No feasible allocations in this scenario. Reduce transport delays or increase capacity.")


st.markdown("### 🏷️ Smart discount impact simulator")

discount_df = recommend_discounts(batches, demand, routes)

if discount_df.empty:
    st.success("No excess stock flagged under current demand assumptions.")
else:
    sell_through = st.slider(
        "Assumed share of at-risk units sold after discount",
        min_value=10,
        max_value=100,
        value=60,
        step=10,
    ) / 100

    discount_df["Expected units sold"] = (
        discount_df["At-risk units (estimated)"] * sell_through
    ).round().astype(int)

    discount_df["Expected units still at risk"] = (
        discount_df["At-risk units (estimated)"]
        - discount_df["Expected units sold"]
    )

    discount_df["Estimated discounted revenue (₹)"] = (
        discount_df["Expected units sold"]
        * discount_df["Discounted price (₹/unit)"]
    ).round(2)

    st.dataframe(
        discount_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These are simulated estimates. The selected sell-through "
        "rate is an assumption, not a demand prediction."
    )
st.markdown("### Inventory and route assumptions")
inv_df = pd.DataFrame(batches)
st.dataframe(inv_df.rename(columns={"batch_id":"Batch","commodity":"Produce","quantity":"Available units","shelf_life_h":"Remaining shelf life (h)","market_price":"Market price/unit (₹)","unit_cost":"Purchase cost/unit (₹)"}), use_container_width=True, hide_index=True)
route_df = pd.DataFrame([{"Destination":d,"Travel hours":v["hours"],"Capacity (units)":v["capacity"],"Transport cost/unit (₹)":v["cost_per_unit"]} for d,v in routes.items()])
st.dataframe(route_df, use_container_width=True, hide_index=True)

csv = smart["alloc"].to_csv(index=False).encode("utf-8") if not smart["alloc"].empty else b""
st.download_button("⬇️ Download FreshRoute dispatch plan (CSV)", data=csv, file_name="freshroute_dispatch_plan.csv", mime="text/csv", disabled=not bool(csv))

with st.expander("Methodology & limitations"):
    st.markdown("""
    - **Data:** generated inside this app; no real distributor dataset or live feed is used.
    - **Baseline:** earliest-expiring feasible batches are considered in a fixed destination order.
    - **FreshRoute optimizer:** mixed-integer linear programming (MILP) maximizes estimated contribution per shipped unit, subject to inventory, route capacity, demand, and shelf-life feasibility constraints.
    - **Feasibility:** respects simulated destination capacity, destination-commodity demand, and arrival-before-expiry.
    - **Net value:** shipped-unit sales revenue minus transport cost on shipped units and purchase cost of all starting inventory. It does not include labor, cold storage, disposal, or discount effects.
    - **Spoilage estimate:** all unallocated inventory is counted as unsold/spoiled for this simplified one-period scenario.
    - **Not production-ready:** validate with representative real data and compare against a documented baseline before making operational claims.
    """)
