"""Criteria of a planning ledger: cost, life-cycle CO2, metals, trenching, shifted energy.

A ledger holds physical quantities only (km of cable, transformer sizes, kWh of battery, MWh lost or
shifted per planning year). Prices and emission factors are applied here, vectorised over N parameter
samples, so the uncertainty analysis reuses the same power-flow results.

Conventions
- Horizon 2025 to 2045 (21 years), money in kEUR of 2025, discounted to the start of 2025 with all
  yearly cash flows (annuities, losses, payments) at the end of their year.
- Investments are turned into annuities over their life and only the annuities that fall inside the
  horizon are counted. This is the same as buying the asset and crediting its residual value at the
  end, and it treats a cable laid in 2044 and one laid in 2025 fairly.
- Embodied emissions are spread over the life of the asset in the same way (undiscounted).
- Values between planning years (losses, shifted energy, controllable devices) are interpolated
  linearly; the planning years are 2025, 2030, ..., 2045.
"""
import numpy as np

YEARS = np.arange(2025, 2046)
END = 2046                                  # assets serve until the end of 2045
AREA = {"rural": "rural", "semi_urban": "semi_urban", "urban": "urban"}

# parameters sampled in the uncertainty analysis: name -> (section, key) in params.yaml
UNCERTAIN = {
    "esc": ("escalation", "from_2016"),
    "copper_source": ("cost", "copper_cost_source_factor"),
    "battery_eur_per_kwh": ("cost", "battery_eur_per_kwh"),
    "battery_eur_per_kw": ("cost", "battery_inverter_eur_per_kw"),
    "battery_life": ("cost", "battery_life_years"),
    "flex_eur": ("cost", "flexibility_eur_per_device_year"),
    "loss_price": ("cost", "loss_price_eur_per_mwh"),
    "r": ("cost", "discount_rate"),
    "life": ("cost", "asset_life_years"),
    "cable_co2": ("lca", "cable_kgco2_per_km_nayy150"),
    "trafo_co2_per_kg": ("lca", "transformer_kgco2_per_kg"),
    "trafo_co2_other": ("lca", "transformer_other_kgco2"),
    "battery_co2": ("lca", "battery_kgco2_per_kwh"),
    "battery_al": ("lca", "battery_al_kg_per_kwh"),
    "zero_year": ("lca", "grid_zero_year"),
}


def central(P):
    """Central value of every uncertain parameter, as arrays of length 1."""
    return {k: np.array([float(P[s][key]["value"])]) for k, (s, key) in UNCERTAIN.items()}


def sample(P, n, seed=0):
    """n samples, each parameter triangular(low, central, high), independent."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, (s, key) in UNCERTAIN.items():
        d = P[s][key]
        lo, mode, hi = float(d["low"]), float(d["value"]), float(d["high"])
        out[k] = rng.triangular(lo, mode, hi, n) if hi > lo else np.full(n, mode)
    return out


def _interp(years, field):
    y = np.array([d["year"] for d in years], float)
    v = np.array([d[field] for d in years], float)
    return np.interp(YEARS, y, v)


def operation(led):
    """Yearly series 2025-2045 from the planning-year records."""
    return {f: _interp(led["years"], f) for f in ("losses_mwh", "curtailed_mwh", "battery_throughput_mwh", "devices")}


def feasible(led):
    return not led["unresolved"] and max(d["viol_hours"] for d in led["years"]) == 0


def evaluate(led, P, p, view="dso"):
    """Criteria of one ledger for parameter arrays p (each of shape (N,)).
    view = "dso": paragraph-14a payments are a cost of the flexibility pathways;
    view = "system": the payments are a transfer between customers and operator and count as zero.
    Returns a dict of arrays (N,): costs in kEUR, CO2 in t, metals in t, trench in km, energy in MWh."""
    c, lca = P["cost"], P["lca"]
    r = p["r"][:, None]
    disc = (1 + r) ** -(YEARS - 2024)[None, :]     # (N, 21) cash flows at the end of each year, PV in 2025

    def pv_annuity(invest, year, life):
        crf = p["r"] / (1 - (1 + p["r"]) ** -life)
        return invest * crf * (disc * (YEARS >= year)[None, :]).sum(axis=1)

    def share_in_horizon(year, life):
        return np.minimum(END - year, life) / life

    N = len(p["r"])
    z = np.zeros(N)
    out = {k: z.copy() for k in ("capex_cable", "capex_transformer", "capex_ront", "capex_battery",
                                 "co2_embodied", "metals_t")}
    trench_km = 0.0
    km_cost = c["cable_eur_per_km"][AREA[led["grid"]]]["value_2016"] * p["esc"] * p["copper_source"]
    swap = c["transformer_swap_eur"]["value_2016"] * p["esc"] * p["copper_source"]
    ront = c["ront_eur"]["value_2016"] * p["esc"]
    m630 = lca["transformer_mass_kg_630"]["value"]
    expo = lca["transformer_mass_exponent"]["value"]
    tr_metal = lca["transformer_al_kg_630"]["value"] + lca["transformer_cu_kg_630"]["value"]
    al_km = lca["cable_al_kg_per_km_nayy150"]["value"]
    for a in led["assets"]:
        y = a["year"]
        if a["kind"] == "cable":
            sec = a["mm2"] / 150
            out["capex_cable"] += pv_annuity(a["qty"] * km_cost, y, p["life"])
            out["co2_embodied"] += a["qty"] * p["cable_co2"] * sec * share_in_horizon(y, p["life"])
            out["metals_t"] += a["qty"] * al_km * sec / 1000
            trench_km += a["qty"]
        elif a["kind"] in ("transformer", "ront"):
            scale = (a["mva"] / 0.63) ** expo
            key = "capex_transformer" if a["kind"] == "transformer" else "capex_ront"
            out[key] += pv_annuity(swap if a["kind"] == "transformer" else ront, y, p["life"])
            emb = m630 * scale * p["trafo_co2_per_kg"] + p["trafo_co2_other"]
            out["co2_embodied"] += emb * share_in_horizon(y, p["life"])
            out["metals_t"] += tr_metal * scale / 1000
        elif a["kind"] == "battery":
            inv = a["qty"] * p["battery_eur_per_kwh"] + a["kw"] * p["battery_eur_per_kw"]
            out["capex_battery"] += pv_annuity(inv, y, p["battery_life"])
            out["co2_embodied"] += a["qty"] * p["battery_co2"] * share_in_horizon(y, p["battery_life"])
            out["metals_t"] += a["qty"] * (lca["battery_cu_kg_per_kwh"]["value"] + p["battery_al"]) / 1000
        else:
            raise ValueError(a["kind"])
    op = operation(led)
    losses = op["losses_mwh"] + lca["battery_loss_share_of_throughput"]["value"] * op["battery_throughput_mwh"]
    g0 = lca["grid_kgco2_per_kwh_2024"]["value"]
    zy = p["zero_year"][:, None]
    factor = g0 * np.clip((zy - YEARS[None, :]) / (zy - 2024), 0, 1)      # kg/kWh = t/MWh
    out["cost_losses"] = (losses[None, :] * p["loss_price"][:, None] * disc).sum(axis=1)
    flex = op["devices"][None, :] * p["flex_eur"][:, None] * disc
    out["cost_flex"] = flex.sum(axis=1) if view == "dso" else z.copy()
    for k in ("capex_cable", "capex_transformer", "capex_ront", "capex_battery", "cost_losses", "cost_flex"):
        out[k] = out[k] / 1000                                           # EUR -> kEUR
    out["cost"] = sum(out[k] for k in ("capex_cable", "capex_transformer", "capex_ront", "capex_battery",
                                       "cost_losses", "cost_flex"))
    out["co2_embodied"] = out["co2_embodied"] / 1000                     # kg -> t
    out["co2_losses"] = (losses[None, :] * factor).sum(axis=1)
    out["co2"] = out["co2_embodied"] + out["co2_losses"]
    out["trench_km"] = np.full(N, trench_km)
    out["shifted_mwh"] = np.full(N, op["curtailed_mwh"].sum())
    out["losses_mwh"] = np.full(N, losses.sum())
    return out
