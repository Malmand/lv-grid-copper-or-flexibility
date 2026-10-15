"""Electrification of a SimBench low-voltage feeder over the planning years.

Start: the SimBench feeder as of today (scenario 0): households, commercial and agricultural loads and
the PV already installed, each with its own measured-type annual profile (15 min, 2016).
Growth: heat pumps, home EV chargers and rooftop PV are added to households. Their annual profiles
are real SimBench device profiles taken from the future-scenario versions (scenario 2) of the three
feeders, with the sizes SimBench gives them. The share of households with each technology follows
an S-curve through the 2030 and 2045 values in params.yaml.

Adoption is nested: households are ranked once per feeder and technology, and the first k adopt. A
household that has a heat pump in the low scenario has one in the central and high scenarios too,
and keeps it in later years, so differences between scenarios and years are not random noise.
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import simbench as sb

GRID_CODES = {"rural": "1-LV-rural2", "semi_urban": "1-LV-semiurb5", "urban": "1-LV-urban6"}
TECH = ("hp", "ev", "pv")


@dataclass
class Device:
    kind: str
    P: np.ndarray            # MW, positive = consumption (PV: generation, positive)
    Q: np.ndarray            # Mvar
    size_mw: float


@dataclass
class Library:
    hp: list = field(default_factory=list)
    ev: list = field(default_factory=list)
    pv: list = field(default_factory=list)


def _clean_net(code):
    net = sb.get_simbench_net(code)
    net.ext_grid["vm_pu"] = 1.0           # MV reference, voltages are then changes from the MV side
    net.trafo["tap_pos"] = 0
    net.storage["in_service"] = False     # home batteries are left out (not part of any pathway)
    return net


@lru_cache(maxsize=None)
def library():
    """Real device profiles from the scenario-2 versions of the three feeders."""
    lib = Library()
    for code in GRID_CODES.values():
        net = _clean_net(f"{code}--2-sw")
        prof = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
        P, Q = prof[("load", "p_mw")], prof[("load", "q_mvar")]
        for i, ld in net.load.iterrows():
            name = str(ld.profile)
            if name.startswith(("Air", "Soil")):
                lib.hp.append(Device("hp", P[i].to_numpy(), Q[i].to_numpy(), float(ld.p_mw)))
            elif name.startswith("HLS"):                  # home charging stations
                lib.ev.append(Device("ev", P[i].to_numpy(), Q[i].to_numpy(), float(ld.p_mw)))
        G = prof[("sgen", "p_mw")]
        for i, sg in net.sgen.iterrows():
            if sg.p_mw <= 0.03:                           # rooftop systems, not the large plants
                lib.pv.append(Device("pv", G[i].to_numpy(), np.zeros(len(G)), float(sg.p_mw)))
    return lib


def s_curve(year, y2030, y2045, y0, year0=2025):
    """Share in a given year: logistic curve through (2030, y2030) and (2045, y2045), starting from y0
    in 2025 (linear before 2030 if the logistic fit is not possible)."""
    if year <= year0:
        return y0
    if year <= 2030:
        return y0 + (y2030 - y0) * (year - year0) / (2030 - year0)
    # logistic with saturation 1 through the two anchor points
    a1 = np.log(y2030 / (1 - y2030))
    a2 = np.log(y2045 / (1 - y2045))
    k = (a2 - a1) / 15
    return float(1 / (1 + np.exp(-(a1 + k * (year - 2030)))))


class Feeder:
    """Base feeder plus the ordered list of households for each technology."""

    def __init__(self, grid, seed=0):
        self.grid = grid
        self.net = _clean_net(f"{GRID_CODES[grid]}--0-sw")
        prof = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=True)
        self.T = prof[("load", "p_mw")].shape[0]
        self.base_P = {i: prof[("load", "p_mw")][i].to_numpy() for i in self.net.load.index}
        self.base_Q = {i: prof[("load", "q_mvar")][i].to_numpy() for i in self.net.load.index}
        self.base_G = {i: prof[("sgen", "p_mw")][i].to_numpy() for i in self.net.sgen.index}
        loads = self.net.load
        self.households = loads.index[loads.profile.astype(str).str.startswith("H0")].to_numpy()
        pv_buses = set(self.net.sgen.bus)
        self.pv0 = np.mean([loads.bus[h] in pv_buses for h in self.households])
        rng = np.random.default_rng(seed)
        lib = library()
        self.order, self.device = {}, {}
        for tech in TECH:
            cand = self.households if tech != "pv" else np.array(
                [h for h in self.households if loads.bus[h] not in pv_buses])
            self.order[tech] = rng.permutation(cand)
            pool = getattr(lib, tech)
            self.device[tech] = {h: pool[rng.integers(len(pool))] for h in cand}

    def adopters(self, tech, share):
        n_hh = len(self.households)
        if tech == "pv":
            k = int(round(max(share - self.pv0, 0) * n_hh))
        else:
            k = int(round(share * n_hh))
        return self.order[tech][:min(k, len(self.order[tech]))]

    def shares(self, year, adoption):
        """Technology shares in a year for one adoption scenario (dict from params['adoption'][name])."""
        start = {"hp": 0.0, "ev": 0.0, "pv": self.pv0}
        return {t: max(start[t], s_curve(year, adoption[t][2030], adoption[t][2045], start[t])) for t in TECH}

    def injections(self, year, adoption, flex_devices=None):
        """Per-element series for a year: returns dict bus -> (P_load, Q_load, P_gen) and the list of
        controllable devices (heat pumps and EV chargers) with their bus and profile."""
        sh = self.shares(year, adoption)
        loads = self.net.load
        P = {}
        for i in loads.index:
            b = loads.bus[i]
            p, q, g = P.get(b, (0.0, 0.0, 0.0))
            P[b] = (p + self.base_P[i], q + self.base_Q[i], g)
        for i in self.net.sgen.index:
            b = self.net.sgen.bus[i]
            p, q, g = P.get(b, (0.0, 0.0, 0.0))
            P[b] = (p, q, g + self.base_G[i])
        devices = []
        for tech in TECH:
            for h in self.adopters(tech, sh[tech]):
                d = self.device[tech][h]
                b = loads.bus[h]
                p, q, g = P.get(b, (0.0, 0.0, 0.0))
                if tech == "pv":
                    P[b] = (p, q, g + d.P)
                else:
                    devices.append((tech, b, d))
                    P[b] = (p + d.P, q + d.Q, g)
        counts = {t: len(self.adopters(t, sh[t])) for t in TECH}
        return P, devices, sh, counts
