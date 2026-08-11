"""Route-similarity challenge: geometry, features and label aggregation."""
import json, numpy as np, pandas as pd

R_EARTH = 6_371_000.0
SAME, DIFF, IDK = "Both are the same", "They differ", "I don't know"

# ====================================================================== geometry
def to_metres(coords, origin=None):
    a = np.asarray(coords, float)
    lat, lon = a[:, 0], a[:, 1]
    lat0, lon0 = origin if origin is not None else (lat.mean(), lon.mean())
    x = np.radians(lon - lon0) * R_EARTH * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * R_EARTH
    return np.column_stack([x, y])


def path_length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()) if len(P) > 1 else 0.0


def resample(P, n):
    """Arc-length resampling to n points."""
    if len(P) == 1:
        return np.repeat(P, n, axis=0)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    if d[-1] <= 0:
        return np.repeat(P[:1], n, axis=0)
    t = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(t, d, P[:, 0]), np.interp(t, d, P[:, 1])])


def dist_to_polyline(pts, poly):
    """Min distance from each point to the *segments* of poly (not just its vertices)."""
    if len(poly) < 2:
        return np.linalg.norm(pts - poly[0], axis=1)
    A, B = poly[:-1], poly[1:]
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    AP = pts[:, None, :] - A[None, :, :]
    t = np.clip((AP * AB[None]).sum(2) / L2[None], 0, 1)
    proj = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2).min(1)


def discrete_frechet(P, Q):
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    n, m = D.shape
    ca = np.empty((n, m))
    ca[0, 0] = D[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], D[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], D[0, j])
    for i in range(1, n):
        cur, prev, Di = ca[i], ca[i - 1], D[i]
        for j in range(1, m):
            cur[j] = max(min(prev[j], prev[j - 1], cur[j - 1]), Di[j])
    return float(ca[-1, -1])


def dtw_distance(P, Q):
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=2)
    n, m = D.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        cur, prev, Di = acc[i], acc[i - 1], D[i - 1]
        for j in range(1, m + 1):
            cur[j] = Di[j - 1] + min(prev[j], cur[j - 1], prev[j - 1])
    return float(acc[-1, -1] / (n + m))


def bearings(P):
    d = np.diff(P, axis=0)
    return np.arctan2(d[:, 1], d[:, 0])

# ====================================================================== features
NRS = 120   # resampling density for distance profiles
NDP = 40    # resampling density for the O(n*m) dynamic-programme metrics


def route_features(est_ll, real_ll):
    """All features are derived from the two coordinate lists alone."""
    origin = (np.asarray(est_ll, float)[:, 0].mean(), np.asarray(est_ll, float)[:, 1].mean())
    E, Rr = to_metres(est_ll, origin), to_metres(real_ll, origin)
    f = {}

    le, lr = path_length(E), path_length(Rr)
    scale = max((le + lr) / 2, 1.0)
    f["len_est"], f["len_real"] = le, lr
    f["len_ratio"] = min(le, lr) / max(le, lr, 1.0)
    f["len_rel_diff"] = abs(le - lr) / scale
    f["detour_ratio"] = lr / max(le, 1.0)

    f["start_gap"] = float(np.linalg.norm(E[0] - Rr[0]))
    f["end_gap"] = float(np.linalg.norm(E[-1] - Rr[-1]))
    f["endpoint_gap_norm"] = (f["start_gap"] + f["end_gap"]) / scale

    Er, Rs = resample(E, NRS), resample(Rr, NRS)
    d_r2e = dist_to_polyline(Rs, E)      # how far the real route strays from the estimate
    d_e2r = dist_to_polyline(Er, Rr)
    both = np.concatenate([d_r2e, d_e2r])

    f["dist_mean"] = float(both.mean())
    f["dist_median"] = float(np.median(both))
    f["dist_p90"] = float(np.percentile(both, 90))
    f["dist_max"] = float(both.max())          # symmetric Hausdorff
    f["dist_mean_norm"] = f["dist_mean"] / scale
    f["dist_p90_norm"] = f["dist_p90"] / scale
    f["hausdorff_norm"] = f["dist_max"] / scale
    f["asym_gap"] = float(abs(d_r2e.mean() - d_e2r.mean()))

    # what a human actually sees: how much of the route visibly overlaps
    for thr in (10, 25, 50, 100, 200):
        f[f"overlap_{thr}m"] = float(((d_r2e <= thr).mean() + (d_e2r <= thr).mean()) / 2)

    # a single sustained detour reads as "different" even when overall overlap is high
    step = max(lr / (NRS - 1), 1e-6)
    off = d_r2e > 30.0
    runs, cur = [], 0
    for v in off:
        if v:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    runs = np.array(runs) * step if runs else np.array([0.0])
    f["longest_deviation_m"] = float(runs.max())
    f["longest_deviation_frac"] = float(runs.max() / max(lr, 1.0))
    f["n_deviation_episodes"] = int((runs > 50).sum())
    f["deviated_frac"] = float(off.mean())

    # area swept between the two curves, normalised
    f["area_between_norm"] = float(np.trapezoid(d_r2e, dx=step) / max(lr, 1.0))

    # shape: heading profiles
    be, br = bearings(resample(E, NDP)), bearings(resample(Rr, NDP))
    dth = np.abs(np.angle(np.exp(1j * (be - br))))
    f["bearing_diff_mean"] = float(dth.mean())
    f["bearing_diff_p90"] = float(np.percentile(dth, 90))

    # bounding-box overlap
    def bbox(P):
        return P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()
    ax0, ay0, ax1, ay1 = bbox(E)
    bx0, by0, bx1, by1 = bbox(Rr)
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    f["bbox_iou"] = float(inter / union) if union > 0 else 0.0

    Ed, Rd = resample(E, NDP), resample(Rr, NDP)
    f["frechet"] = discrete_frechet(Ed, Rd)
    f["frechet_norm"] = f["frechet"] / scale
    f["dtw_norm"] = dtw_distance(Ed, Rd) / scale

    f["n_pts_est"], f["n_pts_real"] = len(est_ll), len(real_ll)
    return f

# ====================================================================== labels
def majority_label(votes):
    """Majority over decisive votes; None when tied or wholly undecided."""
    n_s = sum(v == SAME for v in votes)
    n_d = sum(v == DIFF for v in votes)
    if n_s == n_d:
        return None
    return 1 if n_d > n_s else 0


def soft_label(votes):
    n_s = sum(v == SAME for v in votes)
    n_d = sum(v == DIFF for v in votes)
    return np.nan if (n_s + n_d) == 0 else n_d / (n_s + n_d)


def dawid_skene(df, n_iter=60, tol=1e-7):
    """EM for latent binary truth with per-annotator 2x3 confusion matrices.
    'I don't know' is modelled as a third emitted category, not discarded."""
    cats = [SAME, DIFF, IDK]
    jid = {j: i for i, j in enumerate(df.journey_id.unique())}
    aid = {a: i for i, a in enumerate(df.annotator.unique())}
    J, A, C = len(jid), len(aid), 3
    ji = df.journey_id.map(jid).to_numpy()
    ai = df.annotator.map(aid).to_numpy()
    ci = df.annotation.map({c: k for k, c in enumerate(cats)}).to_numpy()

    T = np.zeros((J, 2))
    for j, c in zip(ji, ci):
        if c == 0: T[j, 0] += 1
        elif c == 1: T[j, 1] += 1
    T[T.sum(1) == 0] = 0.5
    T /= T.sum(1, keepdims=True)

    prev = None
    for _ in range(n_iter):
        pi = T.mean(0)
        cm = np.full((A, 2, C), 0.5)          # Dirichlet pseudo-counts
        for k in range(2):
            np.add.at(cm[:, k, :], (ai, ci), T[ji, k])
        cm = cm / cm.sum(2, keepdims=True)
        logp = np.tile(np.log(np.maximum(pi, 1e-12)), (J, 1))
        for k in range(2):
            np.add.at(logp[:, k], ji, np.log(np.maximum(cm[ai, k, ci], 1e-12)))
        logp -= logp.max(1, keepdims=True)
        T = np.exp(logp); T /= T.sum(1, keepdims=True)
        cur = T[:, 1].sum()
        if prev is not None and abs(cur - prev) < tol:
            break
        prev = cur
    out = pd.Series(T[:, 1], index=list(jid.keys()), name="ds_p_differ")
    rel = pd.DataFrame({"annotator": list(aid.keys()),
                        "p_same_correct": cm[:, 0, 0], "p_differ_correct": cm[:, 1, 1],
                        "p_idk_when_same": cm[:, 0, 2], "p_idk_when_differ": cm[:, 1, 2]})
    return out, rel

# ====================================================================== synthetic stand-in
def _make_route(rng, n_seg, seg_len):
    pt = np.array([0.0, 0.0]); th = rng.uniform(0, 2 * np.pi)
    pts = [pt.copy()]
    for _ in range(n_seg):
        th += rng.normal(0, 0.55)
        L = seg_len * rng.uniform(0.6, 1.4)
        k = max(2, int(L / 80))
        for _ in range(k):
            pt = pt + np.array([np.cos(th), np.sin(th)]) * (L / k)
            pts.append(pt.copy())
    return np.array(pts)


def _perturb(rng, P, kind):
    Q = P.copy()
    n = len(P)
    if kind == "same":
        return Q + rng.normal(0, 6, Q.shape)
    lo = rng.integers(int(0.15 * n), int(0.55 * n))
    span = {"minor": (0.10, 0.25), "major": (0.25, 0.55)}[kind] if kind != "endpoint" else (0.05, 0.12)
    hi = min(n - 1, lo + int(rng.uniform(*span) * n))
    amp = {"minor": rng.uniform(50, 180), "major": rng.uniform(250, 1200),
           "endpoint": rng.uniform(300, 1500)}[kind]
    d = np.diff(P, axis=0)
    th = np.arctan2(d[:, 1], d[:, 0])
    nrm = np.column_stack([-np.sin(th), np.cos(th)])
    nrm = np.vstack([nrm, nrm[-1]])
    w = np.zeros(n)
    if kind == "endpoint":
        w[-max(2, n // 8):] = np.linspace(0, 1, max(2, n // 8))
    else:
        L = hi - lo
        w[lo:hi] = np.sin(np.linspace(0, np.pi, L))
    Q = Q + nrm * (amp * w)[:, None]
    return Q + rng.normal(0, 6, Q.shape)


def generate_dataset(path, n_journeys=5000, n_annotators=8, seed=7):
    """Synthetic stand-in with the schema of challenge_dataset.json."""
    rng = np.random.default_rng(seed)
    kinds = rng.choice(["same", "minor", "major", "endpoint"], n_journeys,
                       p=[0.50, 0.24, 0.16, 0.10])
    thr = rng.normal(0.023, 0.013, n_annotators).clip(0.007, 0.055)
    sharp = rng.uniform(85, 240, n_annotators)
    idk_base = rng.uniform(0.015, 0.075, n_annotators)
    rows = []
    for i in range(n_journeys):
        lat0, lon0 = -12.05 + rng.normal(0, .045), -77.03 + rng.normal(0, .045)
        E = _make_route(rng, rng.integers(3, 9), rng.uniform(300, 1400))
        Rr = _perturb(rng, E, kinds[i])
        Es, Rs = resample(E, 90), resample(Rr, 90)
        score = dist_to_polyline(Rs, E).mean() / max(path_length(E), 1.0)   # latent divergence
        jid = f"{i:08x}-54c5-11ec-ae0a-{rng.integers(0, 16**12):012x}"
        to_ll = lambda P: [[round(lat0 + p[1] / R_EARTH * 180 / np.pi, 6),
                            round(lon0 + p[0] / (R_EARTH * np.cos(np.radians(lat0))) * 180 / np.pi, 6)]
                           for p in P]
        est_ll, real_ll = to_ll(Es), to_ll(Rs)
        for a in rng.choice(n_annotators, rng.integers(1, 5), replace=False):
            p_diff = 1 / (1 + np.exp(-sharp[a] * (score - thr[a])))
            p_idk = idk_base[a] + 0.30 * np.exp(-((p_diff - 0.5) / 0.16) ** 2)
            u = rng.random()
            lab = IDK if u < p_idk else (DIFF if rng.random() < p_diff else SAME)
            rows.append({"journey_id": jid, "annotator": int(a), "annotation": lab,
                         "estimated_route": est_ll, "real_route": real_ll})
    rng.shuffle(rows)
    with open(path, "w") as fh:
        json.dump(rows, fh, separators=(",", ":"))
    return len(rows)


# ====================================================================== geography
CITIES = {
    "A Coruña": (43.36, -8.41), "Sevilla": (37.39, -5.99), "Madrid": (40.42, -3.70),
    "Barcelona": (41.39, 2.17), "Valencia": (39.47, -0.38), "Málaga": (36.72, -4.42),
    "Bilbao": (43.26, -2.93), "Zaragoza": (41.65, -0.89), "Alicante": (38.35, -0.48),
    "Murcia": (37.99, -1.13), "Lisboa": (38.72, -9.14), "Ciudad de México": (19.43, -99.13),
    "Puebla": (19.04, -98.20), "Monterrey": (25.69, -100.32), "Guadalajara": (20.66, -103.35),
    "Querétaro": (20.59, -100.39), "Buenos Aires": (-34.60, -58.44), "Córdoba (AR)": (-31.42, -64.18),
    "Rosario": (-32.95, -60.64), "Mendoza": (-32.89, -68.84), "Montevideo": (-34.90, -56.16),
    "Santiago": (-33.45, -70.67), "Concepción": (-36.83, -73.05), "Valparaíso": (-33.05, -71.62),
    "Antofagasta": (-23.65, -70.40), "Lima": (-12.05, -77.04), "Arequipa": (-16.41, -71.54),
    "Bogotá": (4.71, -74.07), "Barranquilla": (10.97, -74.80), "Medellín": (6.24, -75.57),
    "Cali": (3.45, -76.53), "Quito": (-0.18, -78.47), "Guayaquil": (-2.19, -79.89),
}


def assign_city(lat, lon, max_deg=1.6):
    """Nearest known Cabify city centroid, or 'other'."""
    best, bd = "other", 1e9
    for name, (la, lo) in CITIES.items():
        d = ((lat - la) ** 2 + (lon - lo) ** 2) ** 0.5
        if d < bd:
            best, bd = name, d
    return best if bd <= max_deg else "other"
