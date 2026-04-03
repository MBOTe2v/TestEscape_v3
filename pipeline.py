"""
Pipeline de détection de doublons — TestEscape_v3

Détecte si des pièces d'un NewBatch ont déjà été testées dans la Database
historique, en comparant leurs signatures électriques via distance de
Mahalanobis sur les valeurs brutes des features lot-invariantes.

Usage:
    python pipeline.py
"""
import numpy as np
import pandas as pd
import config

# ═══════════════════════════════════════════════════════════════
# Grille de recherche pour l'entraînement automatique
# ═══════════════════════════════════════════════════════════════
ICC_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
TOPN_GRID = [10, 20, 30, 40, 50, 60, 80, 100]
# Grille complète pour α : la métrique cross-lot empêche le sur-apprentissage
# within-lot, donc on peut explorer toutes les valeurs sans biais.
ALPHA_GRID = [round(a * 0.1, 1) for a in range(11)]  # 0.0 → 1.0


def load_raw(path):
    """Charge un CSV européen sans aucun filtrage."""
    df = pd.read_csv(path, sep=";", decimal=",")
    df = df[df["lot_id"].notna()].copy()
    df["lot_id"] = df["lot_id"].astype(np.int64)
    df["part_id"] = df["part_id"].astype(np.int64)
    return df


def get_feature_cols(df):
    """Colonnes de mesure (exclut ID, filtres qualité)."""
    feat = [c for c in df.columns if c not in config.ID_COLS]
    feat = [c for c in feat if df[c].notna().mean() >= 0.8
            and "Nb_loop" not in c and "Setup_time" not in c]
    return feat


def find_duplicate_pairs(df):
    """Trouve les paires (lot_id, part_id) avec 2+ lignes = retests."""
    dup = df.groupby(["lot_id", "part_id"]).filter(lambda x: len(x) > 1)
    pairs = []
    for (lid, pid), grp in dup.groupby(["lot_id", "part_id"]):
        pairs.append({"lot_id": lid, "part_id": pid,
                      "idx_0": grp.index[0], "idx_1": grp.index[1]})
    return pairs


def compute_icc(raw_matrix, lots):
    """ICC = var_between / (var_between + var_within) par feature."""
    n_feat = raw_matrix.shape[1]
    icc = np.zeros(n_feat)
    for j in range(n_feat):
        lot_means, lot_vars, lot_ns = [], [], []
        for lid in np.unique(lots):
            vals = raw_matrix[lots == lid, j]
            vals = vals[~np.isnan(vals)]
            if len(vals) < 2:
                continue
            lot_means.append(np.mean(vals))
            lot_vars.append(np.var(vals))
            lot_ns.append(len(vals))
        if len(lot_means) < 2:
            icc[j] = 1.0
            continue
        lot_means = np.array(lot_means)
        lot_vars = np.array(lot_vars)
        lot_ns = np.array(lot_ns)
        sw = np.average(lot_vars, weights=lot_ns)
        grand = np.average(lot_means, weights=lot_ns)
        sb = np.average((lot_means - grand) ** 2, weights=lot_ns)
        total = sb + sw
        icc[j] = sb / total if total > 0 else 0.0
    return icc


def compute_noise_snr(raw_all, pairs, feat_cols):
    """SNR = std_global / std_retest sur les valeurs brutes."""
    raw = raw_all[feat_cols].values.astype(np.float64)
    diffs = np.array([raw[p["idx_0"]] - raw[p["idx_1"]] for p in pairs])
    retest_std = np.std(diffs, axis=0)
    global_std = np.nanstd(raw, axis=0)
    snr = np.where(retest_std > 0, global_std / retest_std, 0)
    return snr, diffs


def _select_features_idx(icc, snr, icc_thresh, top_n):
    """Retourne les indices des features sélectionnées (sans affichage)."""
    valid = icc < icc_thresh
    idx = np.where(valid)[0]
    snr_sub = snr[idx]
    ranked = idx[np.argsort(snr_sub)[::-1]]
    return ranked[:top_n]


def select_features(icc, snr, feat_cols, icc_thresh, top_n):
    """Sélectionne les top_n features avec ICC < seuil, triées par SNR."""
    sel = _select_features_idx(icc, snr, icc_thresh, top_n)
    print(f"Features sélectionnées: {len(sel)} (ICC<{icc_thresh}, top-{top_n})")
    for k in range(min(10, len(sel))):
        i = sel[k]
        print(f"  {k+1:2d}. {feat_cols[i]:<50s} ICC={icc[i]:.4f} SNR={snr[i]:.2f}")
    if len(sel) > 10:
        print(f"  ... ({len(sel) - 10} de plus)")
    return sel


def build_mahalanobis_inv(diffs_sel, alpha, n):
    """Construit la matrice inverse pour Mahalanobis avec shrinkage."""
    cov = np.cov(diffs_sel, rowvar=False)
    diag = np.diag(np.diag(cov))
    cov_reg = (1 - alpha) * cov + alpha * diag + np.eye(n) * 0.005
    return np.linalg.inv(cov_reg)


def mahalanobis_search(query, search_matrix, cov_inv):
    """Distance de Mahalanobis de query vers chaque ligne de search_matrix."""
    diff = search_matrix - query
    return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))


def train(db, db_raw, pairs, feat_cols, icc, snr, raw_diffs):
    """Grid search automatique optimisé pour la performance cross-lot.

    Métrique : rang cross-lot — pour chaque paire de retest (même lot),
    on mesure combien de parts d'AUTRES lots battent la cible.
    Si la cible bat toutes les parts cross-lot → rank_xl = 1.
    Cela élimine l'avantage within-lot et sélectionne des configs
    qui fonctionnent quand le query vient d'un lot différent.

    Critère de sélection (par priorité) :
      1. Max cross-lot rank-1 count
      2. Min median cross-lot rank (départage)
      3. Max alpha (régularisation = robustesse)
      4. Min top_n (parcimonie)
    """
    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT — optimisation cross-lot")
    print("=" * 70)

    raw_db_all = db_raw[feat_cols].values.astype(np.float64)
    raw_db_dedup = db[feat_cols].values.astype(np.float64)
    lots_dedup = db["lot_id"].values

    # Pré-calcul des indices cibles dans la DB dédupliquée
    pair_targets = []
    for p in pairs:
        mask = (lots_dedup == p["lot_id"]) & (db["part_id"].values == p["part_id"])
        tidx = np.where(mask)[0]
        if len(tidx) > 0:
            pair_targets.append((p, tidx[0]))

    n_pairs = len(pair_targets)
    n_configs = len(ICC_GRID) * len(TOPN_GRID) * len(ALPHA_GRID)
    print(f"Paires d'évaluation: {n_pairs}")
    print(f"Grille: {len(ICC_GRID)} ICC × {len(TOPN_GRID)} top_n × {len(ALPHA_GRID)} α "
          f"= {n_configs} configs")

    # Pré-calculer les masques cross-lot pour chaque paire
    xl_masks = []
    for p, target_idx in pair_targets:
        xl_masks.append(lots_dedup != p["lot_id"])

    # score tuple : maximiser rank1, puis minimiser median rank,
    # puis préférer alpha élevé et top_n petit (robustesse cross-lot)
    best_score = (-1, -9999, -1.0, 0)
    best_config = (0.6, 40, 1.0)

    for icc_thresh in ICC_GRID:
        for top_n in TOPN_GRID:
            sel = _select_features_idx(icc, snr, icc_thresh, top_n)
            if len(sel) < 2:
                continue

            search = np.nan_to_num(raw_db_dedup[:, sel], nan=0.0)
            diffs_sel = raw_diffs[:, sel]

            for alpha in ALPHA_GRID:
                try:
                    cov_inv = build_mahalanobis_inv(diffs_sel, alpha, len(sel))
                except np.linalg.LinAlgError:
                    continue

                xl_rank1 = 0
                xl_ranks = []
                for i, (p, target_idx) in enumerate(pair_targets):
                    query = np.nan_to_num(raw_db_all[p["idx_0"], sel], nan=0.0)
                    d = mahalanobis_search(query, search, cov_inv)
                    d_target = d[target_idx]
                    # Rang cross-lot : combien de parts d'autres lots sont plus proches
                    xl_rank = 1 + int(np.sum(d[xl_masks[i]] < d_target))
                    xl_ranks.append(xl_rank)
                    if xl_rank == 1:
                        xl_rank1 += 1

                med_xl = int(np.median(xl_ranks))
                score = (xl_rank1, -med_xl, alpha, -top_n)
                if score > best_score:
                    best_score = score
                    best_config = (icc_thresh, top_n, alpha)

    icc_t, top_n, alpha = best_config
    xl_r1 = best_score[0]
    med = -best_score[3]
    print(f"\nMeilleure config trouvée:")
    print(f"  ICC < {icc_t}")
    print(f"  top_n = {top_n}")
    print(f"  α     = {alpha}")
    print(f"  Score cross-lot: {xl_r1}/{n_pairs} rank-1, médiane rank = {med}")

    return icc_t, top_n, alpha


def run():
    # ── 1. Chargement ──
    print("=" * 70)
    print("CHARGEMENT DES DONNÉES")
    print("=" * 70)
    db_raw = load_raw(config.DATABASE_CSV)
    nb_raw = load_raw(config.NEWBATCH_CSV)
    feat_cols = get_feature_cols(db_raw)
    print(f"Database: {len(db_raw)} lignes, {db_raw['lot_id'].nunique()} lots")
    print(f"NewBatch: {len(nb_raw)} lignes, {nb_raw['lot_id'].nunique()} lots")
    print(f"Features disponibles: {len(feat_cols)}")

    # ── 2. Déduplication DB ──
    db = db_raw.drop_duplicates(
        subset=["lot_id", "part_id"], keep="last"
    ).reset_index(drop=True)
    print(f"Database dédupliquée: {len(db)} pièces uniques")

    # ── 3. Paires de retest (bruit) ──
    pairs = find_duplicate_pairs(db_raw)
    print(f"Paires de retest trouvées: {len(pairs)}")
    if len(pairs) == 0:
        print("ERREUR: aucune paire de retest, impossible de calibrer le modèle.")
        return

    # ── 4. Calcul ICC + SNR ──
    print("\n" + "=" * 70)
    print("SÉLECTION DE FEATURES")
    print("=" * 70)
    raw_matrix = db[feat_cols].values.astype(np.float64)
    icc = compute_icc(raw_matrix, db["lot_id"].values)
    snr, raw_diffs = compute_noise_snr(db_raw, pairs, feat_cols)

    # ── 5. Entraînement automatique ──
    icc_thresh, top_n, alpha = train(db, db_raw, pairs, feat_cols, icc, snr, raw_diffs)

    # ── 6. Sélection features avec la config optimale ──
    print("\n" + "=" * 70)
    print("FEATURES SÉLECTIONNÉES")
    print("=" * 70)
    sel = select_features(icc, snr, feat_cols, icc_thresh, top_n)

    # ── 7. Matrice Mahalanobis ──
    diffs_sel = raw_diffs[:, sel]
    cov_inv = build_mahalanobis_inv(diffs_sel, alpha, len(sel))

    # ── 8. Préparation données ──
    search = np.nan_to_num(raw_matrix[:, sel], nan=0.0)
    nb_matrix = np.nan_to_num(
        nb_raw[feat_cols].values.astype(np.float64)[:, sel], nan=0.0
    )

    # ── 9. Recherche ──
    print("\n" + "=" * 70)
    print("RECHERCHE DE DOUBLONS")
    print("=" * 70)
    results = []
    for i in range(len(nb_raw)):
        d = mahalanobis_search(nb_matrix[i], search, cov_inv)
        order = np.argsort(d)
        d1, d2 = d[order[0]], d[order[1]]
        gap = d2 / d1 if d1 > 0 else 1.0
        results.append({
            "nb_lot": int(nb_raw.iloc[i]["lot_id"]),
            "nb_part": int(nb_raw.iloc[i]["part_id"]),
            "db_lot_1": int(db.iloc[order[0]]["lot_id"]),
            "db_part_1": int(db.iloc[order[0]]["part_id"]),
            "dist_1": round(float(d1), 4),
            "db_lot_2": int(db.iloc[order[1]]["lot_id"]),
            "db_part_2": int(db.iloc[order[1]]["part_id"]),
            "dist_2": round(float(d2), 4),
            "gap_ratio": round(float(gap), 4),
        })

    df = pd.DataFrame(results).sort_values("gap_ratio", ascending=False)

    # ── 10. Résultats ──
    print(f"\nTop 20 pièces les plus suspectes (par gap ratio):")
    print(f"{'NB_part':>8} {'DB_lot':>12} {'DB_part':>8} {'dist_1':>8} {'dist_2':>8} {'gap':>7}")
    for _, r in df.head(20).iterrows():
        print(f"{int(r['nb_part']):8d} {int(r['db_lot_1']):12d} {int(r['db_part_1']):8d} "
              f"{r['dist_1']:8.4f} {r['dist_2']:8.4f} {r['gap_ratio']:7.4f}")

    # ── 11. Export ──
    out_path = config.OUTPUT_DIR + "/duplicate_detection_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nRésultats sauvegardés: {out_path}")

    return df


if __name__ == "__main__":
    run()
