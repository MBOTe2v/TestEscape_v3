# Journal de bord — TestEscape_v3

## 2026-04-03

### Exploration initiale
- DB: 3639 parts (22 lots), NB: 102 parts (1 lot), 709 colonnes
- SFDR dispo sur 3 fréquences (100M, 620M, 1.2G) × 4 cœurs ADC
- Ground truth initial: NB 228→DB(660864365, 37), NB 229→DB(660950327, 118). Part 229 absente du NB.
- GT retiré ensuite — on travaille à l'aveugle

### Tentative 1 : lot-normalisation + corrélation de profil (ÉCHEC)
- 150 features sélectionnées par discrimination score (within_lot_std/global_std)
- Lot-normalisation puis corrélation Pearson + cosine + euclidean pondérés
- Résultat : cible part 228 rang **1857/3639**. Totalement raté.
- Cause : le scoring de features ne pénalisait pas l'effet lot

### Tentative 2 : z-score global + euclidien (ÉCHEC)
- 150 features, normalisation par stats DB globales
- Distance euclidienne pure + gap ratio
- Résultat : les matches sont dominés par l'effet lot. Pas de séparation.

### Tentative 3 : recherche de doublons dans la DB
- SFDR 1.2G : trouvé 15 paires cross-lot proches (dist 0.05, gap >2x)
- **Mais** vérification avec toutes les features → avg_zdiff > 0.5 → FAUX POSITIFS
- SFDR seul (4 dimensions) produit des collisions accidentelles entre dies différents
- Il n'y a pas de vrais doublons dans la DB

### Tentative 4 : SNR scoring + évaluation synthétique (SUCCÈS ✓)
- SNR = within_lot_std / between_lot_std → pénalise les features dominées par l'effet lot
- Top feature : SDAMaxCorA (SNR=15.2), puis DNL/INL 100MHz (SNR 10-14)
- **Surprise** : SFDR 1.2GHz PAS dans le top 20, 100MHz est meilleur
- Éval synthétique (500 trials, cross-lot injection) :
  - Top-8 : 97.8% rank-1
  - Top-50 : 100% rank-1, ratio distance 3.36x
- **La méthode lot-normalisation + top-N features SNR fonctionne**

### Tentative 5 : SNR v2 — bruit de retest correct (PARTIEL)
- Correction SNR : noise = variation lot-normalisée du même die retesté dans un lot différent
- SNR_eff = 1/noise_median. Top: SDAMaxCorA (20.21), A_DNLmaxcor100M (11.32)
- Pipeline: weighted Euclidean (weight=1/retest_variance) calibré sur 23 paires réelles
- NB228 rank=3 avec top-15 features (gap=1.343) — mais matche DB part 159 (pas 85)
- Problème: instabilité du match selon le nombre de features

### Tentative 6 : Inter/intra ratio + retest-calibrated weights (INFORMATIF)
- Ratio = mean|dieA-dieB| / mean|test1-test2| calculé sur les 23 paires réelles
- Top 3: SDA_RangeA (3.03), Bcor_THD1G2_2G5 (2.51), SDAMaxCorA (2.49)
- Weighted Euclidean: NB228 rank=3 mais toujours mauvaise pièce identifiée

### Tentative 7 : Cross-core composites A-B, A-C, (A+B)-(C+D) (MITIGÉ)
- 37 groupes 4-cores, 555 features composites créées
- Meilleur ratio composite: cor_SNR1G2_2G5_BmC = 1.70 (plus faible que raw top-3)
- Synthetic eval: 3 raw + 15 composites → 95.8% rank-1
- Mais inference NB instable: NB228 varie entre rank 6 et 19 selon config

### Tentative 8 : Combined MEAN(A,B,C,D) (NÉGATIF)
- 50 groupes avec 4 cores A/B/C/D trouvés
- Moyenner les 4 cores DÉGRADE le SNR des meilleures features
  - SDAMaxCorA seul: SNR=2.13, MEAN4: SNR=1.43 (perte 33%)
  - Le core A est systématiquement le plus discriminant
- Découverte: NB 201 → DB part 85 lot=660989448 (rank 3) — mauvais lot !

### Ground truth confirmé par l'utilisateur
- **NB 228 ↔ DB lot=660864365, part_id=85**

### Tentative 9 : Mahalanobis distance (SUCCÈS ✓✓✓)
- Approche: distance de Mahalanobis entraînée sur les 23 paires de doublons réels
- Covariance du bruit Σ estimée sur les z-différences des 23 paires
- d(a,b) = sqrt((za-zb)^T Σ^{-1} (za-zb))
- Grid search: top_n features × alpha (shrinkage diagonal)
- **Résultat: 23/23 rank-1 (100%) avec top-100 features, α=0.0**
- Toutes les 23 paires retrouvées en rank-1 parmi ~150-213 pièces par lot
- Mahalanobis >> Weighted Euclidean (5/23 max) — les corrélations entre features sont cruciales
- Ledoit-Wolf shrinkage (sklearn): 21/23 → inférieur à α=0
- La covariance complète sans shrinkage est optimale (23 échantillons suffisent pour 100 features)

#### Configs notables (rank-1 / top-3):
| top_n | α=0.00 | α=0.02 | α=0.05 |
|-------|--------|--------|--------|
| 30    | 17/22  | 19/23  | 19/22  |
| 50    | 19/22  | 19/23  | 21/23  |
| 100   | **23/23** | **23/23** | **23/23** |

#### Pourquoi Mahalanobis fonctionne:
- Le bruit de retest est *corrélé* entre features (ex: si THD monte, SFDR descend)
- La covariance capture ces corrélations et les "annule"
- Le weighted Euclidean (diagonal) ignore ces corrélations → perd beaucoup d'info

### Tentative 10 : Mahalanobis cross-lot NB228 → DB (ÉCHEC)
- Part 85 (lot 660864365) a soft_bin=45 (FAIL only) → absent de la DB filtrée
- Inclusion des FAIL dans l'espace de recherche (3753 parts uniques)
- **Résultat cross-lot: target rank ≈ 400-1500 selon config** → complètement raté
- Within-lot toujours 23/23 rank-1 (le retrait du soft_bin ne change rien)
- Diagnostic: la normalisation z-score utilise des stats lot différentes pour NB vs DB
  → l'empreinte du même die est distordue par le changement de référence

### Tentative 11 : Trois normalisations alternatives (AMÉLIORATION PARTIELLE)
- **Percentile intra-lot** : rang de chaque feature converti en percentile (0-1) dans le lot
- **Center-only** : soustraire la moyenne du lot, NE PAS diviser par le std
- **Z-score** : baseline (soustraire mean, diviser par std)

#### Within-lot (les 3 méthodes atteignent 23/23):
| Méthode     | top-50 | top-100 |
|-------------|--------|---------|
| Percentile  | 23/23  | 23/23   |
| Center-only | 19/23  | 23/23   |
| Z-score     | 21/23  | 23/23   |

#### Cross-lot NB228 → DB(660864365, 85) — meilleur rang obtenu:
| Méthode     | Best rank | Config        |
|-------------|-----------|---------------|
| **Percentile**  | **175**   | top-50, α=0.5 |
| Center-only | 394       | top-100, α=0.5|
| Z-score     | 414       | top-100, α=0.0|

- Le percentile divise la distance au target par ~2.4× vs z-score → direction correcte
- Mais rank 175/3753 reste insuffisant pour une détection fiable
- Note : avec top-100, α=0.0, le best match pour NB228 est lot=660864365 **part 84** (!), la voisine directe du part 85 → signal géographique potentiel

### Analyse du blocage cross-lot
- Le bruit de normalisation inter-lot domine le signal d'identité du die
- Les 23 paires d'entraînement sont toutes within-lot → la covariance ne capture pas le bruit cross-lot
- Piste : il faudrait estimer le bruit *cross-lot* (shift de normalisation entre lots)

### Tentative 12 : Features lot-invariantes, ratios, combinaisons (PROGRÈS ✓)

#### Analyse ICC (Intra-Class Correlation)
- ICC = var_between_lots / (var_between + var_within) → mesure l'effet lot
- ICC ≈ 0 → feature lot-invariante (variance interne au lot)
- ICC ≈ 1 → feature dominée par le lot

**Most lot-invariant (ICC ≈ 0):**
| Feature | ICC |
|---------|-----|
| SDAMaxCorA | 0.003 |
| A_DNLmaxcor100M2G5 | 0.006 |
| SDA_RangeA | 0.007 |
| Dcor_THD100M2G5 | 0.007 |

**Most lot-dominated (ICC ≈ 1):**
| Feature | ICC |
|---------|-----|
| RES_CAL | 0.999 |
| B_LvlRef_BW1_1750M | 0.843 |
| SCLK_LO | 0.797 |

→ Les meilleures features SNR (SDAMaxCorA, SDA_RangeA) sont AUSSI les plus lot-invariantes !

#### A) Lot-invariant (raw values, no normalization) + Mahalanobis
- **ICC < 0.5 (254 features), top-30, α=0.5: target rank = 19 ✓✓**
- ICC < 0.5, top-10, α=0.0: rank = 28
- ICC < 0.3 (198 features), top-100: rank = 67 (mais distances NaN → matrice singulière)
- Top-5 matches : lot=660864365 part=234 (bon lot!), part=62, part=223
- **Utiliser les valeurs brutes sans normalisation fonctionne mieux que toute normalisation !**

#### B) Percentile + lot-invariant
- Percentile + ICC<0.3: rank = 506 → pire que percentile seul
- Percentile ALL (baseline): rank = 175 → identique à tentative 11
- Le percentile détruit l'information de distance absolue que les features lot-invariantes préservent

#### C) Raw log-ratios (feature_A / feature_B)
- 435 ratios formés à partir des top-30 features z-SNR
- Ratios ICC très bas (mean=0.077, median=0.042, 404/435 < 0.3) → lot-invariants comme attendu
- **Mais rank = 908-1058** → totalement inefficace
- Les ratios SNR les plus élevés (335, 274...) sont des comparaisons inter-cores → trop stables, pas discriminants

#### Weighted Euclidean simple (sans Mahalanobis)
- ICC<0.5, top-10: rank = 340 → Mahalanobis reste crucial

#### Conclusion tentative 12
- **Meilleur résultat global: rank 19/3753 (top 0.5%)**
- L'approche gagnante: features lot-invariantes ICC<0.5, valeurs **RAW** (pas de normalisation), Mahalanobis avec forte régularisation (α=0.5)
- La normalisation (z-score OU percentile) DÉTRUIT le signal : les features clés ont déjà un faible effet lot, pas besoin de normaliser
- Prochaine étape: affiner la sélection de features et la régularisation pour descendre en top-5

### À faire
- Grille plus fine autour de ICC<0.4-0.6, top-20-40, α=0.3-0.7
- Tester retrait des features avec trop de NaN dans le lot cible
- Considérer normalisation hybride: raw pour lot-invariant + percentile pour le reste

### Tentative 13 : Grille fine (PROGRÈS ✓✓)

#### NaN discovery
- **Part 85 (target) a 100 features NaN** (c'est un FAIL, soft_bin=45 → tests incomplets)
- NB228 a 0 NaN, le lot cible 660864365 a 0% NaN moyen
- Les 100 features NaN du target sont remplies par 0 → handicap pour la distance

#### Grid fine: ICC × top_n × alpha (avec exclusion des features NaN)
- 5 ICC thresholds × 11 top_n × 11 alpha = 605 configs testées
- **Best: ICC<0.6, top-40, α=1.0 → target rank = 11**
- Multiple configs à rank 11-12 : robuste

#### Top 3 configs:
| ICC  | top_n | α    | Rank | Best match         |
|------|-------|------|------|--------------------|
| 0.6  | 40    | 1.00 | 11   | 661068643, part 4  |
| 0.6  | 40    | 0.90 | 11   | 660864365, part 234|
| 0.6  | 40    | 0.80 | 11   | 660864365, part 234|

#### Détail best config (top-25 matches NB228):
- 14/25 matches dans le **bon lot** (660864365)
- Part 85 (target) toujours en rank 11, d=3.28
- Rank 1-2 : part 234 (d=2.56) et lot 661068643 part 4 (d=2.53) — se disputent la 1ère place
- α=1.0 → Mahalanobis se réduit à distance Euclidienne pondérée (diag only) — pas besoin des corrélations cross-lot
- Les parts qui battent la cible sont des PASS du même lot → pas de NaN, signal plus propre

#### Blocage identifié
- La cible (FAIL) a 100/213 features = 0 au lieu de leur vraie valeur
- Les PASS du même lot n'ont pas ce problème → plus proches artificiellement
- **Ce n'est PAS un problème de méthode mais de données** : le part 85 est mal représenté

### À faire
- Imputer les NaN du part 85 par la médiane du lot (au lieu de 0) → devrait améliorer
- Tester si en excluant les 100 features NaN du part 85 on descend en top-5
- Envisager un score combiné distance + lot-consistency

### Tentative 14 : Imputation NaN par médiane du lot (NÉGATIF)
- Imputation des NaN du part 85 par la médiane du lot 660864365
- **Résultat: rank 28** (pire que rank 11 avec NaN→0)
- L'imputation pousse le die vers le centroïde du lot → moins distinguable
- Conclusion: les NaN remplacées par la médiane n'apportent aucune info d'identité, juste du bruit moyen
- En production avec 2 tests complets (0 NaN), ce problème n'existe pas
