

Détecte si des pièces (dies) d'un **NewBatch** ont déjà été testées dans une **Database** historique, en comparant leurs signatures électriques.

## Contexte

En production de semi-conducteurs, chaque pièce (die) passe une batterie de tests électriques (~300 mesures : courant, SFDR, THD, SNR, DNL, etc.). Un die frauduleusement réinjecté aura une signature très similaire à son test original, mais **dans un lot différent**.

Le problème : chaque lot de fabrication introduit un **effet lot** (décalage systématique de toutes les mesures). Deux dies différents du même lot se ressemblent plus que le même die testé dans deux lots différents. Il faut séparer l'identité du die de l'effet lot.

## Usage

```
python pipeline.py
```

Aucune configuration manuelle nécessaire — les hyperparamètres (ICC seuil, nombre de features, α) sont **automatiquement optimisés** par grid search sur les paires de retest de la Database.

## Pipeline (`pipeline.py`)

### Étape 1 — Chargement brut

Les CSV sont chargés sans filtrage (pas de filtre sur `soft_bin`). Format européen (`;` séparateur, `,` décimale).

- **Database** : ~3750 pièces, 22 lots
- **NewBatch** : ~100 pièces, 1 lot

### Étape 2 — Déduplication

Certains dies dans la Database ont été testés 2 fois (FAIL puis retest). On garde **la dernière ligne** par couple `(lot_id, part_id)` pour l'espace de recherche. Les 2 lignes sont conservées pour l'étape 3.

### Étape 3 — Estimation du bruit de retest

Les 23 paires de retest (même die, 2 lignes, réparties dans 12 lots) servent à mesurer le **bruit de mesure** : pour chaque feature, on calcule `diff = valeur_test1 - valeur_test2`. Ce vecteur de différences donne la matrice de covariance du bruit.

### Étape 4 — ICC (Intra-Class Correlation)

Pour chaque feature, on calcule :

$$ICC = \frac{\sigma^2_{entre\_lots}}{\sigma^2_{entre\_lots} + \sigma^2_{intra\_lot}}$$

- **ICC ≈ 0** → la feature varie surtout *à l'intérieur* du lot → **lot-invariante** (utile)
- **ICC ≈ 1** → la feature est dominée par l'effet lot → inutile pour le cross-lot

### Étape 5 — Entraînement automatique (grid search)

Un grid search explore toutes les combinaisons de 3 hyperparamètres :

| Paramètre | Grille | Rôle |
|-----------|--------|------|
| ICC seuil | 0.3 — 0.8 | Filtre les features lot-invariantes |
| top_n | 10 — 100 | Nombre de features retenues (triées par SNR) |
| α | 0.0 — 1.0 | Shrinkage : 0 = Mahalanobis complète, 1 = diagonale pure |

**Métrique d'évaluation — rang cross-lot** : pour chaque paire de retest, on mesure combien de parts **d'autres lots** sont plus proches que la cible. Cela élimine l'avantage within-lot et sélectionne des configs robustes au cas cross-lot (le vrai cas d'usage).

**Critère de sélection** (par priorité) :
1. Max rang-1 cross-lot
2. Min rang médian cross-lot
3. Max α (régularisation = robustesse)
4. Min top_n (parcimonie)

### Étape 6 — Sélection de features

Avec la config optimale trouvée, on filtre les features avec **ICC < seuil**, puis on trie par **SNR** décroissant :

$$SNR = \frac{\sigma_{global}}{\sigma_{retest}}$$

Top features habituelles : `SDA_RangeA` (SNR=11.8, ICC=0.007), `SDAMaxCorA` (SNR=7.5, ICC=0.003), `VOL_max` (SNR=2.4, ICC=0.035).

### Étape 7 — Distance de Mahalanobis

Pour chaque pièce du NewBatch, on calcule la distance vers **chaque pièce de la Database** :

$$d(a, b) = \sqrt{(x_a - x_b)^T \, \Sigma^{-1} \, (x_a - x_b)}$$

où $\Sigma$ est la matrice de covariance du bruit de retest, régularisée par α (shrinkage vers la diagonale).

**Point clé : on utilise les valeurs brutes, sans normalisation par lot.** Les features sélectionnées (ICC faible) ont naturellement un faible effet lot, donc la normalisation est inutile — et on a montré qu'elle dégrade les résultats.

### Étape 8 — Scoring

Pour chaque pièce NB, on produit :
- **Best match** : la pièce DB la plus proche (distance $d_1$)
- **Gap ratio** : $d_2 / d_1$ — ratio entre le 2ème et le 1er plus proche voisin

Un **gap élevé** signifie que le meilleur match est nettement plus proche que le suivant → forte suspicion de doublon.

## Résultats

### Validation within-lot (23 paires connues)

Quand les 2 tests d'un même die sont dans le **même lot**, la méthode retrouve **23/23 paires en rank-1** avec la distance de Mahalanobis complète.

### Performance cross-lot (NB → DB)

Le cas réel est plus difficile : le NewBatch est un lot différent de ceux de la Database. Avec la config manuellement calibrée (ICC<0.6, top-40, α=1.0) :

- **Ground truth** : NB part 228 = DB lot 660864365, part 85
- **Résultat** : la cible est en **rank 11** sur 3753 pièces (top 0.3%)
- 14 des 25 meilleurs matches sont dans le **bon lot**

Le rank n'est pas 1 car :
1. Le **part 85 est un FAIL** (soft_bin=45) → 100 features sur 313 sont NaN (tests non complétés). En production réelle avec 2 tests complets, ce handicap disparaît.
2. Les **voisins géographiques sur le wafer** (parts 234, 62, 178...) ont des signatures proches car ils sont physiquement adjacents.

### Pourquoi ça marche

1. **Features lot-invariantes** : les mesures SDA, DNL, ENOB à 100 MHz varient naturellement peu entre lots (ICC < 0.01). Elles capturent les propriétés physiques intrinsèques du die.
2. **Pas de normalisation** : la normalisation par lot (z-score) divise par le std du lot, ce qui injecte du bruit dans des features déjà stables. En raw, on préserve la distance absolue.
3. **Pondération par bruit de retest** : les 23 paires de retest calibrent la métrique — les features à faible bruit de mesure pèsent plus.

## Structure

```
config.py       — chemins et colonnes ID
pipeline.py     — pipeline complet (chargement + entraînement + inférence)
JOURNAL.md      — historique détaillé de toutes les tentatives
outputs/        — résultats CSV
```

## Dépendances

```
numpy
pandas
```
