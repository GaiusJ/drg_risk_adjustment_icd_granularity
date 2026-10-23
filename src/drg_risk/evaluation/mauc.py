"""
Implementierung der Multiclass AUROC (OvO, ordinal) nach Xu (2024).
Quelle: https://www.sciencedirect.com/science/article/pii/S0925231224002911
"""

import numpy as np

def _fun_auc (ratings1, ratings2) :
  """
  Funktion zum Berechnen der Area under the ROC Curve.
  Entnommen von: Comparing multi-class classifier performance by multi-class ROC analysis: A nonparametric approach, Jingyan Xu 2024, https://github.com/jingyanxu/mauc/blob/main/utils/utils_auc.py: https://www.sciencedirect.com/science/article/pii/S0925231224002911
  """

  n1 = len(ratings1)
  n2 = len(ratings2)

  ratings1_sorted = np.sort(ratings1)
  ratings2_sorted = np.sort(ratings2)

  # beide Seiten für Gleichstände
  postmat1_l = np.searchsorted(ratings2_sorted, ratings1, side = "left")
  postmat1_r = np.searchsorted(ratings2_sorted, ratings1, side = "right")

  postmat2_l = np.searchsorted(ratings1_sorted, ratings2, side = "left")
  postmat2_r = np.searchsorted(ratings1_sorted, ratings2, side = "right")

  # comp1: Rang von ratings1 relativ zu ratings2 (Placement-Werte für AUC)
  # comp2: Rang von ratings2 relativ zu ratings1 (symmetrisches Gegenstück)
  comp1 = (postmat1_l + postmat1_r) / 2. / n2
  comp2 = 1.0 - ((postmat2_l + postmat2_r) / 2. / n1)

  auc_slow = np.mean(comp1)

  # Varianzschätzung der AUC nach DeLong-Methode (aus den Placement-Werten comp1/comp2)
  auc_var = np.sum ((comp1 - auc_slow)**2/(n1-1)/n1) +np.sum ((comp2 - auc_slow)**2/(n2-1)/n2)
  auc_std = np.sqrt (auc_var)

  return auc_slow,  auc_std, comp1, comp2

def fun_mauc_ovo_ordinal(y_score_all, labels):
  """
  Funktion zum Berechnen der Multiclass AUROC sowie der Standardabweichung.
  Verwendet den one-vs-one Ansatz.
  """

  # 7 ordinale Verweildauerkategorien entsprechend der Campus-File-Kodierung (typ_vwd 1-7)
  n_classes = 7

  threshold_aucs = []
  threshold_variances = []

  # One-vs-One (OvO) Ansatz, über alle Klassen iterieren
  for k in range(n_classes - 1):
    # binäres Label, ob Label größer als bisherige Klasse
    binary_labels = (labels > k).astype(int)

    if len(np.unique(binary_labels)) < 2:
      continue

    # Kumulierte Score-Wahrscheinlichkeit für alle Klassen > k (verwendet ordinale Struktur)
    binary_scores = np.sum(y_score_all[:, k + 1:], axis = 1)

    scores_class0 = binary_scores[binary_labels == 0]
    scores_class1 = binary_scores[binary_labels == 1]

    # AUROC und Standardabweichung berechnen
    auc_val, auc_std, _, _ = _fun_auc(scores_class1, scores_class0)

    # Werte speichern
    threshold_aucs.append(auc_val)
    threshold_variances.append(auc_std**2)

  if not threshold_aucs:
    raise ValueError("Berechnung fehlgeschlagen. Zu wenig Klassen vorhanden.")

  # Mittelwerte berechnen
  ordinal_mauc = np.mean(threshold_aucs)
  # Varianz von AUROC ist die Summe der Varianzen geteilt durch die Anzahl der Schwellenwerte
  ordinal_var = np.mean(threshold_variances) / len(threshold_variances)
  ordinal_std = np.sqrt(ordinal_var)

  return ordinal_mauc, ordinal_std