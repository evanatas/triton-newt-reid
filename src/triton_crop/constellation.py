"""Матчинг созвездия пятен (Блок 5, шаги 5.4–5.5) — ЧИСТАЯ геометрия (numpy).

Созвездие = набор центроидов пятен. Сопоставление двух созвездий инвариантно к **similarity**
(сдвиг+масштаб+поворот), но **НЕ к зеркалу** (зеркальный тритон = ДРУГАЯ особь). Это даёт:
  • score = ДОЛЯ совпавших пятен (честный «% совпадения»);
  • matched_pairs — список совпавших (idx_a, idx_b) для ОВЕРЛЕЯ (запрос заказчика №1).

Матчер ПОДКЛЮЧАЕМЫЙ (cfg.match_method), выбор лучшего — честным A/B на dev:
  • 'ransac' (дефолт) — RANSAC по 2-точечным выборкам: оценка proper-similarity (БЕЗ отражения) →
    подсчёт inlier-соответствий (взаимные ближайшие в допуске). Устойчив к over-detection (лишние
    пятна = outliers), встроенно отвергает зеркало (proper similarity не умеет отражать);
  • 'nn' — дешёвый baseline: нормализация облака (центр+масштаб+ориентация) + взаимный ближайший сосед.

estimate_similarity_no_reflection — proper similarity scale·R+t из ≥2 пар (форма [[a,-b],[b,a]],
det=a²+b²>0 → отражение невозможно по построению; зеркало → большой остаток).
"""
import numpy as np

from .spots import spots_to_array


def _as_pts(x) -> np.ndarray:
    """list[Spot] | list[(x,y)] | (N,2) array -> (N,2) float."""
    if isinstance(x, np.ndarray):
        return np.asarray(x, float).reshape(-1, 2) if x.size else np.zeros((0, 2), float)
    if len(x) == 0:
        return np.zeros((0, 2), float)
    if hasattr(x[0], "x") and hasattr(x[0], "y"):       # list[Spot]
        return spots_to_array(x)
    return np.asarray(x, float).reshape(-1, 2)


def apply_similarity(M, pts) -> np.ndarray:
    """Применить 2×3 similarity к точкам (N,2). -> (N,2)."""
    pts = np.asarray(pts, float)
    return pts @ M[:, :2].T + M[:, 2]


def estimate_similarity_no_reflection(src, dst):
    """Proper similarity (scale>0 · поворот + сдвиг, БЕЗ отражения) из ≥2 пар (least squares).
    Форма [[a,-b],[b,a]] гарантирует det>0. -> 2×3 матрица или None (вырождено/<2 пар/несовпадение длин).
    Референс-реализация (тесты + докстринг-контракт); боевой путь (guided/ransac) использует
    векторизованный эквивалент той же формулы в _best_hypothesis (для 2 точек least-squares = точное
    решение)."""
    src = _as_pts(src); dst = _as_pts(dst)
    if len(src) < 2 or len(dst) < 2 or len(src) != len(dst):
        return None
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    denom = float((sc ** 2).sum())
    if denom < 1e-12:
        return None
    a = float((sc[:, 0] * dc[:, 0] + sc[:, 1] * dc[:, 1]).sum()) / denom
    b = float((sc[:, 0] * dc[:, 1] - sc[:, 1] * dc[:, 0]).sum()) / denom
    if a * a + b * b < 1e-12:
        return None
    R = np.array([[a, -b], [b, a]])
    t = mu_d - R @ mu_s
    return np.array([[a, -b, t[0]], [b, a, t[1]]])


def _rms_scale(pts) -> float:
    if len(pts) == 0:
        return 0.0
    c = pts - pts.mean(0)
    return float(np.sqrt((c ** 2).sum(1).mean()))


def normalize_constellation(pts, axis_start=None, axis_end=None) -> np.ndarray:
    """Привести облако к similarity-инвариантному виду (центр 0, RMS-радиус 1, ориентация по оси/PCA),
    БЕЗ отражения. Для NN-матчера. -> (N,2)."""
    pts = _as_pts(pts)
    if len(pts) == 0:
        return pts
    c = pts - pts.mean(0)
    s = _rms_scale(pts)
    if s < 1e-9:
        return c
    c = c / s
    if axis_start is not None and axis_end is not None:
        v = np.asarray(axis_end, float) - np.asarray(axis_start, float)
    else:
        w, V = np.linalg.eigh(c.T @ c)          # PCA: главная ось — собств. вектор макс. собств. значения
        v = V[:, int(np.argmax(w))]
    th = (np.pi / 2.0) - np.arctan2(v[1], v[0])  # повернуть так, чтобы ось стала вертикалью
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    c = c @ R.T
    if float(np.mean(c[:, 1] ** 3)) < 0:         # sign-fix 180° (поворот, НЕ отражение: det=+1)
        c = c @ np.array([[-1.0, 0.0], [0.0, -1.0]])
    return c


def _mutual_nn(A, B, tol):
    """Взаимные ближайшие соседи A↔B в допуске tol. -> list[(i,j)] (каждый B — максимум раз)."""
    if len(A) == 0 or len(B) == 0:
        return []
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    a2b = D.argmin(1)
    b2a = D.argmin(0)
    pairs, used = [], set()
    for i, j in enumerate(a2b):
        j = int(j)
        if b2a[j] == i and D[i, j] <= tol and j not in used:
            pairs.append((i, j)); used.add(j)
    return pairs


def _local_descriptor(pts, knn):
    """Для каждого пятна — similarity-инвариантный (масштаб+поворот, БЕЗ зеркала) дескриптор по knn
    ближайшим соседям. Векторы к соседям нормируются на медианное расстояние (масштаб-инвариант) и
    поворачиваются на circular-mean угол ВСЕХ knn-соседей (поворот-инвариант, устойчив к near-tie
    перестановке). Угол → (cos, sin) (без wrap). Зеркало меняет знак sin → дескриптор
    отличается. -> (N, 3*knn) float.
    Ограничение: при N-1 < knn свободные слоты остаются нулями (zero-padding) → при сравнении
    созвездий разного размера у knn-границы евклидово расстояние получает спурьёзную компоненту
    (реальные значения vs нули). Полная информативность — при N >= knn+1."""
    pts = _as_pts(pts)
    N = len(pts)
    out = np.zeros((N, 3 * knn), float)
    if N == 0:
        return out
    D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    for i in range(N):
        order = [j for j in np.argsort(D[i]) if j != i][:knn]
        if not order:
            continue
        vecs = pts[order] - pts[i]
        dists = np.linalg.norm(vecs, axis=1)
        med = float(np.median(dists))
        med = med if med > 1e-9 else 1.0
        ang = np.arctan2(vecs[:, 1], vecs[:, 0])
        ang0 = float(np.arctan2(np.sin(ang).sum(), np.cos(ang).sum()))   # circular-mean соседей: устойчив к
        th = ang - ang0                                      # near-tie перестановке; инвариант сохранён
        k = len(order)
        out[i, :k] = dists / med
        out[i, knn:knn + k] = np.cos(th)
        out[i, 2 * knn:2 * knn + k] = np.sin(th)
    return out


def _candidate_correspondences(A, B, cfg):
    """Кандидатные соответствия (i_probe, j_gallery): для каждого i — match_knn ближайших j по
    дескриптор-расстоянию. -> list[(i, j)]. Дескриптор инвариантен к similarity → истинные пары всплывают.
    Для малых/разноразмерных созвездий качество кандидатов снижено zero-padding'ом дескриптора
    (см. _local_descriptor); ошибка ослабляется inlier-фильтрацией в _best_hypothesis."""
    A = _as_pts(A); B = _as_pts(B)
    knn = int(getattr(cfg, "descriptor_knn", 5))
    da = _local_descriptor(A, knn)
    db = _local_descriptor(B, knn)
    if len(da) == 0 or len(db) == 0:
        return []
    Dd = np.linalg.norm(da[:, None, :] - db[None, :, :], axis=2)   # (nA, nB)
    take = min(int(getattr(cfg, "match_knn", 4)), len(B))
    cands = []
    for i in range(len(A)):
        for j in np.argsort(Dd[i])[:take]:
            cands.append((i, int(j)))
    return cands


def _best_hypothesis(A, B, seeds, tol):
    """ВЕКТОРИЗОВАННЫЙ выбор лучшей 2-точечной similarity-гипотезы (БЕЗ зеркала) среди seeds (H×4:
    [i1,i2,j1,j2]). Для каждой гипотезы — closed-form similarity (s1→d1, s2→d2; форма [[a,-b],[b,a]],
    det>0 → отражение невозможно), затем inliers = ВЗАИМНЫЕ ближайшие A↔B в tol. best по
    (число inliers, затем min RMS остаток). Чистый numpy, чанк по гипотезам (память). -> list[(i,j)].
    Та же 2-точечная proper-similarity, что в estimate_similarity_no_reflection, выведенная инлайн
    через комплексное отношение z=(d2-d1)/(s2-s1) ради векторизации; эквивалентность закреплена тестом."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    seeds = np.asarray(seeds, int)
    if seeds.size == 0:
        return []
    nA = len(A)
    Bx, By = B[:, 0], B[:, 1]
    best_n, best_res, best_inl, best_a2b = -1, np.inf, None, None
    tol2 = float(tol) * float(tol)
    CH = 4096
    for c0 in range(0, len(seeds), CH):
        sd = seeds[c0:c0 + CH]
        s1 = A[sd[:, 0]]; s2 = A[sd[:, 1]]; d1 = B[sd[:, 2]]; d2 = B[sd[:, 3]]
        cs = (s2[:, 0] - s1[:, 0]) + 1j * (s2[:, 1] - s1[:, 1])
        cd = (d2[:, 0] - d1[:, 0]) + 1j * (d2[:, 1] - d1[:, 1])
        ok = np.abs(cs) > 1e-9
        z = np.zeros(len(cs), complex)
        z[ok] = cd[ok] / cs[ok]
        a = z.real; b = z.imag                                   # R=[[a,-b],[b,a]] (масштаб·поворот, det>0)
        tx = d1[:, 0] - (a * s1[:, 0] - b * s1[:, 1])
        ty = d1[:, 1] - (b * s1[:, 0] + a * s1[:, 1])
        Ax = A[:, 0][None, :]; Ay = A[:, 1][None, :]             # (1,nA)
        Atx = a[:, None] * Ax - b[:, None] * Ay + tx[:, None]    # (h,nA)
        Aty = b[:, None] * Ax + a[:, None] * Ay + ty[:, None]
        dx = Atx[:, :, None] - Bx[None, None, :]                 # (h,nA,nB)
        dy = Aty[:, :, None] - By[None, None, :]
        D2 = dx * dx + dy * dy
        a2b = D2.argmin(axis=2)                                  # (h,nA) ближайший B для каждого A
        b2a = D2.argmin(axis=1)                                  # (h,nB) ближайший A для каждого B
        h_idx = np.arange(len(sd))[:, None]
        mutual = b2a[h_idx, a2b] == np.arange(nA)[None, :]       # взаимность A↔B
        mind2 = np.take_along_axis(D2, a2b[:, :, None], axis=2)[:, :, 0]
        inl = mutual & (mind2 <= tol2) & ok[:, None]             # (h,nA) inlier-маска
        ncount = inl.sum(axis=1)
        res2 = np.where(inl, mind2, 0.0).sum(1) / np.maximum(ncount, 1)
        h0 = int(np.lexsort((res2, -ncount))[0])                 # max inliers, тай-брейк min остаток
        if (ncount[h0] > best_n) or (ncount[h0] == best_n and res2[h0] < best_res):
            best_n, best_res = int(ncount[h0]), float(res2[h0])
            best_inl, best_a2b = inl[h0], a2b[h0]
    if best_inl is None or best_n < 2:
        return []
    return [(int(n), int(best_a2b[n])) for n in np.where(best_inl)[0]]


def _match_guided(A, B, cfg):
    """Канонический матчер: кандидаты по дескриптору → ИСЧЕРПЫВАЮЩИЙ перебор пар кандидатов (векторизовано)
    → лучшая similarity-гипотеза. Детерминирован, без зависимости от числа итераций (устраняет
    iteration-starvation случайного RANSAC). Использует _best_hypothesis (numpy-батч)."""
    tol = cfg.match_inlier_tol * (_rms_scale(B) or 1.0)
    cands = _candidate_correspondences(A, B, cfg)
    if len(cands) < 2:
        return []
    ci = np.array([c[0] for c in cands]); cj = np.array([c[1] for c in cands])
    aa, bb = np.triu_indices(len(cands), k=1)                    # все пары кандидатов (a<b)
    i1, i2, j1, j2 = ci[aa], ci[bb], cj[aa], cj[bb]
    valid = (i1 != i2) & (j1 != j2)                              # разные probe И разные gallery
    seeds = np.stack([i1[valid], i2[valid], j1[valid], j2[valid]], axis=1)
    return _best_hypothesis(A, B, seeds, tol)


def _match_ransac(A, B, cfg):
    """Случайный 2-точечный RANSAC (сохранён для sensitivity-сравнения vs guided). Векторизован через
    _best_hypothesis; обе ориентации пары gallery. Детерминирован (seed)."""
    nA, nB = len(A), len(B)
    tol = cfg.match_inlier_tol * (_rms_scale(B) or 1.0)
    rng = np.random.RandomState(cfg.seed)
    H = int(cfg.ransac_iters)
    ii = np.array([rng.choice(nA, 2, replace=False) for _ in range(H)])
    jj = np.array([rng.choice(nB, 2, replace=False) for _ in range(H)])
    seeds = np.concatenate([
        np.stack([ii[:, 0], ii[:, 1], jj[:, 0], jj[:, 1]], axis=1),
        np.stack([ii[:, 0], ii[:, 1], jj[:, 1], jj[:, 0]], axis=1),   # вторая ориентация gallery-пары
    ], axis=0)
    return _best_hypothesis(A, B, seeds, tol)


def _match_nn(A, B, cfg):
    na, nb = normalize_constellation(A), normalize_constellation(B)
    tol = cfg.match_inlier_tol * 3.0                 # на нормированном облаке (RMS=1) допуск крупнее
    best = []
    for flip in (False, True):                       # 180° (proper-поворот, НЕ зеркало) — снять ось-неоднозначность
        nbb = nb @ np.array([[-1.0, 0.0], [0.0, -1.0]]) if flip else nb
        pairs = _mutual_nn(na, nbb, tol)
        if len(pairs) > len(best):
            best = pairs
    return best


def _score(n_in, nA, nB, norm):
    if norm == "max":
        denom = max(nA, nB)                      # доля БОЛЬШЕГО созвездия — устойчиво к спурьёзным мелким
    elif norm == "min":
        denom = min(nA, nB)
    elif norm == "jaccard":
        denom = nA + nB - n_in
    else:
        denom = (nA + nB) / 2.0
    return float(n_in / denom) if denom > 0 else 0.0


def match_constellations(spots_a, spots_b, cfg):
    """Сопоставить два созвездия → (score ∈ [0,1] = доля совпавших, matched_pairs[(idx_a, idx_b)]).
    method ∈ {guided, ransac, nn}. < min_spots_for_match пятен с любой стороны → (0.0, []). Детерминирован."""
    A, B = _as_pts(spots_a), _as_pts(spots_b)
    if len(A) < cfg.min_spots_for_match or len(B) < cfg.min_spots_for_match:
        return 0.0, []
    if cfg.match_method == "nn":
        pairs = _match_nn(A, B, cfg)
    elif cfg.match_method == "guided":
        pairs = _match_guided(A, B, cfg)
    elif cfg.match_method == "ransac":
        pairs = _match_ransac(A, B, cfg)
    else:
        raise ValueError(f"match_constellations: неизвестный match_method={cfg.match_method!r} (guided|ransac|nn)")
    if len(pairs) < getattr(cfg, "min_inliers", 1):    # коинцидентное 2-3-точечное выравнивание ≠ матч
        return 0.0, []                                 # score=0 ⇒ матча нет ⇒ оверлею нечего рисовать (согласованность)
    return _score(len(pairs), len(A), len(B), cfg.score_norm), pairs


def build_match_sim_matrix(probe_spots, gallery_spots, cfg) -> np.ndarray:
    """Матрица score'ов матчера (n_probe × n_gallery): sim[i,j] = match_constellations(probe_i, gallery_j).
    Вход для spot_ab (как эмбеддер даёт probe@gallery.T). probe/gallery — одним путём. Детерм."""
    P, G = len(probe_spots), len(gallery_spots)
    sim = np.zeros((P, G), float)
    for i in range(P):
        for j in range(G):
            sim[i, j] = match_constellations(probe_spots[i], gallery_spots[j], cfg)[0]
    return sim
