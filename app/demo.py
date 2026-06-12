"""Streamlit-демо «NewtID» — браузерное опознание тритона по узору брюшка (финализация ВКР).

Концепция как у существующих систем фотоидентификации: загрузка/выбор фото → top-K похожих ОСОБЕЙ карточками
(миниатюра + номер + уверенность % + бейдж «Лучшее совпадение») → решение known/new.
ДИФФЕРЕНЦИАТОР (запрос заказчика №1): side-by-side ОВЕРЛЕЙ совпавших пятен (матчинг созвездия центроидов).

Запуск ЛОКАЛЬНО (offline, ноутбук заказчика без интернета), из корня репозитория:
    pip install -e ".[crop]" streamlit
    streamlit run app/demo.py
→ откроется в браузере (http://localhost:8501).

Логика опознания — в `triton_crop.demo_backend` (тестируется). Здесь только UI-слой.
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from triton_crop.config import load_crop_config, load_spot_config  # noqa: E402
from triton_crop.demo_backend import (  # noqa: E402
    calibration_range, crop_from_raw, embed_crop, filter_scope, known_new_verdict, load_gallery, load_headline,
    load_heldout, load_rgb, rank_individuals, sample_probes, spot_overlay,
)

_OOF = _REPO / "artifacts" / "embed" / "oof"
_HEADLINE = _REPO / "artifacts" / "ab_test_headline.json"   # финальные sealed-числа ВКР
_HELDOUT = _REPO / "artifacts" / "embed" / "heldout"     # нетронутые test/open_test (после sealed-test)
_CROPS = _REPO / "crops_belly"
_SEG_W = _REPO / "artifacts" / "runs" / "belly_seg" / "weights" / "best.pt"
_POSE_W = _REPO / "artifacts" / "runs" / "belly_pose" / "weights" / "best.pt"
_VARIANT = "belly_oriented"

st.set_page_config(page_title="NewtID — re-ID тритонов по узору брюшка", page_icon="🦎", layout="wide")


# ─────────────── кэш ресурсов ───────────────
@st.cache_resource(show_spinner="Загрузка галереи особей…")
def _gallery():
    g = load_gallery(_OOF, _CROPS, _VARIANT)
    lo, hi = calibration_range(g)
    return g, lo, hi


@st.cache_resource(show_spinner="Загрузка проб (dev)…")
def _probes():
    return sample_probes(_OOF, _CROPS, _VARIANT)


def _heldout_available():
    return (_HELDOUT / f"{_VARIANT}.npy").exists()


@st.cache_resource(show_spinner="Загрузка held-out (sealed-test)…")
def _heldout():
    return load_heldout(_HELDOUT, _CROPS, _VARIANT)


def _device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


# ─────────────── шапка ───────────────
st.title("🦎 NewtID — реидентификация тритонов по узору брюшка")
st.caption("ВКР «Прикладной ИИ» (ТГУ) · заказчик ИПЭЭ РАН · open-set fine-grained re-ID. "
           "Эмбеддер (MegaDescriptor) даёт top-K, матчинг созвездия пятен — интерпретируемый оверлей. Offline, локально.")

gallery, CAL_LO, CAL_HI = _gallery()
tab_id, tab_ind, tab_about = st.tabs(["🔎 Опознание", "🗂 Особи (галерея)", "ℹ️ О системе"])

# ═══════════════ ВКЛАДКА: ОПОЗНАНИЕ ═══════════════
with tab_id:
    c_ctrl, c_query = st.columns([2, 1])
    with c_ctrl:
        source = "Загрузить фото"
        scope = st.selectbox("Область поиска", ["Вся база", "TK", "PW"], index=0)
        topk = st.slider("Сколько кандидатов показать (top-K)", 3, 10, 7)
    gal = filter_scope(gallery, scope)

    query_emb = None
    query_rgb = None
    true_id = None
    gt_is_new = None        # ground-truth known/new (только для held-out: открытая проверка вердикта)

    if source.startswith("Выбрать"):
        probes = _probes()
        if scope != "Вся база":
            filt = [p for p in probes if p["cohort"] == scope]
            if filt:
                probes = filt
            else:                            # пустая когорта → не сбрасывать фильтр молча
                st.info(f"Для вида {scope} нет dev-проб — показаны все (галерея при этом срезана по {scope}).")
        labels = [f"{p['cohort']} · {p['individual_id']} · {p['md5'][:8]}" for p in probes]
        i = st.selectbox("Проба (известная особь — проверяем, найдёт ли система)",
                         range(len(probes)), format_func=lambda k: labels[k])
        p = probes[i]
        query_emb, query_rgb, true_id = p["emb"], load_rgb(p["crop_path"]), p["individual_id"]
    elif source.startswith("Held-out"):
        held = _heldout()
        if scope != "Вся база":
            filt = [p for p in held if p["cohort"] == scope]
            if filt:
                held = filt
            else:                            # пустая когорта → не сбрасывать фильтр молча
                st.info(f"Для вида {scope} нет held-out проб — показаны все.")
        # known-first: сначала известные особи (есть в базе), затем явные НОВЫЕ — чтобы быстро показать обе ветки
        held = sorted(held, key=lambda p: (bool(p["is_new"]), str(p["cohort"]), str(p["individual_id"])))
        labels = [f"{'НОВАЯ особь' if p['is_new'] else p['individual_id']} · {p['cohort']} · {p['md5'][:8]}"
                  for p in held]
        i = st.selectbox("Held-out проба (нетронутый test/open_test — финальная честная проверка)",
                         range(len(held)), format_func=lambda k: labels[k])
        p = held[i]
        gt_is_new = bool(p["is_new"])
        query_emb, query_rgb = p["emb"], load_rgb(p["crop_path"])
        true_id = None if gt_is_new else p["individual_id"]
    else:
        if not (_SEG_W.exists() and _POSE_W.exists()):       # preflight: честно предупредить ДО загрузки
            st.info(f"Сегментация требует весов YOLO ({_SEG_W.name}/{_POSE_W.name}) — их нет в публичном репозитории. "
                    "Отметьте «это уже кроп брюшка» (без сегментации) или используйте «Выбрать пробу»/«Held-out».")
        up = st.file_uploader("Фото брюшка (JPG/PNG)", type=["jpg", "jpeg", "png"])
        is_crop = st.checkbox("Это уже готовый кроп брюшка (не запускать сегментацию)", value=False)
        if up is not None:
            import hashlib

            import cv2
            data = up.read()
            key = hashlib.md5(data).hexdigest() + ("_crop" if is_crop else "_seg")   # кэш по содержимому+режиму
            cache = st.session_state.get("_upload_cache")
            if cache and cache.get("key") == key:            # тот же файл/режим → не пересегментировать/переэмбеддить
                query_rgb, query_emb = cache["rgb"], cache["emb"]
            else:
                # imdecode возвращает None на битом/не-изображении (фильтр type проверяет только расширение)
                buf = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if buf is None:
                    st.error("Не удалось прочитать файл изображения — он повреждён или не является настоящим JPG/PNG.")
                else:
                    raw = cv2.cvtColor(buf, cv2.COLOR_BGR2RGB)
                    try:
                        if is_crop:
                            query_rgb = raw
                        else:
                            with st.spinner("Сегментация + кроп брюшка (YOLO)…"):
                                query_rgb = crop_from_raw(raw, load_crop_config(), _SEG_W, _POSE_W, _device())
                        if query_rgb is None:
                            st.error("Не удалось выделить брюшко. Попробуйте отметить «это уже кроп брюшка».")
                        else:
                            with st.spinner("Эмбеддинг (MegaDescriptor, первая загрузка весов может занять время)…"):
                                query_emb = embed_crop(query_rgb, device=_device())
                        st.session_state["_upload_cache"] = {"key": key, "rgb": query_rgb, "emb": query_emb}
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Инференс недоступен ({type(e).__name__}: {e}). "
                                 f"Нужны веса YOLO ({_SEG_W.name}/{_POSE_W.name}) и доступ к MegaDescriptor (timm). "
                                 f"Для надёжного демо используйте режим «Выбрать пробу».")

    with c_query:
        if query_rgb is not None:
            st.image(query_rgb, caption="Запрос (кроп брюшка)" + (f" · истинно {true_id}" if true_id else ""),
                     width="stretch")

    if query_emb is not None and len(gal.ids):
        ranked = rank_individuals(query_emb, gal, topk=topk, lo=CAL_LO, hi=CAL_HI)
        verdict = known_new_verdict(ranked)
        if verdict["verdict"] == "known":
            st.success(f"✅ Особь в базе: **{ranked[0]['individual_id']}** · уверенность **{verdict['confidence']:.0f}%** "
                       f"(отрыв от №2: {verdict['margin']:.0f} п.п.)")
        else:
            st.warning(f"🟠 Кандидат в НОВУЮ особь · макс. уверенность {verdict['confidence']:.0f}% "
                       f"(отрыв от №2: {verdict['margin']:.0f} п.п. — недостаточно для однозначного known)")
        st.caption("«Уверенность» — калиброванная монотонная шкала (НЕ вероятность); на temporal-кадрах Карелины (TK) "
                   "занижена — потолок данных (open-set AUROC≈0.45 sealed). Надёжный сигнал — РАНЖИР top-1/top-5 "
                   "(на нём KPI), а не порог known/new — он экспериментальный и слабый на этих данных.")
        if gt_is_new is not None:       # held-out: открытая сверка вердикта с истиной (известна/новая)
            ok = (verdict["verdict"] == "new_candidate") == gt_is_new
            gt = "НОВАЯ особь (нет в галерее)" if gt_is_new else f"известная особь {true_id}"
            (st.success if ok else st.error)(
                f"Истина (held-out): {gt} → вердикт {'✓ совпал' if ok else '✗ не совпал'}")

        st.subheader(f"Top-{topk} похожих особей")
        per_row = 4
        for r0 in range(0, len(ranked), per_row):
            cols = st.columns(per_row)
            for col, rank in zip(cols, range(r0, min(r0 + per_row, len(ranked)))):
                r = ranked[rank]
                with col:
                    crop = load_rgb(r["best_crop_path"])
                    if crop is not None:
                        st.image(crop, width="stretch")
                    badge = "🏆 Лучшее совпадение" if rank == 0 else f"№{rank + 1}"   # явный индекс, не ranked.index
                    hit = " ✓" if true_id and r["individual_id"] == true_id else ""
                    st.markdown(f"**{r['individual_id']}**{hit} · {badge}")
                    st.progress(min(int(r["confidence"]), 100), text=f"уверенность {r['confidence']:.0f}%")
                    st.caption(f"вид {r['cohort']} · фото в базе: {r['n_photos']}")

        # ── ДИФФЕРЕНЦИАТОР (опционально): оверлей совпавших пятен (запрос vs top-1) ──
        st.divider()
        show_overlay = st.checkbox("🔬 Показать, по каким пятнам совпало (оверлей созвездия пятен)", value=False)
        if show_overlay and query_rgb is not None:
            st.caption("Матчинг созвездия центроидов пятен — инвариантный признак «центр пятна не двигается»; линии соединяют "
                       "совпавшие пятна (наш дифференциатор — embedding-only системы на DINOv2 так не могут). "
                       "Исследовательская визуализация: на кадрах с разной позой/изгибом возможны единичные ложные линии.")
            try:
                top_crop = load_rgb(ranked[0]["best_crop_path"])
                ov, score, npairs = spot_overlay(query_rgb, top_crop, load_spot_config())
                import cv2
                st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB),
                         caption=f"Совпало пятен: {npairs} · доля совпадения {score * 100:.0f}% "
                                 f"(слева — запрос, справа — {ranked[0]['individual_id']})",
                         width="stretch")
                st.caption("Легенда: круги — найденные центроиды пятен; цветные линии — совпавшие пары "
                           "(цвет лишь различает пары); «доля совпадения» = пар / большее созвездие. "
                           + ("Линий нет: совпадений ниже порога — матч отвергнут (0 %)." if npairs == 0 else ""))
            except Exception as e:  # noqa: BLE001
                st.info(f"Оверлей пятен недоступен для этой пары ({type(e).__name__}).")

# ═══════════════ ВКЛАДКА: ОСОБИ ═══════════════
with tab_ind:
    st.subheader("База известных особей")
    scope2 = st.selectbox("Вид", ["Вся база", "TK", "PW"], index=0, key="ind_scope")
    g2 = filter_scope(gallery, scope2)
    uniq = sorted(set(g2.ids.tolist()))
    st.caption(f"Особей: {len(uniq)} · фото в галерее: {len(g2.ids)}")
    sel = st.selectbox("Особь", uniq)
    idxs = [i for i, u in enumerate(g2.ids) if u == sel]
    st.markdown(f"**{sel}** — фотографий в базе: {len(idxs)} (вид {g2.cohort[idxs[0]]})")
    cols = st.columns(5)
    for n, i in enumerate(idxs):
        crop = load_rgb(g2.crop_paths[i])
        if crop is not None:
            cols[n % 5].image(crop, caption=g2.md5[i][:8], width="stretch")

# ═══════════════ ВКЛАДКА: О СИСТЕМЕ ═══════════════
with tab_about:
    st.subheader("Архитектура и честные метрики")
    st.markdown(
        "**Пайплайн:** фото → сегментация брюшка (YOLO seg+pose) → кроп + ориентация → распрямление (unroll) → "
        "**эмбеддер MegaDescriptor-L-384** (Swin-L, zero-shot; ArcFace-дообучение оценено и отвергнуто) → "
        "top-K особей (косинус, numpy; FAISS — план масштабирования) → "
        "**матчер созвездия пятен** (guided, без зеркала) — переранжир + интерпретируемый оверлей → known/new.\n\n"
        "UML-диаграммы: `docs_public/uml/` (Use Case · Component · Deployment · Sequence)."
    )
    head = load_headline(_HEADLINE)
    if head is not None:
        st.markdown("**Финальные числа ВКР — sealed-test (held-out: нетронутые test/open_test, вскрыт 1 раз):**")
        st.table({
            "срез": ["overall", "PW (Ребристый)", "TK temporal (Карелина)"],
            "n": [head["n"], head["PW_n"], head["TK_n"]],
            "recall@1": [f"{head['overall@1']:.3f}", f"{head['PW@1']:.3f}", f"{head['TK@1']:.3f}"],
            "recall@5": [f"{head['overall@5']:.3f}", f"{head['PW@5']:.3f}", f"{head['TK@5']:.3f}"],
        })
        bo_txt = ""
        if head.get("bo@1") is not None:                       # числа belly_oriented — из того же артефакта (by_variant)
            bo_txt = (f"его sealed-числа по тому же артефакту (by_variant): @1 {head['bo@1']:.3f} / "
                      f"@5 {head['bo@5']:.3f}, PW @1 {head['bo_PW@1']:.3f} ")
        st.caption(
            f"Источник — `artifacts/ab_test_headline.json` (не тюнится). Числа таблицы — headline-вариант "
            f"`unroll_ribbon` (лучшая система ВКР); демо работает на варианте `belly_oriented` — {bo_txt}"
            f"(held-out эмбеддинги unroll_ribbon не выгружались; перегенерация запрещена sealed-гейтом). "
            f"**PW проходит KPI** (≥0.75/≥0.95); "
            f"**TK temporal — стена данных**. Pipeline-recall (знаменатель 113 офиц. проб): "
            f"@1 {head['pipeline@1']:.3f} / @5 {head['pipeline@5']:.3f}; open-set AUROC {head['auroc']:.3f}.")
    st.markdown("**Промежуточная оценка на dev (OOF, kpi_core, identity-level) — для сравнения систем "
                "(демо-конвейер выше работает на belly_oriented):**")
    st.table({
        "система": ["эмбеддер (zero-shot Mega + ribbon)", "матчер пятен (guided, покрыт тестами)", "гибрид (no-harm)"],
        "recall@1": ["0.18", "0.094", "0.18"],
        "recall@5": ["0.375", "0.141", "0.375"],
        "PW @1": ["~0.9", "0.82", "0.91"],
        "TK temporal @1": ["~0.11", "0.026", "0.11"],
    })
    st.info(
        "**Честный вывод ВКР.** KPI (top-1 ≥75 % И top-5 ≥95 %) на temporal Карелина не достигается ни эмбеддером "
        "(×2 SoTA), ни корректно построенным матчером пятен (корректность подтверждена тестами и sensitivity-сравнением), ни их гибридом. "
        "Барьер — в ДАННЫХ (воспроизводимость пятен между сессиями; no-new-data), а не в коде. "
        "Ценность работы: честный конвейер + строгая оценка (cross-fit, McNemar, sealed test) + "
        "интерпретируемость (оверлей пятен). Шкала «уверенность» — калиброванная монотонная (не вероятность); "
        "на близких особях возможна сатурация (ограничение open-set, AUROC≈0.45 sealed / ≈0.58 dev)."
    )
    st.caption("Стек: Python · PyTorch · timm (MegaDescriptor) · Ultralytics YOLO · OpenCV/scikit-image · "
               "вектор-поиск numpy-косинус (FAISS — план масштаба) · Streamlit. "
               "Лицензии и источники — `THIRD_PARTY_NOTICES.md`.")
