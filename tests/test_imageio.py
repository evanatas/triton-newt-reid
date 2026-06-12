"""Тесты единого канонического ридера изображений (TDD)."""
import hashlib

from PIL import Image

from triton_data.imageio import file_md5, read_image_stats, load_canonical


def _make_img(path, size=(40, 30), color=(120, 80, 60), orientation=None):
    """Создаёт маленький JPEG; при orientation выставляет EXIF-тег ориентации."""
    img = Image.new("RGB", size, color)
    if orientation is not None:
        exif = img.getexif()
        exif[0x0112] = orientation  # тег Orientation
        img.save(path, exif=exif.tobytes())
    else:
        img.save(path)


# --- file_md5 ---
def test_file_md5_matches_hashlib(tmp_path):
    p = tmp_path / "a.jpg"
    _make_img(p)
    assert file_md5(p) == hashlib.md5(p.read_bytes()).hexdigest()


def test_file_md5_identical_bytes_collide(tmp_path):
    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _make_img(p1)
    p2.write_bytes(p1.read_bytes())  # побайтовая копия
    assert file_md5(p1) == file_md5(p2)


def test_file_md5_differs_on_content(tmp_path):
    p1, p2 = tmp_path / "a.jpg", tmp_path / "b.jpg"
    _make_img(p1, color=(10, 10, 10))
    _make_img(p2, color=(200, 200, 200))
    assert file_md5(p1) != file_md5(p2)


# --- read_image_stats: display-размеры после EXIF-transpose + сырой тег ориентации ---
def test_read_image_stats_plain(tmp_path):
    p = tmp_path / "a.jpg"
    _make_img(p, size=(40, 30))  # W=40, H=30
    st = read_image_stats(p)
    assert (st.width, st.height) == (40, 30)
    assert st.mode == "RGB"
    assert st.orientation in (None, 1)


def test_read_image_stats_applies_exif_orientation(tmp_path):
    p = tmp_path / "rot.jpg"
    _make_img(p, size=(40, 30), orientation=6)  # 6 = поворот 90° → стороны меняются
    st = read_image_stats(p)
    assert (st.width, st.height) == (30, 40)
    assert st.orientation == 6


# --- load_canonical: RGB + применённый поворот ---
def test_load_canonical_rgb_and_transposed(tmp_path):
    p = tmp_path / "rot.jpg"
    _make_img(p, size=(40, 30), orientation=6)
    im = load_canonical(p)
    assert im.mode == "RGB"
    assert im.size == (30, 40)
