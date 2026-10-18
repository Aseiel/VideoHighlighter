"""UI language registry and phase-1 catalogs."""

from modules import i18n
from modules.i18n import catalog, languages


def test_top_20_has_twenty_entries():
    assert len(languages.TOP_20) == 20


def test_phase1_selectable_includes_cjk_phase2():
    codes = {lang.code for lang in i18n.selectable_languages()}
    assert codes == {
        "en",
        "es",
        "pt-BR",
        "fr",
        "de",
        "id",
        "tr",
        "it",
        "pl",
        "vi",
        "nl",
        "zh-CN",
        "ja",
        "ko",
    }


def test_normalize_unknown_falls_back_to_english():
    assert i18n.normalize_code("nope") == "en"
    assert i18n.normalize_code("pt_BR") == "pt-BR"


def test_english_is_identity():
    i18n.set_language("en")
    assert i18n.t("Download") == "Download"
    assert i18n.t("About") == "About"


def test_turkish_tab_and_language_strings():
    i18n.set_language("tr")
    assert i18n.t("Download") == "İndir"
    assert i18n.t("About") == "Hakkında"
    assert i18n.t("Interface language") == "Arayüz dili"
    assert "yeniden başlat" in i18n.t("Language changes apply after you restart the app.").lower()


def test_spanish_and_polish_samples():
    i18n.set_language("es")
    assert i18n.t("Basic Settings") == "Ajustes básicos"
    i18n.set_language("pl")
    assert i18n.t("Timeline Viewer") == "Podgląd osi czasu"


def test_cjk_samples():
    i18n.set_language("zh-CN")
    assert i18n.t("Download") == "下载"
    assert i18n.t("Cancel") == "取消"
    assert i18n.t("About") == "关于"
    i18n.set_language("ja")
    assert i18n.t("Download") == "ダウンロード"
    assert i18n.t("Cancel") == "キャンセル"
    assert i18n.t("About") == "バージョン情報"
    i18n.set_language("ko")
    assert i18n.t("Download") == "다운로드"
    assert i18n.t("Cancel") == "취소"
    assert i18n.t("About") == "정보"


def test_every_phase1_catalog_covers_same_keys():
    keys = None
    for code, table in catalog.TRANSLATIONS.items():
        assert table, code
        if keys is None:
            keys = set(table)
        else:
            assert set(table) == keys, code
    # Full UI surface: main, timeline, teach, LLM, video_ai_editor, …
    assert keys is not None and len(keys) >= 700


def test_turkish_has_almost_no_english_identity():
    """Identity (value==key) is only OK for brands / universal tokens."""
    table = catalog.TRANSLATIONS["tr"]
    identity = [k for k, v in table.items() if v == k]
    allowed = {
        "OK",
        "smb://192.168.0.10/movies/",
        "Model:",
        "Interface",
        "Actions",
        "llama3.2",
        "ChatGPT (OpenAI API)",
        "Claude (Anthropic API)",
        "Kagi FastGPT (API)",
    }
    unexpected = [k for k in identity if k not in allowed]
    assert not unexpected, unexpected


def test_load_from_config():
    assert i18n.load_from_config({"ui": {"language": "de"}}) == "de"
    assert i18n.t("Cancel") == "Abbrechen"
    assert i18n.load_from_config({}) == "en"
