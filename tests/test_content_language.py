"""The footage's language follows the interface language.

Users who chose 简体中文 got menus in Chinese and everything else in English or
Polish: Whisper listened for English, subtitles were written in Polish (the
shipped target), narration came back in English and search read Chinese
queries as English. These pin the rules that fixed it.
"""
from __future__ import annotations

import numpy as np
import pytest

from modules.i18n import content_language as cl


def _old_config(ui="zh-CN"):
    """What a build before this saved, whether anyone chose these or not."""
    return {"ui": {"language": ui},
            "transcript": {"source_lang": "en", "model": "base"},
            "subtitles": {"source_lang": "en", "target_lang": "pl"}}


class TestFollow:
    def test_switching_language_moves_what_still_holds_the_old_default(self):
        cfg = {"transcript": {"source_lang": "en"},
               "subtitles": {"source_lang": "en", "target_lang": "en"}}
        cl.follow(cfg, "en", "ja")
        assert cfg["transcript"]["source_lang"] == "ja"
        assert cfg["subtitles"] == {"source_lang": "ja", "target_lang": "ja"}

    def test_a_language_chosen_on_purpose_stays(self):
        cfg = {"transcript": {"source_lang": "ko"},
               "subtitles": {"source_lang": "ko", "target_lang": "en"}}
        cl.follow(cfg, "en", "zh-CN")
        assert cfg["transcript"]["source_lang"] == "ko"
        assert cfg["subtitles"]["target_lang"] == "zh"

    def test_interface_codes_map_to_whisper_codes(self):
        assert cl.whisper_code("zh-CN") == "zh"
        assert cl.whisper_code("pt-BR") == "pt"
        assert cl.whisper_code("nonsense") == "en"


class TestMigrate:
    def test_an_old_chinese_config_is_moved_once(self):
        cfg = _old_config("zh-CN")
        moved = cl.migrate(cfg)
        assert ("transcript", "source_lang") in moved
        assert cfg["transcript"]["source_lang"] == "zh"
        assert cfg["subtitles"]["target_lang"] == "zh"
        assert cfg["ui"][cl.MIGRATED_KEY] is True
        # A later choice survives every later start.
        cfg["transcript"]["source_lang"] = "en"
        assert cl.migrate(cfg) == []
        assert cfg["transcript"]["source_lang"] == "en"

    def test_an_english_config_is_left_alone(self):
        cfg = _old_config("en")
        assert cl.migrate(cfg) == []
        assert cfg["subtitles"]["target_lang"] == "pl"
        assert cfg["ui"][cl.MIGRATED_KEY] is True


class TestFirstStart:
    def test_shipped_config_leaves_the_language_unchosen(self):
        import os
        import yaml
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "config.yaml"), encoding="utf-8") as fh:
            assert not cl.chosen(yaml.safe_load(fh))

    def test_installer_answer_is_read(self, tmp_path):
        (tmp_path / cl.INSTALL_FILE).write_text("zh-CN\n", encoding="utf-8")
        assert cl.installer_choice(str(tmp_path)) == "zh-CN"

    def test_installer_answer_survives_a_bom(self, tmp_path):
        (tmp_path / cl.INSTALL_FILE).write_bytes("﻿pl".encode("utf-8"))
        assert cl.installer_choice(str(tmp_path)) == "pl"

    def test_no_or_unknown_installer_answer(self, tmp_path):
        assert cl.installer_choice(str(tmp_path)) is None
        (tmp_path / cl.INSTALL_FILE).write_text("xx", encoding="utf-8")
        assert cl.installer_choice(str(tmp_path)) is None

    @pytest.mark.parametrize("tags,want", [
        (["zh-Hans-CN", "en-US"], "zh-CN"),
        (["zh-TW"], "zh-CN"),
        (["pt-PT"], "pt-BR"),
        (["pl_PL"], "pl"),
        (["ja-JP"], "ja"),
        (["sv-SE", "de-DE"], "de"),
        (["sv-SE"], "en"),
        ([], "en"),
    ])
    def test_system_language_guess(self, tags, want):
        assert cl.guess_from_locale(tags) == want


class TestNarration:
    def test_follows_the_interface_language(self):
        assert cl.narration_instruction({"ui": {"language": "zh-CN"}}) == \
            "Write in Chinese (Simplified), whatever language this prompt is in."
        assert cl.narration_instruction({"ui": {"language": "en"}}) == ""

    def test_a_hand_set_language_wins(self):
        cfg = {"ui": {"language": "zh-CN"}, "visualization": {"narration_language": "pl"}}
        assert "Polish" in cl.narration_instruction(cfg)

    def test_system_prompts_carry_it_except_for_captioners(self, monkeypatch):
        from modules.i18n import core
        from modules.narration import chapter_story, clip_story
        monkeypatch.setattr(core, "_current", "ja")
        for module in (chapter_story, clip_story):
            assert module.system_prompt_for("qwen2.5vl:7b").endswith(
                "Write in Japanese, whatever language this prompt is in.")
            assert "Japanese" not in module.system_prompt_for("joycaption:q4")


class TestSearchRouting:
    def test_which_queries_go_to_the_multilingual_model(self):
        from llm import clip_multilingual as cm
        assert not cm.wanted(["a dog running"], ui_language="en")
        assert cm.wanted(["一只狗"], ui_language="en")
        assert cm.wanted(["pies biegnie"], ui_language="pl")
        assert cm.wanted(["a dog"], ui_language="zh-CN")

    def test_clip_text_tower_is_kept_when_the_model_is_missing(self, monkeypatch):
        from llm import clip_multilingual as cm
        cm.reset_for_tests()
        monkeypatch.setattr(cm, "find_model_dir", lambda: None)
        said = []
        assert cm.encode_if_wanted(["一只狗"], log=said.append) is None
        assert cm.encode_if_wanted(["一只狗"], log=said.append) is None
        assert len(said) == 1          # said once, not per query
        cm.reset_for_tests()

    def test_embed_texts_uses_it_when_wanted(self, monkeypatch):
        from llm import clip_index, clip_multilingual as cm
        vec = np.eye(1, 512, dtype=np.float32)
        monkeypatch.setattr(cm, "encode_if_wanted", lambda texts, log=print: vec)
        e = clip_index.ClipEmbedder.__new__(clip_index.ClipEmbedder)
        e._model = object()
        assert e.embed_texts(["一只狗"]) is vec

    def test_pack_unpacks_where_the_model_is_looked_for(self):
        from llm import clip_multilingual as cm
        from modules.packs import pack_manager as pm
        assert pm.MODEL_DIRS[pm.MULTILINGUAL_PACK] == (cm.MODEL_DIRNAME, cm.MODEL_FILE)


def _installed():
    from llm import clip_multilingual as cm
    return cm.is_installed()


@pytest.mark.skipif(not _installed(), reason="multilingual search model not installed")
def test_the_installed_model_puts_translations_together():
    from llm import clip_multilingual as cm
    cm.reset_for_tests()
    model = cm.load(log=lambda *_: None)
    dog_en, dog_zh, car_zh = model.encode(["a photo of a dog", "一只狗的照片", "一辆汽车的照片"])
    assert float(dog_en @ dog_zh) > float(dog_en @ car_zh) + 0.05
    cm.reset_for_tests()
