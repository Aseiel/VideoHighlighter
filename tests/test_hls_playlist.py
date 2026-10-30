"""Reading an HLS playlist for a segment-by-segment download. No network."""

from __future__ import annotations

from modules.media.hls_playlist import best_variant, is_playlist, media_playlist

MASTER = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360,CODECS="avc1.4d401e,mp4a.40.2"
360p/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080
1080p/index.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2500000,RESOLUTION=1280x720
https://cdn.example/720p/index.m3u8
"""

FMP4 = """#EXTM3U
#EXT-X-VERSION:7
#EXT-X-TARGETDURATION:4
#EXT-X-MAP:URI="init_a1.mp4"
#EXTINF:4.000,
seg_1.m4s
#EXTINF:4.000,
seg_2.m4s?token=abc
#EXTINF:2.500,
/abs/seg_3.m4s
#EXT-X-ENDLIST
"""


def test_a_playlist_is_known_by_its_url_or_its_first_bytes():
    assert is_playlist("https://cdn.example/hls/2/main.m3u8?sig=1")
    assert not is_playlist("https://cdn.example/video.mp4")
    assert is_playlist("https://cdn.example/stream", b"#EXTM3U\n#EXT-X-VERSION:3")
    assert is_playlist("https://cdn.example/stream", "﻿#EXTM3U\n")
    assert not is_playlist("https://cdn.example/stream", b"\x00\x00\x00\x18ftypmp42")


def test_the_master_playlist_gives_its_highest_bandwidth_stream():
    url = best_variant(MASTER, "https://cdn.example/hls/2/main.m3u8")
    assert url == "https://cdn.example/hls/2/1080p/index.m3u8"


def test_a_media_playlist_has_no_variant():
    assert best_variant(FMP4, "https://cdn.example/a/index.m3u8") is None


def test_segments_come_back_absolute_and_in_order_with_the_init_segment():
    pl = media_playlist(FMP4, "https://cdn.example/a/index.m3u8")

    assert pl.init == "https://cdn.example/a/init_a1.mp4"
    assert pl.segments == [
        "https://cdn.example/a/seg_1.m4s",
        "https://cdn.example/a/seg_2.m4s?token=abc",
        "https://cdn.example/abs/seg_3.m4s",
    ]
    assert pl.seconds == 10.5
    assert not pl.encrypted


def test_an_encrypted_stream_is_flagged_and_an_unencrypted_key_line_is_not():
    aes = "#EXTM3U\n#EXT-X-KEY:METHOD=AES-128,URI=\"k.bin\"\n#EXTINF:4,\na.ts\n"
    none = "#EXTM3U\n#EXT-X-KEY:METHOD=NONE\n#EXTINF:4,\na.ts\n"

    assert media_playlist(aes, "https://x/p.m3u8").encrypted
    assert not media_playlist(none, "https://x/p.m3u8").encrypted
