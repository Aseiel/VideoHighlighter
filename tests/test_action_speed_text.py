"""The speed action recognition reports is the video's, not the analysed frames'.

It used to report analysed frames per second. With a frame skip of 5 that is a
fifth of the frames read, so a run reading 1,515 frames/s said "303 FPS". A
higher frame skip made the run faster while that number fell, and the setting
looked useless.
"""

from __future__ import annotations

from action_recognition import _speed_text


def test_counts_every_frame_read():
    # 34,367 frames read in 22.7 s, 1 in 5 of them analysed
    assert _speed_text(34367, 22.7, 5) == "1514 fps (1 in 5 analysed)"


def test_a_bigger_frame_skip_does_not_look_slower():
    # the same video at frame skip 5 and 10: the faster run shows the bigger number
    five = _speed_text(34367, 21.2, 5)
    ten = _speed_text(34367, 12.4, 10)
    assert int(ten.split()[0]) > int(five.split()[0])


def test_every_frame_analysed_says_nothing_about_skipping():
    assert _speed_text(1000, 10.0, 1) == "100 fps"


def test_the_first_tick():
    assert _speed_text(0, 0.0, 5) == "starting"
