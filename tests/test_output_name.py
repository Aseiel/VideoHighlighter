"""The highlight filename follows the output-base field."""
from modules.media.output_name import highlight_output_path


def test_blank_keeps_the_current_stem_highlight_name():
    path = highlight_output_path(r"C:\clips\draw.mp4", "", multiple=False)
    assert path.replace("\\", "/").endswith("clips/draw_highlight.mp4")
    # A one-item GUI run is still a list, and the field is often the shipped
    # default. Blank is the only "leave it" value — see the other cases.
    same = highlight_output_path("/tmp/draw.mov", "   ", multiple=True)
    assert same.endswith("draw_highlight.mp4")


def test_a_name_is_used_for_one_video_and_prefixed_for_several():
    one = highlight_output_path("/data/draw.mp4", "reel", multiple=False)
    assert one.replace("\\", "/").endswith("/reel.mp4")
    named = highlight_output_path("/data/draw.mp4", "highlight.mp4", multiple=False)
    assert named.replace("\\", "/").endswith("/highlight.mp4")
    many = highlight_output_path("/data/draw.mp4", "reel.mp4", multiple=True)
    assert many.replace("\\", "/").endswith("/draw_reel.mp4")
    # The shipped default, on a batch, still lands on <stem>_highlight.mp4.
    default_batch = highlight_output_path("/data/draw.mp4", "highlight.mp4", multiple=True)
    assert default_batch.replace("\\", "/").endswith("/draw_highlight.mp4")


def test_a_directory_in_the_field_is_ignored_and_quotes_are_stripped():
    path = highlight_output_path("/data/draw.mp4", r"..\out\my'reel.mp4", multiple=False)
    assert path.replace("\\", "/").endswith("/myreel.mp4")
    assert ".." not in path.split("/")[-1]
