from modules.media.timecode import clock, parse_clock


def test_under_an_hour_is_minutes_and_seconds():
    assert clock(0) == "00:00"
    assert clock(5) == "00:05"
    assert clock(2870) == "47:50"


def test_an_hour_and_over_counts_hours_not_minutes_past_sixty():
    assert clock(4145) == "1:09:05"
    assert clock(3600) == "1:00:00"


def test_decimals_round_once_on_the_whole_value():
    assert clock(1291.5, 1) == "21:31.5"
    assert clock(59.96, 1) == "01:00.0"
    assert clock(4145.25, 3) == "1:09:05.250"


def test_negative_and_missing_read_as_zero():
    assert clock(-3) == "00:00"
    assert clock(None) == "00:00"


def test_parse_reads_every_form_a_person_types():
    assert parse_clock("21:31") == 1291
    assert parse_clock("1:09:05") == 4145
    assert parse_clock("69:05") == 4145
    assert parse_clock("4145") == 4145
    assert parse_clock(" 47:50.5 ") == 2870.5
    assert parse_clock("47:50,5") == 2870.5
    assert parse_clock("1h9m5s") == 4145
    assert parse_clock("1h 9m") == 4140
    assert parse_clock("45s") == 45


def test_parse_refuses_what_is_not_a_time():
    for bad in ("", "abc", "1:2:3:4", "1::2", "-5", "1.5:30", "12:-3", "h"):
        assert parse_clock(bad) is None, bad


def test_round_trip():
    for s in (0, 59, 61, 3599, 3600, 4145, 10000):
        assert parse_clock(clock(s)) == s
