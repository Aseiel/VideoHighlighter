from tools.release_notes import Commit, classify, render, subjects


def _c(subject):
    return Commit("0" * 40, subject)


def test_types_map_to_sections():
    assert classify("feat(report): add a thing")[0] == "New"
    assert classify("fix(llm): stop a thing")[0] == "Fixed"
    assert classify("perf(hover): faster thing")[0] == "Faster"


def test_internal_work_is_hidden_by_type_and_by_scope():
    assert classify("docs: explain a thing")[0] is None
    assert classify("test(directml): cover a thing")[0] is None
    # A fix to the pipeline is not a fix to the app.
    assert classify("fix(ci): sync configs")[0] is None
    assert classify("fix(release): reorder assets")[0] is None


def test_an_unprefixed_subject_still_ships():
    assert classify("Tidy the splash screen") == ("Other", "", "Tidy the splash screen")


def test_render_groups_in_order_and_counts_what_it_hid():
    notes = render([_c("fix(ui): keep the button visible"),
                    _c("feat(install): give buyers an installer"),
                    _c("docs: preserve drafts"),
                    _c("fix(ci): sync configs")], "0.11.0")
    assert notes.startswith("## What changed since 0.11.0")
    assert notes.index("### New") < notes.index("### Fixed")
    assert "- Give buyers an installer (install)" in notes
    assert "2 internal commits" in notes
    assert "preserve drafts" not in notes


def test_render_says_so_when_nothing_user_facing_changed():
    notes = render([_c("docs: a note")], None)
    assert "No user-facing changes." in notes
    assert "1 internal commit " in notes


def test_a_branch_titled_squash_merge_is_read_from_its_body():
    squash = Commit("0" * 40, "Claude/amazing ptolemy yf1m4v (#22)",
                    "* feat(mac): run detection on the Apple GPU\n\nWhy it matters.\n\n"
                    "* test(mac): cover it\n* fix(log): name Core ML\n\n"
                    "Co-authored-by: Someone <someone@example.com>\n")
    assert subjects(squash) == ["feat(mac): run detection on the Apple GPU",
                                "test(mac): cover it", "fix(log): name Core ML"]
    notes = render([squash], "0.12.1")
    assert "- Run detection on the Apple GPU (mac)" in notes
    assert "- Name Core ML (log)" in notes
    assert "Claude/amazing" not in notes
    assert "1 internal commit " in notes


def test_a_squash_merge_without_a_list_keeps_its_title():
    plain = Commit("0" * 40, "Let the app name the file (#34)", "Some prose.\n")
    assert subjects(plain) == ["Let the app name the file (#34)"]


def test_a_conventional_squash_merge_is_not_expanded():
    """Its title already says what shipped; the list under it is the steps."""
    merged = Commit("0" * 40, "feat(actions): recognise actions (#33)",
                    "* feat(encoder): a step\n* fix(encoder): another\n")
    assert subjects(merged) == ["feat(actions): recognise actions (#33)"]


def test_stacked_pull_requests_list_a_change_once():
    """A PR branched off another unmerged one lists the earlier PR's commits
    in its own squash body too."""
    first = Commit("0" * 40, "Siglip2 encoder (#28)", "* feat(vision): an encoder\n")
    second = Commit("1" * 40, "Action head trainer (#29)",
                    "* feat(vision): an encoder\n* feat(training): a head\n")
    notes = render([first, second], None)
    assert notes.count("An encoder (vision)") == 1
    assert "- A head (training)" in notes


def test_tools_work_is_internal():
    assert classify("tools(teach_lab): measure a thing")[0] is None

