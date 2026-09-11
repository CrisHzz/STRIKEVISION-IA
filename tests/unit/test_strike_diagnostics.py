from ufc_tracker.strike.diagnostics import analyze_validation


def test_validation_diagnostics_rejects_wrong_fight_and_reports_errors():
    rows = []
    for index, (label, probability) in enumerate(
        [("no_strike", .1), ("no_strike", .8), ("strike", .2), ("strike", .9)]
    ):
        rows.append({
            "window_id": f"v__{index}", "video_id": "v", "fight_id": "validation_fight",
            "start_frame": index * 12, "end_frame": index * 12 + 23,
            "label": label, "p_strike": probability, "threshold": .5,
        })
    report = analyze_validation(rows, {"validation_fight"})
    assert report["errors"]["false_positive"]["count"] == 1
    assert report["errors"]["false_negative"]["count"] == 1
    assert report["test_accessed"] is False

    rows[0]["fight_id"] = "test_fight"
    try:
        analyze_validation(rows, {"validation_fight"})
    except ValueError as error:
        assert "validation" in str(error)
    else:
        raise AssertionError("Expected non-validation fight rejection")
