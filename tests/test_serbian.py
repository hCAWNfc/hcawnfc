"""Serbian rules (no weights and no native engine needed)."""
import json
from hcawn_fc._core import fc_sr, fc_triggers

NAMES = ["open_wifi_settings", "create_contact", "show_map", "create_calendar_event", "send_email", "turn_off_flashlight", "turn_on_flashlight"]


def test_serbian_triggers():
    assert "turn_on_flashlight" in fc_triggers.matched("upali lampu", NAMES)
    assert "send_email" in fc_triggers.matched("posalji mejl na ana@example.com sa temom Zdravo", NAMES)
    assert not fc_triggers.matched("koliko je sati", NAMES)


def test_dates_are_normalised():
    assert "tomorrow" in fc_sr.norm_for_dates("sastanak sutra u 15h")


def test_phone_is_completed():
    out = json.dumps([{"name": "create_contact", "arguments": {"first_name": "A", "phone_number": "555 1212"}}])
    res, changed = fc_sr.fix_phone(out, "dodaj kontakt A B 063 555 1212")
    assert changed and json.loads(res)[0]["arguments"]["phone_number"] == "063 555 1212"


def test_duplicate_call_dropped():
    c = {"name": "turn_on_flashlight", "arguments": {}}
    res, changed = fc_sr.dedupe_calls(json.dumps([c, c]))
    assert changed and len(json.loads(res)) == 1


def test_map_place_taken_from_request():
    out = json.dumps([{"name": "show_map", "arguments": {"query": "Tree Republice, Beogradu"}}])
    res, changed = fc_sr.fix_map_ground(out, "gde je Trg republike u Beogradu")
    assert changed and json.loads(res)[0]["arguments"]["query"] == "Trg republike u Beogradu"
