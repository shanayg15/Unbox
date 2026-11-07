"""B-Show B6: static checks on the UI (layout contract, projector sizes, every event handled).

The behavioural check (replay animates every panel, no scrollbars at 1920x1080) is done in a
real browser against /api/replay; these tests keep the contract from drifting.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "app" / "static"
HTML = (STATIC / "index.html").read_text()
CSS = (STATIC / "style.css").read_text()
JS = (STATIC / "app.js").read_text()


def css_rule(selector: str) -> str:
    m = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", CSS)
    assert m, f"no CSS rule for {selector}"
    return m.group(1)


def px(rule: str, prop: str = "font-size") -> int:
    m = re.search(prop + r"\s*:\s*(\d+)px", rule)
    assert m, f"{prop} not set in {rule!r}"
    return int(m.group(1))


def test_layout_skeleton():
    for el_id in ["bbImg", "cloneImg", "clonePanel", "selPolicy", "selFlaw", "selBrain", "btnRun",
                  "btnReveal", "btnFix", "chkShield", "statQueries", "statMatch", "status",
                  "ledgerRows", "subq", "subqLine", "subqBars", "code", "curve", "results", "banner"]:
        assert f'id="{el_id}"' in HTML, el_id
    assert "Policies under study: SB3 RL Zoo (third-party). Everything else built Nov 7." in HTML
    assert 'class="left"' in HTML and 'class="right"' in HTML and "bottom" in HTML


def test_grid_and_projector_sizes():
    body = css_rule("body")
    assert "grid-template-columns" in body and "100vh" in body and "overflow: hidden" in body
    assert "3fr" in css_rule(".right .bottom") and "2fr" in css_rule(".right .bottom")
    assert "object-fit: contain" in css_rule(".sim img")
    assert px(css_rule(".lrow")) >= 18
    assert px(css_rule(".code"), "font") >= 16


def test_every_event_type_is_handled():
    from contracts.events import EVENT_TYPES

    for etype in set(EVENT_TYPES) | {"snap"}:
        assert f'case "{etype}":' in JS, etype


def test_subq_panel_pieces():
    assert "SubQ Snap" in JS and "raw rows" in JS and "probe next" in JS
    assert "SubQ p=" in JS                        # chip on each ledger row
    assert "feature_relevance" in JS and "rule_support" in JS
    assert ".chip" in CSS and "dashed" in css_rule(".chip-demo, .chip")   # distinct from verdict icons


def test_ledger_icons_and_reveal_labels():
    for icon in ["✓", "✗", "?", "◎", "🔍"]:
        assert icon in JS
    assert "line-through" in css_rule(".lrow.refuted .text")
    assert "BLACK BOX + SHIELD (fixed rule)" in JS
    assert "highlight_rule_ids" in JS


def test_clip_player_contract():
    # one timer drives both images; new clips wait for the loop start
    assert JS.count("setInterval(") == 1
    assert "pendingClip" in JS and "data:image/jpeg;base64," in JS
    assert 'classList.toggle("disagree"' in JS


def test_fallback_banner_is_never_silent():
    import app.server as server

    assert server.FALLBACK_TIMEOUT_TEXT == (
        "Live run exceeded time limit, replaying cached run of this exact configuration.")
    assert "d.banner" in JS and "showBanner" in JS


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_parses():
    r = subprocess.run(["node", "--check", str(STATIC / "app.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
