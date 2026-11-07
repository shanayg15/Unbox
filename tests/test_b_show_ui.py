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
    for el_id in ["bbImg", "cloneImg", "clonePanel", "bbPill", "clonePill", "selPolicy", "selFlaw", "btnRun",
                  "btnFix", "viewToggle", "fixSteps", "checks", "feed", "kMatch", "kDecisions",
                  "ledger", "subqLine", "subqBars", "code", "params", "curve", "results", "banner",
                  "diag", "reportBody", "backupList", "secNav"]:
        assert f'id="{el_id}"' in HTML, el_id
    for sec in ["sec-setup", "sec-analysis", "sec-fix", "sec-report"]:
        assert f'id="{sec}"' in HTML and f'href="#{sec}"' in HTML, sec
    assert "SB3 RL Zoo" in HTML and "Everything else built Nov 7." in HTML
    assert "Deployed policy" in HTML and "Recovered policy" in HTML
    assert "Uncover the network's hidden logic" in HTML
    assert "Apply fix" in HTML and "Diagnostic report" in HTML
    assert "robotics team gets" not in HTML


def test_grid_and_projector_sizes():
    body = css_rule("body")
    assert "grid-template-columns" in body and "100vh" in body and "overflow: hidden" in body
    assert "object-fit: contain" in css_rule(".screen img")
    assert px(css_rule(".lrow")) >= 18
    assert px(css_rule(".code"), "font") >= 16
    assert "pre" in css_rule(".code .src")          # no wrapped code lines


def test_every_event_type_is_handled():
    from contracts.events import EVENT_TYPES

    for etype in set(EVENT_TYPES) | {"snap"}:
        assert f'case "{etype}":' in JS, etype


def test_subq_and_ledger_pieces():
    assert "raw rows" in JS and "anomaly_feature" in JS
    assert "Confirmed" in JS and "Under test" in JS and "Rejected" in JS
    assert "line-through" in css_rule("details.ruled .lrow .text")
    assert "Deployed policy · fix applied" in JS and "No remediation is needed" in JS
    for emoji in ["✓", "✗", "◎", "🔍", "⚠"]:
        assert emoji not in JS and emoji not in HTML


def test_clip_player_contract():
    # one timer drives both images; clips loop in sync and hold the last frame with a label
    assert JS.count("setInterval(") == 1
    assert "HOLD_MS" in JS and "data:image/jpeg;base64," in JS
    assert 'classList.toggle("disagree"' in JS and "endLabel" in JS


def test_fallback_banner_is_never_silent():
    import app.server as server

    assert server.FALLBACK_TIMEOUT_TEXT == (
        "The live run could not finish within the time limit. Showing our earlier run of this exact "
        "configuration.")
    assert "d.banner" in JS and "showBanner" in JS


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_parses():
    r = subprocess.run(["node", "--check", str(STATIC / "app.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
