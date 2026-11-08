"""Run the test suite and create JUnit XML plus a readable Markdown report."""

from datetime import datetime, timezone
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
JUNIT_PATH = REPORTS / "test-results.xml"
MARKDOWN_PATH = REPORTS / "test_report.md"


def _write_markdown_report(pytest_exit_code: int) -> None:
    lines = [
        "# Automated Test Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    try:
        root = ET.parse(JUNIT_PATH).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        totals = {name: 0 for name in ("tests", "failures", "errors", "skipped")}
        passed: list[str] = []
        failed: list[str] = []
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.attrib.get(key, 0))
            for case in suite.findall("testcase"):
                label = f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"
                if case.find("failure") is not None or case.find("error") is not None:
                    failed.append(label)
                elif case.find("skipped") is None:
                    passed.append(label)
        passed_count = len(passed)
        failed_count = totals["failures"] + totals["errors"]
        lines.extend(
            [
                f"**Result:** {'PASSED' if pytest_exit_code == 0 else 'FAILED'}",
                "",
                f"- Passed: {passed_count}",
                f"- Failed: {failed_count}",
                f"- Skipped: {totals['skipped']}",
                f"- Total: {totals['tests']}",
                "",
                "## Failed tests",
                "",
            ]
        )
        lines.extend([f"- `{name}`" for name in failed] if failed else ["None."])
        lines.extend(["", "## Passed tests", ""])
        lines.extend([f"- `{name}`" for name in passed] if passed else ["None."])
    except (ET.ParseError, OSError):
        lines.extend(
            [
                f"**Result:** {'PASSED' if pytest_exit_code == 0 else 'FAILED'}",
                "",
                "JUnit results were unavailable; see the test command output.",
            ]
        )
    MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    exit_code = int(
        pytest.main(
            [
                "-q",
                "--basetemp",
                str(REPORTS / ".pytest-tmp"),
                "--junitxml",
                str(JUNIT_PATH),
                str(ROOT / "tests"),
            ]
        )
    )
    _write_markdown_report(exit_code)
    print(f"Markdown test report: {MARKDOWN_PATH.relative_to(ROOT)}")
    print(f"JUnit test report: {JUNIT_PATH.relative_to(ROOT)}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
