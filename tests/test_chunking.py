from docsage.ingest.chunking import count_tokens, fixed_chunks, heading_chunks
from docsage.models import Line, Page


def page(n, lines):
    return Page("TCS AR", n, [Line(t, s) for t, s in lines])


def test_fixed_chunks_respect_size_and_keep_page():
    words = " ".join(f"w{i}" for i in range(2000))
    chunks = fixed_chunks([page(7, [(words, 9.0)])], size=512, overlap=64)
    assert len(chunks) > 1
    assert all(c.doc == "TCS AR" and c.page == 7 for c in chunks)
    assert all(count_tokens(c.text) <= 512 for c in chunks)


def test_fixed_chunks_overlap():
    words = " ".join(f"w{i}" for i in range(2000))
    a, b = fixed_chunks([page(1, [(words, 9.0)])], size=512, overlap=64)[:2]
    assert set(a.text.split()) & set(b.text.split())


def test_chunks_never_span_pages():
    pages = [page(1, [("alpha " * 50, 9.0)]), page(2, [("beta " * 50, 9.0)])]
    for chunker in (fixed_chunks, heading_chunks):
        for c in chunker(pages):
            assert not ("alpha" in c.text and "beta" in c.text)


def test_heading_chunks_split_at_headings_and_prefix():
    p = page(
        3,
        [
            ("Body text " * 20, 9.0),
            ("Employee Attrition", 14.0),
            ("Attrition was 13.3 percent this year " * 8, 9.0),
            ("Revenue", 14.0),
            ("Revenue grew four percent " * 12, 9.0),
        ],
    )
    chunks = heading_chunks([p])
    heads = [c.heading for c in chunks]
    assert "Employee Attrition" in heads and "Revenue" in heads
    attr = next(c for c in chunks if c.heading == "Employee Attrition")
    assert attr.text.startswith("Employee Attrition")
    assert "Revenue grew" not in attr.text


def test_heading_carries_across_pages():
    pages = [
        page(1, [("Body " * 30, 9.0), ("Risk Management", 14.0), ("first part", 9.0)]),
        page(2, [("second part continues", 9.0)]),
    ]
    chunks = heading_chunks(pages)
    assert next(c for c in chunks if c.page == 2).heading == "Risk Management"


def test_tiny_sections_merge_into_neighbour():
    p = page(
        4,
        [
            ("Intro " * 60, 9.0),
            ("Six", 14.0),
            ("short note", 9.0),
            ("Employees", 14.0),
            ("Total number of employees was 242,156 " * 10, 9.0),
        ],
    )
    chunks = heading_chunks([p])
    assert all(count_tokens(c.text) >= 40 for c in chunks)
    emp = next(c for c in chunks if "242,156" in c.text)
    assert "short note" in emp.text  # tiny section folded forward, same page
    assert emp.page == 4


def test_strip_boilerplate_removes_repeated_footers_keeps_body():
    from docsage.ingest.pdf_loader import strip_boilerplate

    pages = [
        Page(
            "Wipro AR",
            n,
            [
                Line(f"Body text on topic {chr(97 + n % 26)}{chr(97 + n // 26)} about employees", 9.0, 0.4),
                Line(f"WIPRO INTEGRATED ANNUAL REPORT 2025-26 {n}", 7.0, 0.83),  # mid-page footer
                Line(str(n), 7.0, 0.97),  # page number in margin
            ],
        )
        for n in range(1, 41)
    ]
    out = strip_boilerplate(pages)
    for p in out:
        assert len(p.lines) == 1 and p.lines[0].text.startswith("Body text")
