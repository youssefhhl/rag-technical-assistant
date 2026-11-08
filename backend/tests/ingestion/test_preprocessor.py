"""Tests for structure-aware preprocessing (heading hierarchy)."""

from app.ingestion.parser import Block, ParsedPage
from app.ingestion.preprocessor import Section, build_sections


def make_page(blocks, title="Docker Compose"):
    return ParsedPage(url="https://docs.test/compose", title=title, blocks=tuple(blocks))


def test_h1_h2_h3_hierarchy_tracking():
    page = make_page(
        [
            Block(kind="h1", text="Docker Compose"),
            Block(kind="text", text="Docker-compose lets you run multi-container apps."),
            Block(kind="h2", text="Services"),
            Block(kind="text", text="Services describe the containers."),
            Block(kind="h3", text="Ports"),
            Block(kind="text", text="Ports allow services to be exposed on the host machine."),
        ]
    )
    sections = build_sections(page)

    assert [s.heading_path for s in sections] == [
        ("Docker Compose",),
        ("Docker Compose", "Services"),
        ("Docker Compose", "Services", "Ports"),
    ]
    assert sections[-1].display_path == "Docker Compose > Services > Ports"
    assert sections[-1].blocks[0].text.startswith("Ports allow")


def test_content_before_first_heading_uses_page_title():
    page = make_page(
        [Block(kind="text", text="Intro before any heading."), Block(kind="h1", text="Main")]
    )
    sections = build_sections(page)
    assert sections[0].heading_path == ("Docker Compose",)
    assert "Intro before any heading." in sections[0].blocks[0].text


def test_h3_without_parent_heading_falls_back_to_page_title():
    page = make_page([Block(kind="h3", text="Lone Details"), Block(kind="text", text="Detail text.")])
    sections = build_sections(page)
    assert sections[0].heading_path == ("Docker Compose", "Lone Details")


def test_new_h1_resets_hierarchy():
    page = make_page(
        [
            Block(kind="h1", text="A"),
            Block(kind="h2", text="Sub"),
            Block(kind="text", text="under A/Sub."),
            Block(kind="h1", text="B"),
            Block(kind="text", text="under B."),
        ]
    )
    sections = build_sections(page)
    assert sections[-1].heading_path == ("B",)


def test_h4_and_below_are_not_hierarchy_levels():
    page = make_page(
        [
            Block(kind="h1", text="Doc"),
            Block(kind="h2", text="Sec"),
            Block(kind="text", text="Four-level heading kept as text."),
        ]
    )
    page.blocks[2]  # ensure tuple works
    # Parser maps h4+ to "text" blocks; preprocessor must not create a level for them.
    page = ParsedPage(
        url=page.url,
        title=page.title,
        blocks=page.blocks + (Block(kind="text", text="h4 text later"),),
    )
    sections = build_sections(page)
    assert all(s.heading_path == ("Doc", "Sec") for s in sections)


def test_empty_blocks_produce_no_sections():
    assert build_sections(make_page([])) == []
    # whitespace-only blocks filtered out
    page = make_page([Block(kind="text", text="   ")])
    assert build_sections(page) == []


def test_deeper_h3_keeps_first_two_levels():
    page = make_page(
        [
            Block(kind="h1", text="A"),
            Block(kind="h2", text="B"),
            Block(kind="h3", text="C1"),
            Block(kind="text", text="one."),
            Block(kind="h3", text="C2"),
            Block(kind="text", text="two."),
        ]
    )
    sections = build_sections(page)
    assert [s.heading_path for s in sections] == [
        ("A", "B", "C1"),
        ("A", "B", "C2"),
    ]
