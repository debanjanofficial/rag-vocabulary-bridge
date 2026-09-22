"""
Foundations of Modern RAG, Embeddings & Vector Search:
From Mathematical First Principles to Production Grounding.

A Comprehensive Illustrated Textbook & Data Science Handbook.
Compiles a publication-grade PDF using ReportLab with custom vector diagrams,
formula callouts, tables, and structured chapters.
"""

from __future__ import annotations

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import (
    Drawing,
    Rect,
    String,
    Line,
    Circle,
    Group,
    Polygon,
)

# ── Color Palette ─────────────────────────────────────────────────────────────
PRIMARY = HexColor("#0F172A")       # Deep Slate Navy
SECONDARY = HexColor("#1E293B")     # Card Slate
ACCENT_BLUE = HexColor("#2563EB")   # Royal Blue
ACCENT_CYAN = HexColor("#0284C7")   # Deep Sky Blue
ACCENT_ORANGE = HexColor("#EA580C") # REHAU Industrial Orange
ACCENT_GREEN = HexColor("#059669")  # Factual Grounding Green
ACCENT_RED = HexColor("#DC2626")    # Hallucination Red
ACCENT_PURPLE = HexColor("#7C3AED") # Math / Graph Purple
BG_LIGHT = HexColor("#F8FAFC")      # Off-white / canvas
BOX_BG_BLUE = HexColor("#EFF6FF")   # Note box background
BOX_BG_WARN = HexColor("#FEF2F2")   # Warning box background
BOX_BG_MATH = HexColor("#F5F3FF")   # Math box background
BOX_BG_SUCC = HexColor("#ECFDF5")   # Success box background
TEXT_DARK = HexColor("#0F172A")
TEXT_MUTED = HexColor("#475569")
BORDER_LIGHT = HexColor("#CBD5E1")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic 'Page X of Y' and chapter running headers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int):
        if self._pageNumber == 1:
            return  # Skip cover page

        self.saveState()
        page_w, page_h = letter

        # Running Top Header
        self.setStrokeColor(BORDER_LIGHT)
        self.setLineWidth(0.6)
        self.line(48, page_h - 40, page_w - 48, page_h - 40)

        self.setFont("Helvetica", 8)
        self.setFillColor(TEXT_MUTED)
        self.drawString(
            48,
            page_h - 34,
            "Foundations of Industrial RAG, Embeddings & Vector Search",
        )
        self.drawRightString(
            page_w - 48,
            page_h - 34,
            "Mathematical Principles to Production Grounding",
        )

        # Running Bottom Footer
        self.line(48, 42, page_w - 48, 42)
        self.drawString(
            48,
            30,
            "Data Science & AI Engineering Reference Handbook",
        )
        page_str = f"Page {self._pageNumber} of {total_pages}"
        self.drawRightString(page_w - 48, 30, page_str)

        self.restoreState()


def get_handbook_styles():
    """Build typographic style sheet."""
    base = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CoverTitle",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,
        textColor=PRIMARY,
        alignment=0,
        spaceAfter=12,
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=ACCENT_CYAN,
        spaceAfter=24,
    )

    h1_style = ParagraphStyle(
        "HandbookH1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=PRIMARY,
        spaceBefore=22,
        spaceAfter=10,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "HandbookH2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=ACCENT_BLUE,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h3_style = ParagraphStyle(
        "HandbookH3",
        parent=base["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "HandbookBody",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=TEXT_DARK,
        spaceAfter=8,
    )

    bullet_style = ParagraphStyle(
        "HandbookBullet",
        parent=body_style,
        leftIndent=16,
        firstLineIndent=-10,
        spaceAfter=4,
    )

    code_style = ParagraphStyle(
        "HandbookCode",
        parent=base["Normal"],
        fontName="Courier",
        fontSize=8.5,
        leading=11.5,
        textColor=HexColor("#0F172A"),
    )

    formula_style = ParagraphStyle(
        "HandbookFormula",
        parent=base["Normal"],
        fontName="Courier-Bold",
        fontSize=9.5,
        leading=13,
        textColor=ACCENT_PURPLE,
        alignment=1,
    )

    return {
        "title": title_style,
        "subtitle": subtitle_style,
        "h1": h1_style,
        "h2": h2_style,
        "h3": h3_style,
        "body": body_style,
        "bullet": bullet_style,
        "code": code_style,
        "formula": formula_style,
    }


def make_callout(
    title: str,
    text: str,
    callout_type: str = "note",
    styles: dict | None = None,
) -> Table:
    """Build an executive callout block with colored left border."""
    type_configs = {
        "note": (BOX_BG_BLUE, ACCENT_BLUE, "NOTE: "),
        "math": (BOX_BG_MATH, ACCENT_PURPLE, "MATHEMATICAL DEFINITION: "),
        "warn": (BOX_BG_WARN, ACCENT_RED, "CRITICAL WARNING: "),
        "insight": (BOX_BG_SUCC, ACCENT_GREEN, "DATA SCIENCE INSIGHT: "),
    }
    bg, border_col, prefix = type_configs.get(callout_type, type_configs["note"])

    h_p = Paragraph(f"<b>{prefix}{title}</b>", styles["body"])
    b_p = Paragraph(text, styles["body"])
    content = [h_p, b_p]

    t = Table([[content]], colWidths=[516])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("LINEBEFORE", (0, 0), (0, -1), 4, border_col),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    return t


def make_code_block(code_lines: list[str], styles: dict) -> Table:
    """Render a styled monospaced code snippet card."""
    formatted = [Paragraph(line.replace(" ", "&nbsp;"), styles["code"]) for line in code_lines]
    t = Table([[formatted]], colWidths=[516])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F1F5F9")),
                ("BOX", (0, 0), (-1, -1), 1, BORDER_LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    return t


# ── Vector Diagrams Generated with ReportLab Shapes ──────────────────────────

def create_cosine_diagram() -> Drawing:
    """Diagram 1: Cosine Similarity & Vector Dot Product Geometry."""
    d = Drawing(516, 170)
    # Background card
    d.add(Rect(0, 0, 516, 170, rx=6, ry=6, fillColor=HexColor("#F8FAFC"), strokeColor=BORDER_LIGHT, strokeWidth=1))

    # Title
    d.add(String(16, 148, "Geometric Derivation of Cosine Similarity in Latent Space", fontName="Helvetica-Bold", fontSize=10, fillColor=PRIMARY))

    # Coordinate Axes
    ox, oy = 70, 30
    d.add(Line(ox, oy, ox + 180, oy, strokeColor=TEXT_MUTED, strokeWidth=1.2))  # X axis
    d.add(Line(ox, oy, ox, oy + 100, strokeColor=TEXT_MUTED, strokeWidth=1.2))  # Y axis
    d.add(String(ox + 185, oy - 3, "Dimension i", fontName="Helvetica", fontSize=8, fillColor=TEXT_MUTED))
    d.add(String(ox - 35, oy + 98, "Dimension j", fontName="Helvetica", fontSize=8, fillColor=TEXT_MUTED))

    # Vector U (Query)
    ux, uy = ox + 130, oy + 70
    d.add(Line(ox, oy, ux, uy, strokeColor=ACCENT_BLUE, strokeWidth=2.2))
    d.add(Circle(ux, uy, 3.5, fillColor=ACCENT_BLUE, strokeColor=ACCENT_BLUE))
    d.add(String(ux + 8, uy + 2, "Vector u (Query Vector)", fontName="Helvetica-Bold", fontSize=9, fillColor=ACCENT_BLUE))

    # Vector V (Document Passage)
    vx, vy = ox + 150, oy + 25
    d.add(Line(ox, oy, vx, vy, strokeColor=ACCENT_ORANGE, strokeWidth=2.2))
    d.add(Circle(vx, vy, 3.5, fillColor=ACCENT_ORANGE, strokeColor=ACCENT_ORANGE))
    d.add(String(vx + 8, vy - 2, "Vector v (Document Chunk)", fontName="Helvetica-Bold", fontSize=9, fillColor=ACCENT_ORANGE))

    # Angle arc (theta)
    d.add(String(ox + 48, oy + 16, "θ", fontName="Helvetica-Bold", fontSize=11, fillColor=ACCENT_PURPLE))

    # Formula Box on Right Side
    d.add(Rect(270, 20, 230, 115, rx=4, ry=4, fillColor=HexColor("#FFFFFF"), strokeColor=BORDER_LIGHT, strokeWidth=1))
    d.add(String(282, 114, "Fundamental Formula:", fontName="Helvetica-Bold", fontSize=9, fillColor=PRIMARY))
    d.add(String(282, 96, "cos(θ) = ( u · v ) / ( ||u||₂ · ||v||₂ )", fontName="Courier-Bold", fontSize=9.5, fillColor=ACCENT_PURPLE))
    d.add(String(282, 78, "• Dot Product:  u · v = Σ u_i · v_i", fontName="Helvetica", fontSize=8.5, fillColor=TEXT_DARK))
    d.add(String(282, 62, "• L2 Norm:      ||u||₂ = √(Σ u_i²)", fontName="Helvetica", fontSize=8.5, fillColor=TEXT_DARK))
    d.add(String(282, 44, "• Normalized:   cos(θ) = u_norm · v_norm", fontName="Helvetica", fontSize=8.5, fillColor=ACCENT_GREEN))
    d.add(String(282, 28, "• Range:        [-1.0, +1.0] (1.0 = parallel)", fontName="Helvetica", fontSize=8, fillColor=TEXT_MUTED))

    return d


def create_bi_vs_cross_diagram() -> Drawing:
    """Diagram 2: Bi-Encoder vs Cross-Encoder Architecture Comparison."""
    d = Drawing(516, 175)
    d.add(Rect(0, 0, 516, 175, rx=6, ry=6, fillColor=HexColor("#F8FAFC"), strokeColor=BORDER_LIGHT, strokeWidth=1))
    d.add(String(16, 154, "Architectural Comparison: Bi-Encoder (Embedding) vs Cross-Encoder (Reranker)", fontName="Helvetica-Bold", fontSize=10, fillColor=PRIMARY))

    # Left: Bi-Encoder
    bx = 20
    d.add(Rect(bx, 18, 225, 124, rx=4, ry=4, fillColor=HexColor("#FFFFFF"), strokeColor=BORDER_LIGHT, strokeWidth=1))
    d.add(String(bx + 12, 126, "A. Bi-Encoder (Dense Embeddings)", fontName="Helvetica-Bold", fontSize=9, fillColor=ACCENT_BLUE))
    d.add(Rect(bx + 12, 92, 90, 24, rx=3, ry=3, fillColor=HexColor("#EFF6FF"), strokeColor=ACCENT_BLUE, strokeWidth=1))
    d.add(String(bx + 24, 99, "Query Text", fontName="Helvetica", fontSize=8, fillColor=ACCENT_BLUE))
    d.add(Rect(bx + 120, 92, 92, 24, rx=3, ry=3, fillColor=HexColor("#FFF7ED"), strokeColor=ACCENT_ORANGE, strokeWidth=1))
    d.add(String(bx + 126, 99, "Document Text", fontName="Helvetica", fontSize=8, fillColor=ACCENT_ORANGE))

    d.add(Line(bx + 57, 92, bx + 57, 72, strokeColor=TEXT_MUTED, strokeWidth=1))
    d.add(Line(bx + 166, 92, bx + 166, 72, strokeColor=TEXT_MUTED, strokeWidth=1))

    d.add(Rect(bx + 12, 54, 90, 18, rx=2, ry=2, fillColor=HexColor("#F1F5F9"), strokeColor=BORDER_LIGHT, strokeWidth=0.8))
    d.add(String(bx + 26, 59, "Vector u (768d)", fontName="Courier", fontSize=7.5, fillColor=TEXT_DARK))
    d.add(Rect(bx + 120, 54, 92, 18, rx=2, ry=2, fillColor=HexColor("#F1F5F9"), strokeColor=BORDER_LIGHT, strokeWidth=0.8))
    d.add(String(bx + 132, 59, "Vector v (768d)", fontName="Courier", fontSize=7.5, fillColor=TEXT_DARK))

    d.add(Rect(bx + 40, 24, 145, 20, rx=3, ry=3, fillColor=HexColor("#ECFDF5"), strokeColor=ACCENT_GREEN, strokeWidth=1))
    d.add(String(bx + 48, 30, "Cosine Sim = u · v  (O(d))", fontName="Helvetica-Bold", fontSize=8, fillColor=ACCENT_GREEN))

    # Right: Cross-Encoder
    cx = 265
    d.add(Rect(cx, 18, 235, 124, rx=4, ry=4, fillColor=HexColor("#FFFFFF"), strokeColor=BORDER_LIGHT, strokeWidth=1))
    d.add(String(cx + 12, 126, "B. Cross-Encoder (Full Reranker)", fontName="Helvetica-Bold", fontSize=9, fillColor=ACCENT_PURPLE))

    d.add(Rect(cx + 12, 92, 210, 24, rx=3, ry=3, fillColor=HexColor("#F5F3FF"), strokeColor=ACCENT_PURPLE, strokeWidth=1))
    d.add(String(cx + 26, 99, "[CLS] Query [SEP] Document [SEP]", fontName="Courier-Bold", fontSize=8, fillColor=ACCENT_PURPLE))

    d.add(Line(cx + 117, 92, cx + 117, 72, strokeColor=TEXT_MUTED, strokeWidth=1))

    d.add(Rect(cx + 25, 52, 185, 20, rx=3, ry=3, fillColor=HexColor("#F1F5F9"), strokeColor=BORDER_LIGHT, strokeWidth=0.8))
    d.add(String(cx + 36, 58, "All-to-All Full Cross Attention", fontName="Helvetica-Bold", fontSize=8, fillColor=TEXT_DARK))

    d.add(Line(cx + 117, 52, cx + 117, 44, strokeColor=TEXT_MUTED, strokeWidth=1))

    d.add(Rect(cx + 40, 24, 155, 20, rx=3, ry=3, fillColor=HexColor("#EFF6FF"), strokeColor=ACCENT_BLUE, strokeWidth=1))
    d.add(String(cx + 48, 30, "Relevance Score (Sigmoid/Logit)", fontName="Helvetica-Bold", fontSize=8, fillColor=ACCENT_BLUE))

    return d


def create_rag_pipeline_diagram() -> Drawing:
    """Diagram 3: End-to-End Modern RAG Pipeline Architecture."""
    d = Drawing(516, 160)
    d.add(Rect(0, 0, 516, 160, rx=6, ry=6, fillColor=HexColor("#F8FAFC"), strokeColor=BORDER_LIGHT, strokeWidth=1))
    d.add(String(16, 142, "The Production RAG Architecture: Dual-Channel Retrieval to NLI Verification", fontName="Helvetica-Bold", fontSize=10, fillColor=PRIMARY))

    # Step blocks
    blocks = [
        ("1. User Query", "Raw Lay Term", ACCENT_BLUE, 16),
        ("2. Profiler & Router", "6D Vector Check", ACCENT_PURPLE, 114),
        ("3. Hybrid Retrieval", "Dense + BM25 RRF", ACCENT_ORANGE, 212),
        ("4. LLM Generator", "Local LLaMA 3.2B", ACCENT_CYAN, 310),
        ("5. NLI Attribution", "DeBERTa Grounding", ACCENT_GREEN, 408),
    ]

    for title, subtitle, col, x in blocks:
        d.add(Rect(x, 48, 92, 68, rx=4, ry=4, fillColor=HexColor("#FFFFFF"), strokeColor=col, strokeWidth=1.2))
        d.add(Rect(x, 96, 92, 20, rx=4, ry=4, fillColor=col, strokeColor=col))
        d.add(String(x + 6, 102, title[:14], fontName="Helvetica-Bold", fontSize=7.5, fillColor=HexColor("#FFFFFF")))
        d.add(String(x + 6, 76, title[3:], fontName="Helvetica-Bold", fontSize=8, fillColor=TEXT_DARK))
        d.add(String(x + 6, 60, subtitle, fontName="Helvetica", fontSize=7, fillColor=TEXT_MUTED))

        # Connecting arrow
        if x < 408:
            arr_x = x + 92
            d.add(Line(arr_x + 1, 82, arr_x + 5, 82, strokeColor=TEXT_MUTED, strokeWidth=1.5))

    # Bottom closed-loop feedback arrow
    d.add(Line(454, 48, 454, 22, strokeColor=ACCENT_RED, strokeWidth=1.2))
    d.add(Line(454, 22, 160, 22, strokeColor=ACCENT_RED, strokeWidth=1.2))
    d.add(Line(160, 22, 160, 48, strokeColor=ACCENT_RED, strokeWidth=1.2))
    d.add(String(190, 26, "Pillar 6: Closed-Loop Active Learning (Permanent Rule Injection on Error)", fontName="Helvetica-Bold", fontSize=7.5, fillColor=ACCENT_RED))

    return d


def create_nli_verdict_diagram() -> Drawing:
    """Diagram 4: Sentence-Level Natural Language Inference (NLI) Verification."""
    d = Drawing(516, 155)
    d.add(Rect(0, 0, 516, 155, rx=6, ry=6, fillColor=HexColor("#F8FAFC"), strokeColor=BORDER_LIGHT, strokeWidth=1))
    d.add(String(16, 137, "Pillar 5: Sentence-Level Natural Language Inference (DeBERTa-v3 Cross-Encoder)", fontName="Helvetica-Bold", fontSize=10, fillColor=PRIMARY))

    # Premise Box
    d.add(Rect(20, 72, 220, 50, rx=4, ry=4, fillColor=HexColor("#FFFFFF"), strokeColor=BORDER_LIGHT, strokeWidth=1))
    d.add(String(28, 106, "Premise P (Retrieved Manual Snippet):", fontName="Helvetica-Bold", fontSize=8, fillColor=ACCENT_BLUE))
    d.add(String(28, 92, '"▪ Use end mills made of carbide material', fontName="Courier", fontSize=7.5, fillColor=TEXT_DARK))
    d.add(String(28, 80, ' ▪ Select highest speed (4,500 rpm)"', fontName="Courier", fontSize=7.5, fillColor=TEXT_DARK))

    # Hypothesis Box
    d.add(Rect(20, 16, 220, 46, rx=4, ry=4, fillColor=HexColor("#FFFFFF"), strokeColor=BORDER_LIGHT, strokeWidth=1))
    d.add(String(28, 46, "Hypothesis H (Generated Answer Claim):", fontName="Helvetica-Bold", fontSize=8, fillColor=ACCENT_ORANGE))
    d.add(String(28, 32, '"Use carbide end mills at maximum speed."', fontName="Courier", fontSize=7.5, fillColor=TEXT_DARK))

    # Arrow to Cross-Encoder
    d.add(Line(240, 68, 275, 68, strokeColor=TEXT_MUTED, strokeWidth=1.5))

    # Softmax Classification Box
    d.add(Rect(275, 16, 225, 106, rx=4, ry=4, fillColor=HexColor("#FFFFFF"), strokeColor=BORDER_LIGHT, strokeWidth=1))
    d.add(String(287, 104, "DeBERTa-v3 Softmax Probabilities:", fontName="Helvetica-Bold", fontSize=8.5, fillColor=PRIMARY))

    # Verdict 1: Entailment
    d.add(Rect(287, 76, 200, 20, rx=3, ry=3, fillColor=HexColor("#ECFDF5"), strokeColor=ACCENT_GREEN, strokeWidth=1))
    d.add(String(295, 82, "P(Entailment) = 0.983  ──►  🟢 Entailed (True)", fontName="Helvetica-Bold", fontSize=8, fillColor=ACCENT_GREEN))

    # Verdict 2: Neutral
    d.add(Rect(287, 50, 200, 20, rx=3, ry=3, fillColor=HexColor("#FFFBEB"), strokeColor=HexColor("#D97706"), strokeWidth=1))
    d.add(String(295, 56, "P(Neutral) = 0.013     ──►  🟡 Ungrounded", fontName="Helvetica-Bold", fontSize=8, fillColor=HexColor("#D97706")))

    # Verdict 3: Contradiction
    d.add(Rect(287, 24, 200, 20, rx=3, ry=3, fillColor=HexColor("#FEF2F2"), strokeColor=ACCENT_RED, strokeWidth=1))
    d.add(String(295, 30, "P(Contradiction) = 0.004 ──►  🔴 Hallucination", fontName="Helvetica-Bold", fontSize=8, fillColor=ACCENT_RED))

    return d


def build_handbook_story(styles: dict) -> list:
    """Compile the complete textbook contents into Platypus flowables."""
    story = []

    # ═════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 40))
    story.append(
        Paragraph(
            "<font color='#EA580C'><b>DATA SCIENCE & AI ENGINEERING HANDBOOK</b></font>",
            styles["subtitle"],
        )
    )
    story.append(
        Paragraph(
            "Foundations of Modern RAG,<br/>Embeddings & Vector Search",
            styles["title"],
        )
    )
    story.append(
        Paragraph(
            "From Mathematical First Principles to Production Grounding and Zero-Defect Safety",
            styles["subtitle"],
        )
    )

    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY, spaceAfter=20))

    intro_box = (
        "<b>About This Textbook:</b> This handbook is designed for Data Scientists, Machine Learning Engineers, "
        "and applied researchers seeking deep theoretical mastery and production implementation skills in "
        "Retrieval-Augmented Generation (RAG). It systematically demystifies dense vector representations, "
        "Approximate Nearest Neighbors (ANN) indexing, classical lexical search (BM25), Reciprocal Rank Fusion, "
        "transformer attention mechanics, and Natural Language Inference (NLI) factual grounding."
    )
    story.append(make_callout("CURRICULUM & PURPOSE", intro_box, "note", styles))
    story.append(Spacer(1, 20))

    story.append(create_rag_pipeline_diagram())
    story.append(Spacer(1, 25))

    meta_table = [
        [
            Paragraph("<b>Target Audience:</b> Junior to Senior Data Scientists", styles["body"]),
            Paragraph("<b>Mathematical Depth:</b> Linear Algebra, Probability, Information Theory", styles["body"]),
        ],
        [
            Paragraph("<b>Architecture:</b> 6-Pillar Terminology RAG", styles["body"]),
            Paragraph("<b>Empirical Dataset:</b> REHAU RAUVISIO Technical Corpus", styles["body"]),
        ],
    ]
    t_meta = Table(meta_table, colWidths=[258, 258])
    t_meta.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, BORDER_LIGHT),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(t_meta)

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Table of Contents", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_LIGHT, spaceAfter=14))

    toc_items = [
        ("Chapter 1: Mathematical Foundations & Prerequisites", "Vector Spaces, L2 Norm, Cosine Derivation, Softmax, Shannon Entropy"),
        ("Chapter 2: The Evolution of Text Representation & Embeddings", "One-Hot, TF-IDF, Word2Vec, Transformer Attention, Dense Bi-Encoders"),
        ("Chapter 3: Vector Databases & Approximate Nearest Neighbors (ANN)", "The O(N) Search Dilemma, HNSW Multi-Layer Skip Graphs, ChromaDB Engine"),
        ("Chapter 4: The Classical Complement — Okapi BM25", "Inverted Indexes, Term Saturation k1, Length Penalty b, Exact Code Matching"),
        ("Chapter 5: Retrieval-Augmented Generation (RAG) Architectures", "The RAG Triad, The Vocabulary Mismatch Dilemma, Advanced Chunking Science"),
        ("Chapter 6: Hybrid Retrieval, Fusion Math & The Re-Ranking Layer", "Reciprocal Rank Fusion (RRF), The Negative Transfer Trap, Cross-Encoders"),
        ("Chapter 7: Large Language Models, Sampling & Degeneration Traps", "Autoregressive Sampling, Temperature, Top-p, Mode Collapse in <1B Models"),
        ("Chapter 8: Answer Grounding, Hallucination Shields & Sentence NLI", "Cross-Encoder Attribution, Faithfulness Ratio, Active Learning Feedback"),
        ("Chapter 9: The Data Scientist's Mastery Blueprint", "Master Formula Matrix, Production Debugging Checklist, Learning Roadmap"),
    ]

    for ch_title, ch_desc in toc_items:
        story.append(Paragraph(f"<b>{ch_title}</b>", styles["h3"]))
        story.append(Paragraph(f"<font color='#64748B'>{ch_desc}</font>", styles["body"]))
        story.append(Spacer(1, 4))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 1: MATHEMATICAL FOUNDATIONS & PREREQUISITES
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 1: Mathematical Foundations & Prerequisites", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_LIGHT, spaceAfter=12))

    story.append(
        Paragraph(
            "Before an algorithm can search, cluster, or synthesize natural language, words must be translated into "
            "rigid geometric objects. This chapter builds the indispensable mathematical toolkit required to master "
            "embeddings, vector distances, probability distributions, and information entropy.",
            styles["body"],
        )
    )

    story.append(Paragraph("1.1 Vector Spaces and Coordinate Representations", styles["h2"]))
    story.append(
        Paragraph(
            "A vector <b>u</b> is an ordered collection of <i>d</i> real numbers existing in a <i>d</i>-dimensional vector "
            "space ℝ<sup>d</sup>. In modern machine learning models (such as Qwen3-Embedding or BERT), <i>d</i> is typically "
            "768, 1024, or 1536. Each coordinate corresponds to a continuous latent semantic feature discovered during training.",
            styles["body"],
        )
    )

    story.append(Paragraph("1.2 The Dot Product (Inner Product) & Vector Norms", styles["h2"]))
    story.append(
        Paragraph(
            "The algebraic dot product between two vectors <b>u</b> = [u₁, u₂, ..., u_d] and <b>v</b> = [v₁, v₂, ..., v_d] "
            "is defined as the sum of their element-wise products:",
            styles["body"],
        )
    )
    story.append(Paragraph("<b>u · v = Σ_{i=1}^d ( u_i · v_i )</b>", styles["formula"]))
    story.append(
        Paragraph(
            "Geometrically, the dot product measures how much vector <b>u</b> aligns with vector <b>v</b>, scaled by their lengths. "
            "The magnitude (or Euclidean length) of a vector is calculated via the <b>L₂ norm</b>:",
            styles["body"],
        )
    )
    story.append(Paragraph("<b>||u||₂ = √( Σ_{i=1}^d u_i² ) = √( u · u )</b>", styles["formula"]))
    story.append(
        Paragraph(
            "A vector is called a <b>unit vector</b> if ||<b>u</b>||₂ = 1. Dividing any non-zero vector by its L₂ norm "
            "(<b>u_norm = u / ||u||₂</b>) projects it onto the surface of a unit hypersphere.",
            styles["body"],
        )
    )

    story.append(Paragraph("1.3 Geometric Derivation of Cosine Similarity", styles["h2"]))
    story.append(
        Paragraph(
            "From Euclidean geometry, the dot product satisfies the fundamental identity: <b>u · v = ||u||₂ ||v||₂ cos(θ)</b>, "
            "where θ is the angle between the two vectors. Rearranging gives <b>Cosine Similarity</b>:",
            styles["body"],
        )
    )
    story.append(Paragraph("<b>cos(θ) = ( u · v ) / ( ||u||₂ · ||v||₂ )</b>", styles["formula"]))

    story.append(create_cosine_diagram())
    story.append(Spacer(1, 10))

    cos_insight = (
        "<b>Why Cosine Similarity Dominates Information Retrieval:</b> "
        "Unlike Euclidean distance (which grows larger if a document is simply longer and repeats words), "
        "cosine similarity is <b>length-invariant</b>. It only measures the <i>directional angle θ</i> between query and document. "
        "If a 50-word passage and a 500-word passage discuss the exact same topic, their normalized vectors point in nearly "
        "the same direction (θ ≈ 0°, cos(θ) ≈ 1.0), regardless of total word count."
    )
    story.append(make_callout("LENGTH INVARIANCE IN VECTOR SEARCH", cos_insight, "insight", styles))

    story.append(Paragraph("1.4 The Softmax Function & Logits", styles["h2"]))
    story.append(
        Paragraph(
            "Neural networks output raw, unconstrained real numbers called <b>logits</b> (z ∈ ℝ<sup>K</sup>). To convert logits "
            "into a valid probability distribution where each probability p_i ∈ [0, 1] and Σ p_i = 1, we use the <b>Softmax function</b>:",
            styles["body"],
        )
    )
    story.append(Paragraph("<b>P(class = i) = σ(z)_i = e^{z_i} / ( Σ_{j=1}^K e^{z_j} )</b>", styles["formula"]))
    story.append(
        Paragraph(
            "<b>Numerical Stability Tip:</b> In floating-point arithmetic, computing e^{z_i} directly can cause numerical overflow "
            "if z_i > 700. In production code, always subtract the maximum logit before exponentiating: "
            "<i>z' = z - max(z)</i>. This produces mathematically identical probabilities while preventing inf/nan errors.",
            styles["body"],
        )
    )

    story.append(Paragraph("1.5 Shannon Entropy & Retrieval Ambiguity", styles["h2"]))
    story.append(
        Paragraph(
            "In Pillar 1 of our architecture, the router diagnoses whether vector search is confident or confused by calculating "
            "<b>Shannon Entropy</b> across the candidate similarity scores:",
            styles["body"],
        )
    )
    story.append(Paragraph("<b>H(P) = - Σ_{i=1}^k  p_i · log₂(p_i)</b>", styles["formula"]))
    story.append(
        Paragraph(
            "• If one document dominates (e.g. p = [0.95, 0.03, 0.02]), entropy is low (H ≈ 0.3) → High confidence.<br/>"
            "• If scores are diffuse and uncertain (e.g. p = [0.20, 0.20, 0.20, 0.20, 0.20]), entropy is maximal (H ≈ 2.32) "
            "→ High ambiguity. The router uses high entropy to trigger terminology expansion.",
            styles["body"],
        )
    )

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 2: THE EVOLUTION OF TEXT REPRESENTATION & EMBEDDINGS
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 2: The Evolution of Text Representation & Embeddings", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_LIGHT, spaceAfter=12))

    story.append(
        Paragraph(
            "How does raw English or technical German text become a continuous vector? The field of Natural Language "
            "Processing evolved through four distinct eras:",
            styles["body"],
        )
    )

    story.append(Paragraph("2.1 The Four Eras of Text Encoding", styles["h2"]))

    eras = [
        ("Era 1: One-Hot Encoding", "Every word in vocabulary V is assigned a distinct binary vector with a single 1 and |V|-1 zeros. Vectors are orthogonal: (cat · dog = 0). Zero semantic awareness, massive sparsity."),
        ("Era 2: TF-IDF (Term Frequency-Inverse Document Frequency)", "Weights words by frequency in document discounted by frequency across corpus. Captures important keywords, but word order is completely lost ('man bites dog' == 'dog bites man')."),
        ("Era 3: Static Word Embeddings (Word2Vec, GloVe)", "Dense 300d vectors trained via Skip-Gram or CBOW. Words appearing in similar contexts embed close together. Major flaw: Polysemy. 'Apple' (fruit) and 'Apple' (company) have the exact same vector."),
        ("Era 4: Contextual Transformer Embeddings (BERT, Qwen, RoBERTa)", "Self-attention generates dynamic embeddings where every word's vector is shaped by its sentence context. Solves polysemy, syntactic nuance, and cross-lingual alignment."),
    ]
    for e_name, e_desc in eras:
        story.append(Paragraph(f"• <b>{e_name}:</b> {e_desc}", styles["bullet"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("2.2 The Self-Attention Engine", styles["h2"]))
    story.append(
        Paragraph(
            "At the core of modern embedding models is the Transformer <b>Scaled Dot-Product Attention</b> mechanism. "
            "Given input token vectors, the model projects them into three matrices: <b>Queries (Q)</b>, <b>Keys (K)</b>, "
            "and <b>Values (V)</b>:",
            styles["body"],
        )
    )
    story.append(Paragraph("<b>Attention(Q, K, V) = softmax( ( Q · Kᵀ ) / √d_k ) · V</b>", styles["formula"]))
    story.append(
        Paragraph(
            "1. <b>Q · Kᵀ:</b> Measures the pairwise compatibility between every token and every other token.<br/>"
            "2. <b>√d_k:</b> Prevents inner products from growing excessively large in high dimensions (which would push softmax into regions with vanishing gradients).<br/>"
            "3. <b>softmax(...):</b> Converts alignment scores into an attention weight distribution.<br/>"
            "4. <b>· V:</b> Computes a weighted sum of value vectors, allowing words to absorb meaning from their neighbors.",
            styles["body"],
        )
    )

    story.append(Paragraph("2.3 From Token Embeddings to Sentence Embeddings: Pooling", styles["h2"]))
    story.append(
        Paragraph(
            "A transformer outputs a hidden state vector for <i>every individual token</i> in the input sentence. "
            "To represent the <i>entire sentence</i> as a single vector for database storage, we apply <b>Pooling</b>:",
            styles["body"],
        )
    )

    pool_methods = [
        ("Mean Pooling (Standard in SBERT & Qwen):", "Computes the element-wise average of all non-padding token vectors: <b>v_doc = (1 / L) Σ_{t=1}^L h_t</b>. Captures balanced topic semantics across the whole passage."),
        ("[CLS] Token Pooling:", "Uses the vector of the special classification token prepended to the input. Effective for classification, but often underperforms mean pooling on semantic search."),
        ("L₂ Normalization:", "Always applied after pooling: <b>v_final = v_doc / ||v_doc||₂</b>. Ensures all document vectors have length 1.0, enabling lightning-fast dot-product similarity."),
    ]
    for pm, pdesc in pool_methods:
        story.append(Paragraph(f"• <b>{pm}</b> {pdesc}", styles["bullet"]))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 3: VECTOR DATABASES & APPROXIMATE NEAREST NEIGHBORS (ANN)
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 3: Vector Databases & Approximate Nearest Neighbors (ANN)", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_LIGHT, spaceAfter=12))

    story.append(
        Paragraph(
            "Once documents are embedded into dense vectors, how do we search across millions of vectors in under 10 milliseconds? "
            "This chapter explores why exact search fails and how graph-based indexing makes real-time AI possible.",
            styles["body"],
        )
    )

    story.append(Paragraph("3.1 The Computational Complexity of Exact Search", styles["h2"]))
    story.append(
        Paragraph(
            "In exact <b>K-Nearest Neighbors (KNN)</b>, querying a database of <i>N</i> vectors of dimension <i>d</i> requires "
            "computing distance against every single vector. The computational complexity is <b>O(N · d)</b>.<br/>"
            "• For 100,000 document chunks of dimension 768, each query requires <b>76,800,000 floating-point operations</b>.<br/>"
            "• At web scale (10,000,000 vectors), exact search requires seconds of latency per query, rendering real-time RAG impossible.",
            styles["body"],
        )
    )

    story.append(Paragraph("3.2 Approximate Nearest Neighbors (ANN) & The HNSW Graph", styles["h2"]))
    story.append(
        Paragraph(
            "Production vector databases (such as ChromaDB, Milvus, and Pinecone) use <b>Approximate Nearest Neighbors (ANN)</b>. "
            "ANN sacrifices a tiny fraction of accuracy (<0.5% recall drop) to reduce query complexity from <b>O(N)</b> down to <b>O(log N)</b>.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "The premier indexing algorithm used in ChromaDB is <b>HNSW (Hierarchical Navigable Small World)</b>. "
            "Inspired by multi-level skip lists, HNSW constructs a hierarchy of geometric proximity graphs:",
            styles["body"],
        )
    )

    hnsw_steps = [
        ("Top Layers (Sparse):", "Contain few nodes connected by long-distance edges. The query takes rapid, coarse-grained hops across the hyperspace to find the approximate neighborhood in O(1) steps."),
        ("Bottom Layers (Dense):", "Contain all nodes connected by fine-grained local neighborhood edges. The search descends into lower layers to execute precise local greedy routing."),
        ("Termination:", "The search halts when no neighbor closer to the query can be found, returning the top-k nearest nodes in sub-millisecond latency."),
    ]
    for h_name, h_desc in hnsw_steps:
        story.append(Paragraph(f"• <b>{h_name}</b> {h_desc}", styles["bullet"]))

    story.append(Spacer(1, 10))
    ann_callout = (
        "<b>ChromaDB Architecture in This Project:</b><br/>"
        "ChromaDB stores document text, chunk IDs, and JSON metadata inside SQLite (<code>chroma.sqlite3</code>), "
        "while vector embeddings are indexed using an optimized C++ HNSW graph stored in binary index files. "
        "When metadata filters are specified (e.g. <code>where={'product': 'RAUVISIO crystal'}</code>), ChromaDB "
        "prunes the search graph to only traverse documents matching the specified product line."
    )
    story.append(make_callout("CHROMADB DUAL-ENGINE STORAGE", ann_callout, "note", styles))

    story.append(Paragraph("3.3 Distance Metrics Comparison", styles["h2"]))

    dist_table = [
        [
            Paragraph("<b>Metric Name</b>", styles["body"]),
            Paragraph("<b>Mathematical Formula</b>", styles["body"]),
            Paragraph("<b>Best Used For</b>", styles["body"]),
            Paragraph("<b>Key Properties</b>", styles["body"]),
        ],
        [
            Paragraph("<b>Cosine Distance</b>", styles["body"]),
            Paragraph("d = 1 - cos(θ)", styles["code"]),
            Paragraph("Natural Language & RAG", styles["body"]),
            Paragraph("Length-invariant, values ∈ [0, 2]", styles["body"]),
        ],
        [
            Paragraph("<b>Euclidean (L₂)</b>", styles["body"]),
            Paragraph("d = √( Σ (u_i - v_i)² )", styles["code"]),
            Paragraph("Image & Audio Embeddings", styles["body"]),
            Paragraph("Sensitive to vector magnitude", styles["body"]),
        ],
        [
            Paragraph("<b>Inner Product (MIPS)</b>", styles["body"]),
            Paragraph("s = u · v", styles["code"]),
            Paragraph("Recommendation Systems", styles["body"]),
            Paragraph("Fastest on hardware; requires normalized vectors", styles["body"]),
        ],
    ]
    t_dist = Table(dist_table, colWidths=[110, 140, 130, 136])
    t_dist.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#F1F5F9")),
                ("BOX", (0, 0), (-1, -1), 1, BORDER_LIGHT),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t_dist)

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 4: THE CLASSICAL COMPLEMENT — OKAPI BM25
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 4: The Classical Search Complement — Okapi BM25", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_LIGHT, spaceAfter=12))

    story.append(
        Paragraph(
            "Why do we need a 30-year-old lexical algorithm when we have billion-parameter neural networks? "
            "Because dense embeddings have a fatal blind spot that BM25 solves effortlessly.",
            styles["body"],
        )
    )

    story.append(Paragraph("4.1 The Blind Spot of Dense Vectors", styles["h2"]))
    story.append(
        Paragraph(
            "Dense embedding models compress an entire 2,000-character technical manual passage into just 768 numbers. "
            "In doing so, fine-grained details are often compressed away:<br/>"
            "• <b>Exact Numeric Tolerances:</b> Embeddings often treat 'expansion gap 2 mm' and 'expansion gap 5 mm' as 99% identical.<br/>"
            "• <b>Rare Part Numbers & Standards:</b> Rare tokens like <i>DIN 68861</i> or <i>CARB2</i> appear infrequently during "
            "pre-training and get blurred into generic furniture concepts.<br/>"
            "• <b>Specific Speed Parameters:</b> '4,500 rpm' is treated as semantically similar to '1,000 rpm'.",
            styles["body"],
        )
    )

    story.append(Paragraph("4.2 The Okapi BM25 Mathematical Formulation", styles["h2"]))
    story.append(
        Paragraph(
            "<b>BM25 (Best Matching 25)</b> is a probabilistic bag-of-words ranking algorithm that indexes documents "
            "using an <b>Inverted Index</b>. The relevance score between document <i>D</i> and query <i>Q</i> is:",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>BM25(D, Q) = Σ_{q ∈ Q}  IDF(q) · [ ( f(q, D) · (k₁ + 1) ) / ( f(q, D) + k₁ · (1 - b + b · (|D| / avgdl)) ) ]</b>",
            styles["formula"],
        )
    )

    bm25_params = [
        ("f(q, D) (Term Frequency):", "How many times query term q appears in document D."),
        ("k₁ Parameter (Term Saturation, default = 1.2 to 2.0):", "Controls how quickly the score saturates as word count increases. Unlike raw TF, repeating a keyword 50 times in BM25 does not yield 50x the score; it asymptotically plateaus."),
        ("b Parameter (Length Normalization, default = 0.75):", "Penalizes long, verbose documents (|D| > avgdl) to ensure long documents with incidental keyword occurrences don't outrank concise, highly relevant passages."),
        ("IDF(q) (Inverse Document Frequency):", "IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 ). Heavily rewards rare, informative tokens (like 'RAUVISIO' or '68861') while heavily penalizing common stopwords (like 'the' or 'with')."),
    ]
    for b_param, b_desc in bm25_params:
        story.append(Paragraph(f"• <b>{b_param}</b> {b_desc}", styles["bullet"]))

    story.append(Spacer(1, 10))

    bm25_summary = (
        "<b>The Complementary Strengths of Dense vs. Lexical Search:</b><br/>"
        "• <b>Dense Search (Qwen3-Embedding):</b> Excels at semantic concepts, synonyms, high-level intent, and answering colloquial questions.<br/>"
        "• <b>Lexical Search (BM25 Okapi):</b> Excels at exact numeric codes, DIN standards, millimeters, and verbatim product line names.<br/>"
        "<i>Combining both in a hybrid pipeline guarantees that neither conceptual meaning nor exact specifications are missed.</i>"
    )
    story.append(make_callout("THE CORE DUALITY", bm25_summary, "insight", styles))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 5: RETRIEVAL-AUGMENTED GENERATION (RAG) ARCHITECTURES
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 5: Retrieval-Augmented Generation (RAG) Architectures", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_LIGHT, spaceAfter=12))

    story.append(
        Paragraph(
            "Large Language Models (LLMs) suffer from two fundamental weaknesses: their knowledge is frozen at training time, "
            "and they hallucinate when questioned on specialized proprietary documentation. "
            "<b>RAG (Retrieval-Augmented Generation)</b> anchors the LLM to verified factual context.",
            styles["body"],
        )
    )

    story.append(Paragraph("5.1 The Naive RAG Architecture & Its Fatal Flaws", styles["h2"]))
    story.append(
        Paragraph(
            "In <b>Naive RAG</b>, text files are split into arbitrary chunks (e.g. 500 characters), embedded into a vector database, "
            "retrieved via top-k cosine similarity, and prepended directly to the LLM prompt. In industrial production, "
            "Naive RAG collapses due to three critical failure modes:",
            styles["body"],
        )
    )

    naive_flaws = [
        ("Failure 1: The Lay-Expert Vocabulary Mismatch:", "When shopfloor operators ask colloquial questions ('crystal glass'), naive vector search retrieves general glass/window manuals, completely missing the fact that REHAU's product is an engineered PMMA polymer laminate."),
        ("Failure 2: 'Lost in the Middle' (Liu et al., 2023):", "Transformers allocate peak attention to the beginning and end of the input context. When multiple noisy chunks are fed into the prompt, the LLM frequently misses facts buried in the middle passages."),
        ("Failure 3: Context Starvation & Hallucination Pressure:", "If the retriever fails to retrieve the gold chunk in the top candidates, the LLM is starved of evidence. Eager to respond, it invents plausible-sounding falsehoods (e.g. 'clean using standard ammonia glass spray'), ruining workpieces."),
    ]
    for nf, ndesc in naive_flaws:
        story.append(Paragraph(f"• <b>{nf}</b> {ndesc}", styles["bullet"]))

    story.append(Paragraph("5.2 Document Chunking Science", styles["h2"]))
    story.append(
        Paragraph(
            "Chunking is the foundation of retrieval quality. If chunks are too small, context is severed. "
            "If chunks are too large, vector representations get diluted.",
            styles["body"],
        )
    )

    chunk_methods = [
        ("Recursive Character Splitting:", "Splits along natural syntactic boundaries in order: paragraphs (\\n\\n) → lines (\\n) → sentences (. ) → words ( ). Guarantees paragraphs remain coherent units."),
        ("Chunk Overlap (Sliding Window):", "Maintains 50-60 characters of overlap between consecutive chunks. Prevents key sentences that sit exactly on a boundary from being cut in half."),
        ("Technical Bullet-Point Preservation:", "Engineering manuals convey safety instructions using bullet symbols (▪, •). Our chunking pipeline preserves bullet boundaries so that individual procedural instructions can be indexed and evaluated atomically."),
    ]
    for cm, cdesc in chunk_methods:
        story.append(Paragraph(f"• <b>{cm}</b> {cdesc}", styles["bullet"]))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 6: HYBRID RETRIEVAL, FUSION MATH & THE RE-RANKING LAYER
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 6: Hybrid Retrieval, Fusion Math & The Re-Ranking Layer", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_LIGHT, spaceAfter=12))

    story.append(
        Paragraph(
            "To achieve state-of-the-art retrieval, we combine dense vectors and lexical BM25 using rank-based fusion, "
            "and refine the candidate pool with a deep cross-encoder reranker.",
            styles["body"],
        )
    )

    story.append(Paragraph("6.1 Reciprocal Rank Fusion (RRF) Formulation", styles["h2"]))
    story.append(
        Paragraph(
            "Why can't we simply add dense cosine scores to BM25 scores? Because <b>cosine scores live in [-1.0, 1.0]</b> "
            "while <b>BM25 scores live in [0, ∞)</b> with non-linear distributions. Normalizing them is unstable.<br/>"
            "<b>Reciprocal Rank Fusion (RRF)</b> bypasses score calibration entirely by operating purely on <i>ranks</i>:",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "<b>RRF_Score(d) = Σ_{m ∈ {Dense, BM25}}  [ w_m / ( k + rank_m(d) ) ]</b>",
            styles["formula"],
        )
    )
    story.append(
        Paragraph(
            "• <i>rank_m(d)</i> is the 1-based rank position of document <i>d</i> in model <i>m</i>'s retrieval list.<br/>"
            "• <i>k</i> is the smoothing constant (empirically set to <b>60</b> in Cormack et al., 2009). "
            "The constant 60 ensures that a document ranked #1 receives score 1/61 ≈ 0.0163, while rank #2 receives 1/62 ≈ 0.0161, "
            "dampening the outsized advantage of high ranks while strongly rewarding documents appearing on <i>both</i> lists.<br/>"
            "• <i>w_m</i> is the channel weight (controlled by α in our router).",
            styles["body"],
        )
    )

    story.append(Paragraph("6.2 The Negative Transfer Trap (Fusion Dilution)", styles["h2"]))

    rrf_trap = (
        "<b>Why Did Dense-Only Get 70% While Hybrid Dropped to 56.7% in our Benchmark?</b><br/>"
        "In our evaluation dataset, questions are written in colloquial shopfloor terms ('crystal glass', 'shiny acrylic'). "
        "Because these lay terms do NOT appear verbatim in the engineering manuals, BM25 achieved only <b>30% Recall@5</b>. "
        "When naive 50/50 RRF was applied, BM25's false-positive top ranks pushed true gold documents from Rank 4 down to Rank 7! "
        "This is called <b>Fusion Dilution</b>. This finding proves that a static hybrid strategy is suboptimal; "
        "an <b>Adaptive Router</b> must dynamically down-weight BM25 when terminology mismatch S_term is high."
    )
    story.append(make_callout("THE DATA SCIENCE LESSON", rrf_trap, "warn", styles))
    story.append(Spacer(1, 10))

    story.append(Paragraph("6.3 Two-Stage Retrieval: Bi-Encoder vs Cross-Encoder", styles["h2"]))
    story.append(create_bi_vs_cross_diagram())
    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "• <b>Stage 1: Fast Candidate Retrieval (Bi-Encoder + BM25):</b> Pulls the top 30-50 candidate passages "
            "out of 100,000 documents in ~20 milliseconds.<br/>"
            "• <b>Stage 2: Cross-Encoder Re-Ranking (MS-MARCO-MiniLM-L-6-v2):</b> Concatenates the query and each candidate chunk into "
            "a single sequence <i>[CLS] Query [SEP] Document [SEP]</i> and computes <b>full all-to-all cross-attention</b>. "
            "This evaluates exact term interactions and re-orders the true gold document straight to <b>Rank 1</b>, elevating MRR.",
            styles["body"],
        )
    )

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 7: LARGE LANGUAGE MODELS, SAMPLING & DEGENERATION TRAPS
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 7: Large Language Models, Sampling & Degeneration Traps", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_LIGHT, spaceAfter=12))

    story.append(
        Paragraph(
            "Once relevant documents are retrieved and reranked, how does the Language Model generate an answer without "
            "hallucinating or falling into repetitive loops?",
            styles["body"],
        )
    )

    story.append(Paragraph("7.1 Autoregressive Generation & Sampling Parameters", styles["h2"]))
    story.append(
        Paragraph(
            "LLMs generate text token-by-token in an autoregressive loop. At step <i>t+1</i>, the model computes logits <i>z</i> "
            "over the vocabulary and samples a token according to a probability distribution modulated by decoding parameters:",
            styles["body"],
        )
    )

    sampling_params = [
        ("Temperature (T, default = 0.2 for RAG):", "Divides logits: z_i / T. As T → 0 (greedy decoding), the highest logit receives probability ≈ 1.0, producing factual, deterministic text. Higher T (> 0.8) flattens the distribution, increasing randomness."),
        ("Top-p (Nucleus) Sampling (default = 0.9):", "Sorts vocabulary by probability and truncates the tail, only sampling from the smallest set of tokens whose cumulative probability exceeds p. Eliminates bizarre, out-of-context tokens."),
        ("Repetition Penalty (θ, default = 1.15):", "Penalizes tokens that have already been generated in the current output: z_i' = z_i / θ if z_i > 0 else z_i · θ. Strongly suppresses degenerate repetition loops."),
    ]
    for sp, sdesc in sampling_params:
        story.append(Paragraph(f"• <b>{sp}</b> {sdesc}", styles["bullet"]))

    story.append(Paragraph("7.2 The 0.5B Mode Collapse Case Study", styles["h2"]))

    deg_box = (
        "<b>Case Study: The 30x Repeated Sentence Incident:</b><br/>"
        "In our testing, an ultra-compact 0.5B model (<code>qwen2.5:0.5b</code>) was tasked with generating procedural instructions. "
        "Because sub-1B models have limited multi-head attention capacity and no repetition penalty was initially configured, "
        "the model fell into an autoregressive trap, generating:<br/>"
        "<i>'Install the Laminate: Apply the laminate to the router, ensuring it is securely attached.'</i> <b>30 times in a row!</b><br/>"
        "Pillar 5 (our DeBERTa NLI cross-encoder) evaluated all 41 claims, detected that 39 of them were fabricated nonsense, "
        "and raised a <b>95% Hallucination Risk alert</b>.<br/>"
        "<b>The Production Solution:</b> Upgraded to <b>LLaMA-3.2 (3.2B)</b>, added <code>repeat_penalty=1.15</code>, "
        "and implemented an automatic algorithmic loop truncation guard in code."
    )
    story.append(make_callout("AUTOREGRESSIVE DEGENERATION CASE STUDY", deg_box, "warn", styles))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 8: ANSWER GROUNDING, HALLUCINATION SHIELDS & SENTENCE NLI
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 8: Answer Grounding, Hallucination Shields & Sentence NLI", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_LIGHT, spaceAfter=12))

    story.append(
        Paragraph(
            "How do we mathematically verify that an LLM's answer is factually grounded without relying on expensive "
            "human verification? We employ <b>Sentence-Level Natural Language Inference (NLI)</b>.",
            styles["body"],
        )
    )

    story.append(Paragraph("8.1 The NLI Formalism: Premise vs. Hypothesis", styles["h2"]))
    story.append(
        Paragraph(
            "In Natural Language Inference, a model determines the logical relationship between a <b>Premise (P)</b> "
            "and a <b>Hypothesis (H)</b>:",
            styles["body"],
        )
    )

    nli_classes = [
        ("🟢 Entailment (P entails H):", "If P is true, H MUST be true. Example: P = 'Drill with plastic bit at 4,500 rpm', H = 'Use plastic drill bits at high speed'. The claim is verified as factually grounded."),
        ("🔴 Contradiction (P contradicts H):", "If P is true, H CANNOT be true. Example: P = 'Do not use standard window cleaner', H = 'Clean panels using standard glass window cleaner'. The claim is flagged as an active hallucination / safety risk."),
        ("🟡 Neutral (P is neutral to H):", "P neither proves nor disproves H. Example: P = 'Panels are CARB2 certified', H = 'Panels are shipped in wooden crates'. The claim is ungrounded extrapolation."),
    ]
    for nc, nd in nli_classes:
        story.append(Paragraph(f"• <b>{nc}</b> {nd}", styles["bullet"]))

    story.append(Spacer(1, 10))
    story.append(create_nli_verdict_diagram())
    story.append(Spacer(1, 10))

    story.append(Paragraph("8.2 Deconstructing Answers into Verifiable Claims", styles["h2"]))
    story.append(
        Paragraph(
            "A generated paragraph often contains multiple independent propositions. In Pillar 5, our pipeline executes "
            "<b>Atomic Claim Decomposition</b> (in <code>src/grounding/claims.py</code>):<br/>"
            "1. <b>Protects Technical Entities:</b> Standard punctuation splitters break on 'DIN 68861', 'approx. 48h', or '2.0 mm'. "
            "We protect these abbreviations using regex token placeholders.<br/>"
            "2. <b>Discards Conversational Scaffolding:</b> Phrases like <i>'To answer your question:'</i> or <i>'Here are the steps:'</i> "
            "are conversational filler, not testable engineering claims. Filtering them prevents false contradiction flags.<br/>"
            "3. <b>Bullet-Level Passage Windowing:</b> Splits long passages by bullet characters (▪, •) to eliminate attention dilution, "
            "allowing the cross-encoder to evaluate specific recommendations with <b>98.3% entailment accuracy</b>.",
            styles["body"],
        )
    )

    story.append(Paragraph("8.3 Journal-Grade Grounding Metrics", styles["h2"]))

    metrics_table = [
        [
            Paragraph("<b>Metric</b>", styles["body"]),
            Paragraph("<b>Mathematical Formula</b>", styles["body"]),
            Paragraph("<b>Production Target</b>", styles["body"]),
            Paragraph("<b>Interpretation</b>", styles["body"]),
        ],
        [
            Paragraph("<b>Faithfulness Ratio</b>", styles["body"]),
            Paragraph("N_entailed / N_claims", styles["code"]),
            Paragraph("> 80.0%", styles["body"]),
            Paragraph("Percentage of answer claims directly supported by manual.", styles["body"]),
        ],
        [
            Paragraph("<b>Hallucination Risk</b>", styles["body"]),
            Paragraph("N_contradicted / N_claims", styles["code"]),
            Paragraph("< 5.0%", styles["body"]),
            Paragraph("Percentage of claims that contradict the engineering manual.", styles["body"]),
        ],
        [
            Paragraph("<b>Citation Precision</b>", styles["body"]),
            Paragraph("N_valid_citations / N_total_citations", styles["code"]),
            Paragraph("> 85.0%", styles["body"]),
            Paragraph("Did the cited chunk ID actually contain the supporting proof?", styles["body"]),
        ],
        [
            Paragraph("<b>TKG Provenance</b>", styles["body"]),
            Paragraph("N_aligned_entities / N_total_entities", styles["code"]),
            Paragraph("> 70.0%", styles["body"]),
            Paragraph("Percentage of answer entities linked to verified graph nodes.", styles["body"]),
        ],
    ]
    t_met = Table(metrics_table, colWidths=[115, 135, 105, 141])
    t_met.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#F1F5F9")),
                ("BOX", (0, 0), (-1, -1), 1, BORDER_LIGHT),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t_met)

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # CHAPTER 9: THE DATA SCIENTIST'S MASTERY BLUEPRINT
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Chapter 9: The Data Scientist's Mastery Blueprint", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BORDER_LIGHT, spaceAfter=12))

    story.append(
        Paragraph(
            "This chapter summarizes the master formulas, common debugging failure modes, and practical code recipes "
            "to master RAG and vector search in production.",
            styles["body"],
        )
    )

    story.append(Paragraph("9.1 Master Formula Cheat Sheet", styles["h2"]))

    formulas = [
        ("Cosine Similarity:", "cos(θ) = ( u · v ) / ( ||u||₂ ||v||₂ ) = Σ u_i v_i / ( √(Σ u_i²) √(Σ v_i²) )"),
        ("Okapi BM25 Relevance:", "BM25(D, Q) = Σ IDF(q) · [ f(q, D)(k₁+1) / ( f(q, D) + k₁(1 - b + b(|D|/avgdl)) ) ]"),
        ("Reciprocal Rank Fusion:", "RRF_Score(d) = Σ_{m}  [ w_m / ( k + rank_m(d) ) ]   (k = 60)"),
        ("Softmax Probability:", "P(y = i | z) = e^{z_i} / Σ_{j} e^{z_j}"),
        ("Shannon Entropy:", "H(P) = - Σ p_i · log₂(p_i)"),
        ("Faithfulness Ratio:", "Faithfulness = Number of Entailed Claims / Total Extracted Factual Claims"),
    ]
    for f_title, f_eq in formulas:
        story.append(Paragraph(f"• <b>{f_title}</b> <font color='#7C3AED'><code>{f_eq}</code></font>", styles["bullet"]))

    story.append(Paragraph("9.2 Production Debugging Checklist", styles["h2"]))

    checklist = [
        ("Symptom: Retrieval Recall@5 < 40%:", "Check token overlap and vocabulary mismatch. If colloquial terms are used, introduce a Terminology Knowledge Graph or Synonym Probe expansion."),
        ("Symptom: Hybrid Retrieval is worse than Dense-Only:", "Diagnose Fusion Dilution! BM25 is returning random noise on colloquial queries. Down-weight or gate BM25 when S_term is high."),
        ("Symptom: LLM loops or repeats sentences 10+ times:", "Autoregressive mode collapse in compact model. Set repeat_penalty = 1.15 in Ollama options, or upgrade model to 3B+ parameters."),
        ("Symptom: DeBERTa marks valid sentences as 'Contradiction':", "Check passage windowing! Large passages dilute attention. Split by bullet characters (▪, •) to test atomic recommendations."),
        ("Symptom: High latency (> 5 seconds per query):", "Avoid LLM query rewrites on every query. Implement an Adaptive Router fast-path to bypass LLM rewriting on routine queries."),
    ]
    for chk_title, chk_sol in checklist:
        story.append(Paragraph(f"• <b>{chk_title}</b><br/>{chk_sol}", styles["bullet"]))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 10))
    closing_box = (
        "<b>Congratulations!</b> You have mastered the mathematical, architectural, and evaluation foundations "
        "of enterprise RAG. By combining dense latent spaces with exact lexical indexes, cross-encoders, and "
        "sentence-level NLI attribution, you possess the end-to-end expertise to build safe, auditable, "
        "and zero-defect AI systems."
    )
    story.append(make_callout("THE ROAD AHEAD", closing_box, "insight", styles))

    return story


def main() -> None:
    """Compile the complete handbook PDF."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_out = os.path.join(base_dir, "Foundations_of_RAG_and_Vector_Search.pdf")

    print(f"Building Comprehensive RAG Textbook: {pdf_out}...")
    doc = SimpleDocTemplate(
        pdf_out,
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=48,
        bottomMargin=48,
    )

    styles = get_handbook_styles()
    story = build_handbook_story(styles)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"✓ Textbook generated successfully: {os.path.getsize(pdf_out):,} bytes")


if __name__ == "__main__":
    main()
