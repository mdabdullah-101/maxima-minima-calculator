"""
utils/pdf_generator.py
======================
Generates a downloadable PDF report containing the problem statement,
step-by-step solution, result table, and graph.
"""

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def clean_text_for_pdf(text: str) -> str:
    """Removes LaTeX inline delimiters so text renders cleanly in PDF."""
    replacements = [
        ("$$", ""), ("$", ""), ("\\therefore", "Therefore,"), 
        ("\\implies", "=>"), ("\\frac", ""), ("\\left", ""), 
        ("\\right", ""), ("\\Rightarrow", "=>"), ("\\text", ""), 
        ("{", ""), ("}", ""), ("\\", "")
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def generate_pdf_report(title: str, steps: list, fig, df_results) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('PDFTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1E293B'), spaceAfter=12)
    heading_style = ParagraphStyle('PDFHeading', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2563EB'), spaceBefore=10, spaceAfter=6)
    body_style = ParagraphStyle('PDFBody', parent=styles['BodyText'], fontSize=10, leading=14, textColor=colors.HexColor('#334155'), spaceAfter=6)

    elements = []

    elements.append(Paragraph(f"<b>Maxima & Minima Report: {title}</b>", title_style))
    elements.append(Spacer(1, 10))

    for step in steps:
        elements.append(Paragraph(f"<b>{step['title']}</b>", heading_style))
        clean_content = clean_text_for_pdf(step['content']).replace('\n', '<br/>')
        elements.append(Paragraph(clean_content, body_style))
        elements.append(Spacer(1, 6))

    if df_results is not None and not df_results.empty:
        elements.append(Paragraph("<b>Summary Results</b>", heading_style))
        table_data = [list(df_results.columns)] + df_results.astype(str).values.tolist()
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 12))

    if fig is not None:
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        img_buffer.seek(0)
        elements.append(Paragraph("<b>Graphical Solution</b>", heading_style))
        elements.append(Image(img_buffer, width=450, height=280))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()