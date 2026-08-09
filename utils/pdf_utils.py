"""
pdf_utils.py
-------------
Generación de un reporte PDF descargable con el diagnóstico de la hoja de
café y las recomendaciones técnicas generadas por la API de Groq.
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

DARK_GREEN = colors.HexColor("#2F4A34")
CREAM = colors.HexColor("#F5F1E8")
ACCENT = colors.HexColor("#B5651D")


def build_pdf_report(image_bytes: bytes, display_name: str, scientific_name: str,
                      confidence: float, recommendation: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleCustom", parent=styles["Heading1"], textColor=DARK_GREEN, fontSize=20)
    h2_style = ParagraphStyle("H2Custom", parent=styles["Heading2"], textColor=DARK_GREEN, fontSize=13, spaceBefore=10)
    body_style = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontSize=10.5, leading=15)
    small_style = ParagraphStyle("SmallCustom", parent=styles["BodyText"], fontSize=8.5, textColor=colors.grey)

    story = []
    story.append(Paragraph("AgroDetect / Soporte HICAFE", small_style))
    story.append(Paragraph("Reporte de Diagnóstico Foliar - Café", title_style))
    story.append(Paragraph(datetime.now().strftime("Generado el %d/%m/%Y a las %H:%M"), small_style))
    story.append(Spacer(1, 10))

    if image_bytes:
        try:
            img = RLImage(io.BytesIO(image_bytes), width=6 * cm, height=6 * cm)
            story.append(img)
            story.append(Spacer(1, 8))
        except Exception:
            pass

    diag_table = Table(
        [[Paragraph(f"<b>{display_name}</b><br/><i>{scientific_name}</i>", body_style),
          Paragraph(f"<b>{confidence*100:.1f}%</b><br/>confianza", body_style)]],
        colWidths=[11 * cm, 4 * cm],
    )
    diag_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CREAM),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8D0BE")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Descripción", h2_style))
    story.append(Paragraph(recommendation.get("descripcion", "-"), body_style))

    story.append(Paragraph("Diferenciación a simple vista", h2_style))
    story.append(Paragraph(recommendation.get("diferenciacion", "-"), body_style))

    def bullet_list(items):
        return ListFlowable(
            [ListItem(Paragraph(item, body_style), bulletColor=ACCENT) for item in items],
            bulletType="bullet", start="circle", leftIndent=14,
        )

    story.append(Paragraph("Manejo agronómico preventivo y correctivo", h2_style))
    story.append(bullet_list(recommendation.get("manejo_preventivo", [])))

    story.append(Paragraph("Buenas prácticas", h2_style))
    story.append(bullet_list(recommendation.get("buenas_practicas", [])))

    story.append(Paragraph("Monitoreo y seguimiento", h2_style))
    story.append(bullet_list(recommendation.get("monitoreo_seguimiento", [])))

    story.append(Paragraph("Consulta a un técnico", h2_style))
    story.append(Paragraph(recommendation.get("cuando_consultar_tecnico", "-"), body_style))

    story.append(Paragraph("Registro y trazabilidad", h2_style))
    story.append(Paragraph(recommendation.get("registro_trazabilidad", "-"), body_style))

    story.append(Spacer(1, 16))
    story.append(Paragraph(
        "Este reporte fue generado automáticamente por un sistema de inteligencia artificial "
        "(visión por computadora + LLM) y tiene fines de apoyo técnico. No sustituye la "
        "evaluación de un ingeniero agrónomo en campo.",
        small_style,
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
