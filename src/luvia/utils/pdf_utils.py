import cv2
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import uuid
import os
import string

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, PageTemplate, BaseDocTemplate, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, ListFlowable, ListItem
from reportlab.platypus import Image, PageBreak, Table, TableStyle, Paragraph
from reportlab.platypus import KeepTogether

import os

from reportlab.lib.utils import ImageReader



class FormalReport:
    def __init__(self, filename="formal_report.pdf", location="Unknown", agent="Unknown"):
        self.filename = filename
        self.PAGE_WIDTH, self.PAGE_HEIGHT = A4
        self.MARGIN = 20 * mm
        self.styles = self._init_styles()
        self.doc = BaseDocTemplate(
            self.filename,
            pagesize=A4,
            leftMargin=self.MARGIN,
            rightMargin=self.MARGIN,
            topMargin=45 * mm,
            bottomMargin=60 * mm
        )
        self.frame = Frame(self.doc.leftMargin, self.doc.bottomMargin, self.doc.width, self.doc.height, id='normal')
        self.template = PageTemplate(id='formal', frames=self.frame, onPage=self.draw_header, onPageEnd=self.draw_footer)
        self.doc.addPageTemplates([self.template])
        self.story = []
        self.footer_logo_path = "{}/gifs/signal-2025-08-23-160817_002.png".format(os.path.dirname(os.path.abspath(__file__)))
        self.logo_path = "{}/gifs/signal-2025-08-25-003555_002.png".format(os.path.dirname(os.path.abspath(__file__)))
        self.location = location
        self.agent = agent.split("_")[0]
        self.styles_list= getSampleStyleSheet()

    def _init_styles(self):
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name='FormalBodyText', fontName='Times-Roman', fontSize=10, leading=8))
        styles.add(ParagraphStyle(name='SectionTitle', fontName='Times-Bold', fontSize=20, leading=24, spaceAfter=4))
        styles.add(ParagraphStyle(name='SubSectionTitle', fontName='Times-Bold', fontSize=16, spaceAfter=6))
        styles.add(ParagraphStyle(name='SubSubSectionTitle', fontName='Times-Bold', fontSize=12, spaceAfter=6))
        styles.add(ParagraphStyle(name='SubSectionText', fontName='Times-Roman', fontSize=12, leading=10, spaceAfter=6))
        styles.add(ParagraphStyle(name='BulletText', parent=styles['Normal'],fontSize=10, leading=10, spaceAfter=2))
        styles.add(ParagraphStyle(name='WrappedText',fontName='Times-Roman',fontSize=9,leading=10,spaceAfter=4))
        return styles


    def draw_header(self, canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, self.PAGE_WIDTH, self.PAGE_HEIGHT, stroke=0, fill=1)

        # Logo (left side)
        canvas.drawImage(self.logo_path, x=self.MARGIN, y=self.PAGE_HEIGHT - 40 * mm, width=25 * mm, height=25 * mm, mask='auto')

        # Title block (right-aligned)
        canvas.setFont("Times-Bold", 14)
        canvas.setFillColor(colors.black)
        canvas.drawRightString(self.PAGE_WIDTH - self.MARGIN, self.PAGE_HEIGHT - 20 * mm, "LUVIA Technical Report")

        canvas.setFont("Times-Roman", 11)
        canvas.drawRightString(self.PAGE_WIDTH - self.MARGIN, self.PAGE_HEIGHT - 26 * mm, "Agent: {}".format(self.agent))
        canvas.drawRightString(self.PAGE_WIDTH - self.MARGIN, self.PAGE_HEIGHT - 32 * mm, "Location: {}".format(self.location))

        # Divider lines (moved up slightly)
        canvas.setLineWidth(4)
        canvas.setStrokeColor(colors.HexColor("#162F48"))  # Deep Blue
        canvas.line(self.MARGIN, self.PAGE_HEIGHT - 38 * mm, self.PAGE_WIDTH - self.MARGIN, self.PAGE_HEIGHT - 38 * mm)
        canvas.setStrokeColor(colors.HexColor("#DCB68A"))  # Golden
        canvas.line(self.MARGIN, self.PAGE_HEIGHT - 40 * mm, self.PAGE_WIDTH - self.MARGIN, self.PAGE_HEIGHT - 40 * mm)

        canvas.restoreState()



    def draw_footer(self, canvas, doc):
        canvas.saveState()
        footer_top = 50 * mm

        # Draw golden and red lines
        canvas.setLineWidth(4)
        canvas.setStrokeColor(colors.HexColor("#DCB68A"))  # Golden
        canvas.line(self.MARGIN, footer_top, self.PAGE_WIDTH - self.MARGIN, footer_top)
        canvas.setStrokeColor(colors.HexColor("#B8374A"))  # Red
        canvas.line(self.MARGIN, footer_top - 2 * mm, self.PAGE_WIDTH - self.MARGIN, footer_top - 2 * mm)

        # Draw logo on the right side without stretching
        logo_width = 45 * mm
        logo_height = 45 * mm
        logo_x = self.PAGE_WIDTH - self.MARGIN - logo_width
        logo_y = self.MARGIN
        canvas.drawImage(
            self.footer_logo_path,
            x=logo_x,
            y=logo_y,
            width=logo_width,
            height=logo_height,
            preserveAspectRatio=True,
            anchor='sw',
            mask='auto'
        )

        # Draw centered page number
        canvas.setFont("Times-Italic", 8)
        canvas.setFillColor(colors.HexColor("#DCB68A"))  # Golden
        canvas.drawCentredString(self.PAGE_WIDTH / 2, self.MARGIN + 5 * mm, f"Page {doc.page}")

        canvas.restoreState()


    def add_cover_page(self, project_name="[Project Name]", author="[Author Name]", date="[DD/MM/YYYY]"):
        self.story.append(Spacer(1, 100))
        self.story.append(Paragraph(f"<b>Project Name:</b> {project_name}", self.styles['FormalBodyText']))
        self.story.append(Spacer(1, 10))
        self.story.append(Paragraph(f"<b>Author:</b> {author}", self.styles['FormalBodyText']))
        self.story.append(Spacer(1, 10))
        self.story.append(Paragraph(f"<b>Date:</b> {date}", self.styles['FormalBodyText']))
        self.story.append(PageBreak())

    def add_section(self, title, content):
        self.story.append(Paragraph(title, self.styles['SectionTitle']))
        self.story.append(Paragraph("&nbsp;{}".format(content), self.styles['FormalBodyText']))
        self.story.append(Spacer(1, 20))

    def build(self):
        self.doc.build(self.story)

    def add_section_with_image(self, title, text, image_path, width=100, height=75):
        self.story.append(Paragraph(title, self.styles['SectionTitle']))
        self.story.append(Paragraph("&nbsp;{}".format(text), self.styles['FormalBodyText']))
        if os.path.exists(image_path):
            self.story.append(Spacer(1, 12))  # Add space before image
            self.story.append(Image(image_path, width=(self.PAGE_WIDTH/5)*4, height=height*4))
            self.story.append(Spacer(1, 12))  # Add space before image
        self.story.append(Spacer(1, 20))

    # Recursive function to build nested bullet list
    def build_nested_list(self, data, level=0):
        flowables = []
        for item in data:
            if isinstance(item, dict):
                text = item.get("text", "")
                children = item.get("children", [])
            else:
                text = str(item)
                children = []

            para = Paragraph(f'• {text}', self.get_level_style(level))
            flowables.append(para)

            if children:
                flowables.extend(self.build_nested_list(children, level + 1))
        return flowables


    def add_subsection_with_image(self, title, location, proposed_sentences, image_path, width=120, height=60):

        self.story.append(KeepTogether(Paragraph("<u>{}</u>".format(title), self.styles["SubSectionTitle"])))

        PAGE_WIDTH, PAGE_HEIGHT = A4
        MAX_WIDTH = (2*PAGE_WIDTH) / 3  # Half the page width

        if os.path.exists(image_path):
            img_reader = ImageReader(image_path)
            iw, ih = img_reader.getSize()
            aspect = ih / float(iw)

            # Scale width to half the page, adjust height to maintain aspect ratio
            scaled_width = MAX_WIDTH
            scaled_height = scaled_width * aspect

            # Optional: limit height if needed
            if scaled_height > PAGE_HEIGHT * 0.8:
                scaled_height = PAGE_HEIGHT * 0.8
                scaled_width = scaled_height / aspect

            img = Image(image_path, width=scaled_width, height=scaled_height)
            self.story.append(img)
        square = "*"
        possible_sentences = []
        for m, prop in enumerate(proposed_sentences):
            possible_sentences.append(prop["sentence"])
        possible_sentences = ", ".join(possible_sentences)
        self.story.append(KeepTogether(Paragraph(f"- <b>Location:</b> {location}", self.styles["BulletText"])))
        self.story.append(KeepTogether(Paragraph(f"- <b>Proposed sentences:</b> {possible_sentences}", self.styles["BulletText"])))

        alphabet_list = list(string.ascii_lowercase)
        for idx, prop in enumerate(proposed_sentences):
            self.story.append(Spacer(1, 15))
            if idx != 0:
                text = "\u27A4  <font size=14><b>{}</b></font>".format(prop["sentence"])#,prop["sentence"])
            else:
                text = "\u27A4  <font size=14><b>{}</b></font> <i>(Most probable translation)</i>".format(prop["sentence"])#,prop["sentence"])
                
            self.story.append(KeepTogether(Paragraph(text, self.styles['SubSubSectionTitle'])))
            self.story.append(Spacer(1, 4))
            text2 = '&nbsp;&nbsp;&nbsp;-Text: <b>"{}"</b>'.format(prop["sentence"])
            self.story.append(KeepTogether(Paragraph(text2, self.styles['SubSectionText'])))
            self.story.append(Spacer(1, 2))
            text2 = '&nbsp;&nbsp;&nbsp;-Probability: {}%'.format(prop["probability"])
            self.story.append(KeepTogether(Paragraph(text2, self.styles['SubSectionText'])))
            self.story.append(Spacer(1, 2))
            #text2 = '&nbsp;&nbsp;&nbsp;-Syntax:<b>"{}"</b>'.format(prop["syntax"])
            #self.story.append(Paragraph(text2, self.styles['SubSectionText']))
            #self.story.append(Spacer(1, 2))
            text2 = '&nbsp;&nbsp;&nbsp;-Translations:'
            self.story.append(KeepTogether(Paragraph(text2, self.styles['SubSectionText'])))
            texttrans = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· <i>english</i>: {}'.format(prop["translations"]["english"])
            self.story.append(KeepTogether(Paragraph(texttrans, self.styles['BulletText'])))
            texttrans = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· <i>german</i>: {}'.format(prop["translations"]["german"])
            self.story.append(KeepTogether(Paragraph(texttrans, self.styles['BulletText'])))
            texttrans = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· <i>danish</i>: {}'.format(prop["translations"]["danish"])
            self.story.append(KeepTogether(Paragraph(texttrans, self.styles['BulletText'])))
            texttrans = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· <i>arabic</i>: {}'.format(prop["translations"]["ar"])
            self.story.append(KeepTogether(Paragraph(texttrans, self.styles['BulletText'])))
            texttrans = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· <i>latin</i>: {}'.format(prop["translations"]["la"])
            self.story.append(KeepTogether(Paragraph(texttrans, self.styles['BulletText'])))
            texttrans = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;· <i>spanish</i>: {}'.format(prop["translations"]["spanish"])
            self.story.append(KeepTogether(Paragraph(texttrans, self.styles['BulletText'])))
            self.story.append(Spacer(1, 2))
            text3 = '&nbsp;&nbsp;&nbsp;-Word analysis:'
            self.story.append(KeepTogether(Paragraph(text3, self.styles['SubSectionText'])))

            # Prepare the data with Paragraphs
            word_data = [["Word", "Definition", "Synonyms", "Antonyms", "Etymology"]]
            for word in prop["word_analysis"]:
                word_data.append([
                    Paragraph(word, self.styles["WrappedText"]),
                    Paragraph(prop["word_analysis"][word]["definition"],  self.styles["WrappedText"]),
                    Paragraph(", ".join(prop["word_analysis"][word]["synonyms"]),  self.styles["WrappedText"]),
                    Paragraph(", ".join(prop["word_analysis"][word]["antonyms"]),  self.styles["WrappedText"]),
                    Paragraph(", ".join(prop["word_analysis"][word]["etymology"]),  self.styles["WrappedText"])
                ])

            # Create the table
            word_table = Table(word_data, colWidths=[60, 180, 100, 100])  # Adjust widths as needed

            # Apply style
            word_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Bold header
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))

            # Add to story
            self.story.append(Spacer(1, 12))
            self.story.append(KeepTogether(word_table))

        self.story.append(Spacer(1, 15))

    def add_section_with_imageNN(self, title, text, images_dict, width=80, height=60):
        pass

