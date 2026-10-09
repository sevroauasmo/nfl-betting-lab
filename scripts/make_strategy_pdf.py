"""One-page-ish shareable PDF of the strategy (README content above the Caveats section)."""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/surprise_inactive_unders.pdf"
F = "/System/Library/Fonts/Supplemental/"
pdfmetrics.registerFont(TTFont("Arial", F + "Arial.ttf"))            # has ≤ ≥ − ¢ glyphs
pdfmetrics.registerFont(TTFont("Arial-Bold", F + "Arial Bold.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold")

INK, MUTED, ACCENT, RULE, FILL = colors.HexColor("#17211B"), colors.HexColor("#5B6860"), colors.HexColor("#1E6B45"), colors.HexColor("#D5DCD5"), colors.HexColor("#EEF3EE")
H1 = ParagraphStyle("h1", fontName="Arial-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName="Arial-Bold", fontSize=13.5, leading=17, textColor=ACCENT, spaceBefore=9, spaceAfter=4)
H3 = ParagraphStyle("h3", fontName="Arial-Bold", fontSize=11, leading=14, textColor=INK, spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("body", fontName="Arial", fontSize=10, leading=14, textColor=INK, alignment=TA_LEFT, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.8, leading=12, textColor=MUTED)
CELL = ParagraphStyle("cell", fontName="Arial", fontSize=9, leading=11.5, textColor=INK)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Arial-Bold")


def table(rows, widths, header=True):
    data = [[Paragraph(c, CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    st = [("GRID", (0, 0), (-1, -1), 0.5, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), FILL))
    t.setStyle(TableStyle(st))
    return t


def bullets(items, style=BODY):
    return ListFlowable([ListItem(Paragraph(i, style), leftIndent=12) for i in items], bulletType="bullet", start="•",
                        leftIndent=12, bulletFontName="Arial", bulletFontSize=9)


s = []
s.append(Paragraph("Surprise-inactive receiver unders", H1))
s.append(Paragraph("One NFL betting edge that survived a season-length search through game lines, totals, quarter and half "
                   "markets, player-prop models and prediction-market liquidity.", BODY))

s.append(Paragraph("The rule", H2))
s.append(Paragraph("About <b>90 minutes before kickoff</b>, NFL teams publish their inactive lists.", BODY))
s.append(ListFlowable([
    ListItem(Paragraph("<b>Trigger:</b> a WR, TE or RB is <b>inactive</b> but was <b>not listed Out or Doubtful</b> on that "
                       "week's final injury report, and he had <b>≥ 10% of his team's targets</b> over his previous 3 games.", BODY)),
    ListItem(Paragraph("<b>Bet:</b> the <b>under</b> on each remaining teammate's <b>receptions</b> and <b>receiving yards</b>.", BODY)),
    ListItem([Paragraph("<b>Where:</b>", BODY),
              bullets(["<b>Novig</b> first. Take the listed under at the consensus line.",
                       "<b>Kalshi</b> if Novig doesn't list the prop. Buy NO on the ladder contract that matches the line "
                       "(\"5+ receptions\" for under 4.5), <b>only when the spread is ≤ 4¢</b>. Otherwise rest a NO bid at the "
                       "mid until kickoff. Never take a wide Kalshi book: the spread eats the whole edge."])]),
    ListItem(Paragraph("<b>Stake:</b> flat and small. About 1–2 triggers a week, 15–20 bets.", BODY)),
], bulletType="1", leftIndent=14, bulletFontName="Arial-Bold", bulletFontSize=10))

s.append(Paragraph("Why it works", H2))
s.append(Paragraph("When a role player is a surprise scratch, books and bettors push his teammates' receiving lines up to "
                   "absorb the freed targets, and they push them too far. The freed volume spreads across more players and a "
                   "less efficient offense than the lines assume. The market prices absences it has known about for days "
                   "correctly (no edge there). The overreaction is specific to <b>late, surprise</b> scratches, and it grows "
                   "with the size of the vacated role.", BODY))

s.append(Paragraph("Evidence", H2))
s.append(Paragraph("Backtest, 2023–2026", H3))
s.append(Paragraph("All triggers use only information available before kickoff: the scratched player's usage from earlier "
                   "games, Friday's report and game-day inactives. Graded at the closing line, with standard errors "
                   "clustered by team-game.", SMALL))
W = 7.0 * inch
s.append(table([
    ["", "Result"],
    ["Qualifying team-games", "217"],
    ["Teammates' receiving overs vs. de-vigged fair price", "43.9% vs 49.7%"],
    ["Edge beyond the normal receiving under lean", "−4.0 pts, t = −2.6"],
    ["ROI at the best sportsbook price", "<b>+5.4% to +7.5%</b> (95% CI on the broad prop set: +0.2% to +10.6%)"],
    ["ROI at Novig/ProphetX prices", "<b>+10.2%</b> (584 bets)"],
    ["ROI at Novig only, 2025 (Novig listed 87.5% of qualifying props)", "<b>+11.5%</b> (70 team-games). Novig baseline for all receiving unders: +0.9%"],
    ["By season, best book", "2023 +0.7%, 2024 +8.5%, 2025 +13.1%"],
    ["Dose-response (vacated target share 10–18% / 18–30% / 30%+)", "−3.3 / −9.6 / −7.9 pts vs. fair"],
    ["Kalshi (2025–26, priced 1 min after inactives)", "Taking any ask −1.1%. Spread ≤ 4¢ <b>+13.1%</b> (105 contracts). At the mid +11.6%"],
    ["Probability the true ROI > 0 (bootstrap)", "~98%"],
], [W * 0.46, W * 0.54]))

s.append(Paragraph("Live paper test, 2026 (weeks 2–4)", H3))
s.append(table([
    ["Venue", "Bets", "Win rate", "ROI"],
    ["Novig", "41", "56%", "+10.2%"],
    ["Best exchange (Novig/ProphetX)", "65", "62%", "+16.4%"],
    ["Best sportsbook (reference only)", "71", "59%", "+10.5%"],
    ["Kalshi, spread ≤ 4¢", "64", "66%", "+19.0%"],
    ["Kalshi, resting at the mid", "37 filled of 42", "62%", "+20.5%"],
], [W * 0.46, W * 0.2, W * 0.14, W * 0.2]))
s.append(Spacer(1, 4))
s.append(Paragraph("Six triggered team-games (Puka Nacua, RJ Harvey, Adonai Mitchell, Jalen Coker, Keenan Allen, Terry "
                   "McLaurin). Weeks 2–3 overlap the backtest window, so only week 4 is fully out of sample.", SMALL))

s.append(Paragraph("Projection", H3))
s.append(Paragraph("Backtest winners shrink going forward, so plan on <b>about +5% on Novig</b> (90% range roughly −2% to +12%). "
                   "About 50 more triggered team-games (roughly the rest of this season) confirms or rules out +5% at 2 "
                   "standard errors.", BODY))

doc = SimpleDocTemplate(str(OUT), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                        topMargin=0.55 * inch, bottomMargin=0.55 * inch, title="Surprise-inactive receiver unders")
doc.build(s)
print(OUT)
