import os
from fpdf import FPDF
from fpdf.enums import XPos, YPos

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=10)
pdf.set_margins(8, 8, 8)

folder = r"D:\DIP proj"

for filename in sorted(os.listdir(folder)):
    if filename.endswith(".py"):
        filepath = os.path.join(folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        pdf.add_page()
        pdf.set_font("Courier", size=10)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, f"File: {filename}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        pdf.ln(2)
        pdf.set_font("Courier", size=7.5)

        for line in code.splitlines():
            # Replace tabs and strip non-latin characters that fpdf can't render
            line = line.replace("\t", "    ")
            line = line.encode("latin-1", errors="replace").decode("latin-1")
            # Break very long lines into chunks of 100 characters
            chunks = [line[i:i+100] for i in range(0, max(len(line), 1), 100)]
            for chunk in chunks:
                pdf.cell(0, 4, chunk, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

pdf.output("project_code.pdf")
print("Done! project_code.pdf created.")