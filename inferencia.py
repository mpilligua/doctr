from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(pretrained=True)
# PDF
doc = DocumentFile.from_pdf(r"C:\Users\Maria\OneDrive - UAB\Documentos\3r de IA\Synthesis project II\Sample documents\Certificates of no criminal records\Constancia de no antecedentes penales federales.pdf")
# Analyze
result = model(doc)