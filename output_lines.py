from doctr.io import DocumentFile
from doctr.models import ocr_predictor
# from doctr.utils.visualization import vis_and_synth
import os
import json

out_dir = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/"
in_dir = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Sample documents/"

file = "Academic transcipts/Certificado de estudios de maestría de la BUAP con apostilla.pdf"

model = ocr_predictor(pretrained=True)
model.to('cuda')

for root, dirs, files in os.walk(in_dir):
    for file in files:
        if file.endswith(".pdf"):
            print(os.path.join(root, file))
            doc = DocumentFile.from_pdf(os.path.join(root, file))
            result = model(doc)
            
            
            for i, res in enumerate(result.pages):
                out_name = os.path.join(root.replace(in_dir, out_dir), file.replace(".pdf", "_" + str(i+5) + ".png"))
                os.makedirs(os.path.dirname(out_name), exist_ok=True)
                # vis_and_synth(res.export(), res.page, fname=os.path.join(out_dir, out_name))
                dictWithInfo = res.export()
                
                # save the dict into a json file
                json_dir = out_name.replace(".png", ".json")
                with open(json_dir, 'w') as f:
                    json.dump(dictWithInfo, f)
                    
                # why "cannot close object"
                # 