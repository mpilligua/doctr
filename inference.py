from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.utils.visualization import vis_and_synth
import os

out_dir = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Datasets/DocTR_output/"
in_dir = "/hhome/ps2g07/document_analysis/github/Project_Synthesis2-/Sample documents/"

file = "Academic transcipts/Certificado de estudios de maestría de la BUAP con apostilla.pdf"

model = ocr_predictor(pretrained=True)
model.to('cuda')

for root, dirs, files in os.walk(in_dir):
    for file in files:
        if file.endswith(".pdf"):
            doc = DocumentFile.from_pdf(os.path.join(root, file))
            result = model(doc)

            for i, res in enumerate(result.pages):
                out_name = os.path.join(root.replace(in_dir, out_dir), file.replace(".pdf", "_" + str(i) + ".png"))
                os.makedirs(os.path.dirname(out_name), exist_ok=True)
                vis_and_synth(res.export(), res.page, fname=os.path.join(out_dir, out_name))









# visualize_page(self.export(), self.page, interactive=interactive, preserve_aspect_ratio=preserve_aspect_ratio)
# plt.savefig(**kwargs)
# result.show(fname="output.png")
# res = result.synthesize()

# # res is a np array with the image of the document
# import matplotlib.pyplot as plt
# import time 
# import numpy as np

# for r,info in res: 
#     plt.imsave(f"synth{time.time()}.png", r.astype(np.uint8))
#     print(info)

# result = str(result)


# import pickle
# with open("/hhome/ps2g07/document_analysis/OCR/DocTR/exemples/1.pkl", "wb") as f:
#     pickle.dump(result, f)