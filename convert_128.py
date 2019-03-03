from PIL import Image
import os.path
import glob
from tqdm import tqdm
def convertjpg(jpgfile,outdir,width=128,height=128):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)   
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)
for jpgfile in tqdm(glob.glob("beauty_image/beauty_image/*.jpg")):
    convertjpg(jpgfile,"beauty_image_128_bk/")

