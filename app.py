from flask import Flask,render_template,request
import os
from predict import predict_image

app=Flask(__name__)

UPLOAD_FOLDER="static/uploads"
os.makedirs(UPLOAD_FOLDER,exist_ok=True)
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    file=request.files['image']
    path=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
    file.save(path)

    label,conf,sev,ndvi,ndbi,ndwi=predict_image(path)

    return render_template("result.html",
                           label=label,
                           confidence=round(conf*100,2),
                           severity=sev,
                           ndvi=round(ndvi,3),
                           ndbi=round(ndbi,3),
                           ndwi=round(ndwi,3),
                           image_path=path)

if __name__=="__main__":
    app.run(debug=True)