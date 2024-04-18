import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np

classes = ["一般画像","ドキュメント画像"]
image_size = 50

UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.secret_key = "12345"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#学習済みモデルをロード
model = load_model('./model.keras', compile=False)

# request.method == 'POST'であるとき、これから後に続くコードが実行されます。
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み、np形式に変換
            # 今回モデルはグレースケール指定ではないため、color_mode="grayscale"無指定
            org_img = image.load_img(filepath, target_size=(image_size,image_size))
            img = image.img_to_array(org_img)
            data = np.array([img])
            #変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            if(result[0] < 0.5):
              pred_answer = "これは 一般画像 です"
            else:
              pred_answer = "これは ドキュメント画像 です"

            return render_template("index.html",answer=pred_answer)

    return render_template("index.html",answer="")


if __name__ == "__main__":
#    app.run()
    port = int(os.environ.get('PORT', 8080))
    app.run(host ='0.0.0.0',port = port)