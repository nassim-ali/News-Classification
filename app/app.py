from flask import Flask, render_template, request
from modules.processing import process_text, labels
from pyspark.ml.classification import LogisticRegressionModel

app = Flask(__name__)

# load the model
lrModel = LogisticRegressionModel.load("lrModel")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        news_text = request.form["news"]
        if news_text.strip():
            df = process_text(news_text)
            predictions = lrModel.transform(df)
            prediction = predictions.select("prediction").collect()[0]["prediction"]
            prediction = labels[prediction]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

