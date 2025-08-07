# app.py

from flask import Flask, request, jsonify
import os
from inference_system import analyze_image

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400
    f = request.files["file"]
    path = os.path.join("uploads", f.filename)
    os.makedirs("uploads", exist_ok=True)
    f.save(path)
    result = analyze_image(path)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
