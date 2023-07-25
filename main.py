from flask import Flask, request, jsonify
from deepface import DeepFace

app = Flask(__name__)

def compare_images(image1_path, image2_path):
    img1 = DeepFace.detectFace(image1_path, detector_backend='opencv')
    img2 = DeepFace.detectFace(image2_path, detector_backend='opencv')
    result = DeepFace.verify(img1, img2, enforce_detection=False)
    distance = result["distance"]
    similarity_score = 1 - distance
    return similarity_score

@app.route('/compare', methods=['POST'])
def compare():
    data = request.get_json()
    test_image = data.get('test_image', None)
    reference_image = data.get('reference_image', None)

    if not test_image or not reference_image:
        return jsonify({'error': 'Please provide both test_image and reference_image.'}), 400

    similarity_score = compare_images(test_image, reference_image)

    if similarity_score >= 0.66:
        output = "Similarity"
    else:
        output = "Dissimilarity Score"

    return jsonify({output: similarity_score}), 200

if __name__ == '__main__':
    app.run(debug=True)
