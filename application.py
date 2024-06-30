import os
from datetime import datetime

import matplotlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template, jsonify


# Non-GUI 백엔드를 사용하도록 설정
matplotlib.use('Agg')

# Flask 애플리케이션 초기화
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'

# 데이터 경로 설정
org_path = os.path.join('static', 'ORG')
after_path = os.path.join('static', 'AFTER')
upload_folder = os.path.join('static', 'UPLOAD')
train_result_folder = os.path.join('static', 'TRAIN_RESULT')

# 업로드 폴더가 없으면 생성
for folder in [upload_folder, train_result_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 이미지 크기 설정
img_width, img_height = 256, 256


# 데이터 로드 함수
def load_images(folder_path, img_width, img_height):
    images = []
    files = os.listdir(folder_path)
    files.sort()

    for file in files:
        img = load_img(os.path.join(folder_path, file), target_size=(img_width, img_height))
        img_array = img_to_array(img).flatten()
        images.append(img_array)

    return np.array(images)
# 모델 학습 함수
def train_model():
    global scaler, kmeans, svm
    # 경로가 올바른지 확인
    if not os.path.exists(org_path):
        print(f"Original path does not exist: {org_path}")
    if not os.path.exists(after_path):
        print(f"Enhanced path does not exist: {after_path}")

    # 데이터 로드
    org_images = load_images(org_path, img_width, img_height)
    after_images = load_images(after_path, img_width, img_height)

    # 레이블 생성 (0: 원본, 1: 보정됨)
    labels = np.concatenate((np.zeros(len(org_images)), np.ones(len(after_images))))

    # 데이터 합치기
    images = np.concatenate((org_images, after_images))

    # 데이터 정규화
    scaler = StandardScaler()
    images_scaled = scaler.fit_transform(images)

    # 군집화 모델 학습
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(images_scaled)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(images_scaled, labels, test_size=0.2, random_state=42)

    # 분류 모델 학습 (SVM)
    svm = SVC(kernel='linear', random_state=42, probability=True)
    svm.fit(X_train, y_train)

    # 모델 예측
    y_pred = svm.predict(X_test)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # 학습 결과 저장
    save_training_result(X_train, y_train, X_test, y_test, y_pred, accuracy)

# 학습 결과 저장 함수
def save_training_result(X_train, y_train, X_test, y_test, y_pred, accuracy):
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    plt.figure(figsize=(12, 10))

    # Visualize training data
    for label, color in zip([0, 1], ['blue', 'red']):
        plt.scatter(X_train_pca[y_train == label, 0], X_train_pca[y_train == label, 1],
                    c=color, label=f"Train Data: {'Original' if label == 0 else 'Enhanced'}", alpha=0.5, marker='o')

    # Visualize test data (actual labels)
    for label, color in zip([0, 1], ['cyan', 'orange']):
        plt.scatter(X_test_pca[y_test == label, 0], X_test_pca[y_test == label, 1],
                    edgecolor='k', label=f"Test Data: {'Original' if label == 0 else 'Enhanced'}", alpha=0.5,
                    marker='s')

    # Visualize test data (predicted labels)
    for label, color in zip([0, 1], ['purple', 'yellow']):
        plt.scatter(X_test_pca[y_pred == label, 0], X_test_pca[y_pred == label, 1],
                    edgecolor='k', label=f"Predicted: {'Original' if label == 0 else 'Enhanced'}", alpha=0.5,
                    marker='x')

    plt.title("PCA of Images with SVM Classification", fontsize=15)
    plt.xlabel("Principal Component 1 (PC1)", fontsize=12)
    plt.ylabel("Principal Component 2 (PC2)", fontsize=12)
    plt.legend(loc='best')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{accuracy * 100:.2f}_percent_accuracy_{timestamp}.png"
    filepath = os.path.join(train_result_folder, filename)
    plt.savefig(filepath)
    plt.close()


# 초기 모델 학습
train_model()

@app.route('/')
def index():
    return render_template('main.html')
def classify_image(img_path, img_width, img_height, scaler, kmeans, svm):
    img = load_img(img_path, target_size=(img_width, img_height))
    img_array = img_to_array(img).flatten().reshape(1, -1)
    img_scaled = scaler.transform(img_array)

    cluster_label = kmeans.predict(img_scaled)
    classification = svm.predict(img_scaled)
    classification_prob = svm.decision_function(img_scaled)

    prob_original = prob_enhanced = 0.0

    if classification[0] == 0:
        result = "원본"
        prob_original = classification_prob[0]
        prob_enhanced = 1 - prob_original
    else:
        result = "보정"
        prob_enhanced = classification_prob[0]
        prob_original = 1 - prob_enhanced

    prob_original = max(0.0, min(prob_original, 1.0))
    prob_enhanced = max(0.0, min(prob_enhanced, 1.0))

    return result, prob_original, prob_enhanced

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filepath = os.path.join(upload_folder, file.filename)
        file.save(filepath)
        result, prob_original, prob_enhanced = classify_image(filepath, img_width, img_height, scaler, kmeans, svm)
        return jsonify({
            'file_path': filepath,
            'result': result,
            'prob_original': float(prob_original) * 100,
            'prob_enhanced': float(prob_enhanced) * 100
        })

@app.route('/retrain', methods=['POST'])
def retrain():
    train_model()
    return jsonify({'message': '모델이 성공적으로 다시 학습되었습니다.'})

if __name__ == '__main__':
    app.run( debug=True)
