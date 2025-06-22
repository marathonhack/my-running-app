import streamlit as st
import cv2
import os
from PIL import Image
import shutil

# 保存フォルダの準備
FRAME_DIR = "extracted_frames"
os.makedirs(FRAME_DIR, exist_ok=True)

st.title("ランニングフォーム手動比較アプリ（MediaPipeなし）")

# 動画アップロード
video_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "mov", "avi"])

if video_file:
    # 保存先の動画パス
    video_path = os.path.join("temp_video.mp4")
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.video(video_file)

    # フレーム抽出（10フレームおき）
    st.info("動画からフレーム画像を抽出します...")
    cap = cv2.VideoCapture(video_path)
    frame_rate = 10
    frame_count = 0
    saved_count = 0

    shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(FRAME_DIR, f"frame_{saved_count:03d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_count += 1
    cap.release()
    st.success(f"{saved_count}枚のフレーム画像を保存しました。")

    # フレーム画像一覧表示と選択
    image_files = sorted(os.listdir(FRAME_DIR))
    selected_frame1 = st.selectbox("比較したい画像1（例：着地）を選んでください", image_files)
    selected_frame2 = st.selectbox("比較したい画像2（例：蹴り出し）を選んでください", image_files)

    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(FRAME_DIR, selected_frame1), caption="着地（例）", use_column_width=True)
    with col2:
        st.image(os.path.join(FRAME_DIR, selected_frame2), caption="蹴り出し（例）", use_column_width=True)

    # 理想フォームとの比較（任意）
    st.markdown("---")
    st.subheader("理想フォームとの比較（任意）")
    ideal_image = st.file_uploader("理想フォーム画像をアップロード", type=["jpg", "png"])
    if ideal_image:
        st.image(ideal_image, caption="理想フォーム", use_column_width=True)
