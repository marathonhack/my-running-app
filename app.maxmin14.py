import streamlit as st 
import pandas as pd
import datetime
import os
import cv2
import mediapipe as mp
import math

SAVE_DIR = "videos"
IDEAL_DIR = "ideal_videos"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(IDEAL_DIR, exist_ok=True)

# Streamlit ページ定義
st.title("ランニングフォーム分析アプリ")
page = st.sidebar.selectbox(
    "ページを選択",
    ["動画アップロード", "動画リスト表示", "結果表示", "フォーム比較ビュー", "レポート出力"]
)

def calculate_angle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) -
                       math.atan2(a[1] - b[1], a[0] - b[0]))
    return abs(ang + 360) if ang < 0 else ang

def analyze_video(video_path, output_csv="angle_result.csv"):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    frame_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            angle_left_knee = calculate_angle(l_hip, l_knee, l_ankle)

            frame_data.append({
                "フレーム": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                "左膝角度": angle_left_knee
            })

    cap.release()
    df = pd.DataFrame(frame_data)
    df.to_csv(output_csv, index=False)
    return df

def detect_phases(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)

    landing_frame = push_off_frame = None
    max_knee_angle = -1
    max_heel_y = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
            l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            l_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]

            knee_angle = calculate_angle(
                [l_hip.x, l_hip.y],
                [l_knee.x, l_knee.y],
                [l_ankle.x, l_ankle.y]
            )

            if knee_angle > max_knee_angle:
                max_knee_angle = knee_angle
                push_off_frame = frame_index

            if l_heel.y < 1.0:
                if max_heel_y == -1 or l_heel.y < max_heel_y:
                    max_heel_y = l_heel.y
                    landing_frame = frame_index

    cap.release()
    return landing_frame, push_off_frame

def draw_landmarks_and_angle(video_path, frame_num, output_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imwrite(output_path, frame)
    cap.release()

if page == "動画アップロード":
    st.header("① 動画アップロード")
    uploaded_file = st.file_uploader("ランニング動画をアップロードしてください", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        st.video(uploaded_file)
        save_path = os.path.join(SAVE_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"動画を保存しました: {save_path}")

elif page == "動画リスト表示":
    st.header("② 保存済み動画リスト")
    video_files = [f for f in os.listdir(SAVE_DIR) if f.endswith((".mp4", ".mov", ".avi"))]

    if not video_files:
        st.warning("保存済み動画がありません。")
    else:
        selected_video = st.selectbox("再生する動画を選択してください", video_files)
        video_path = os.path.join(SAVE_DIR, selected_video)
        st.video(video_path)

        if st.button("この動画を削除する"):
            try:
                os.remove(video_path)
                st.success(f"{selected_video} を削除しました。")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"削除に失敗しました: {e}")

elif page == "フォーム比較ビュー":
    st.header("フォーム比較ビュー（局面別）")
    user_videos = [f for f in os.listdir(SAVE_DIR) if f.endswith((".mp4", ".mov", ".avi"))]
    ideal_videos = [f for f in os.listdir(IDEAL_DIR) if f.endswith((".mp4", ".mov", ".avi"))]
    selected_user = st.selectbox("ユーザー動画を選択", user_videos)
    selected_ideal = st.selectbox("理想フォーム動画を選択", ideal_videos)
    user_path = os.path.join(SAVE_DIR, selected_user)
    ideal_path = os.path.join(IDEAL_DIR, selected_ideal)

    if st.button("着地と蹴り出しのフレームを抽出して比較"):
        with st.spinner("局面抽出中..."):
            frame_land_u, frame_push_u = detect_phases(user_path)
            frame_land_i, frame_push_i = detect_phases(ideal_path)
            dir_images = "key_frames"
            os.makedirs(dir_images, exist_ok=True)

            paths_landing = []
            paths_landing_ideal = []
            for offset in [-2, -1, 0, 1, 2]:
                path_user = os.path.join(dir_images, f"user_landing_{offset:+d}.jpg")
                path_ideal = os.path.join(dir_images, f"ideal_landing_{offset:+d}.jpg")
                draw_landmarks_and_angle(user_path, frame_land_u + offset, path_user)
                draw_landmarks_and_angle(ideal_path, frame_land_i + offset, path_ideal)
                paths_landing.append(path_user)
                paths_landing_ideal.append(path_ideal)

            path_push_u = os.path.join(dir_images, "user_push_off.jpg")
            path_push_i = os.path.join(dir_images, "ideal_push_off.jpg")
            draw_landmarks_and_angle(user_path, frame_push_u, path_push_u)
            draw_landmarks_and_angle(ideal_path, frame_push_i, path_push_i)

        st.subheader("着地局面比較")
        st.image(paths_landing, caption=["ユーザー: 着地-2", "ユーザー: 着地-1", "ユーザー: 着地", "ユーザー: 着地+1", "ユーザー: 着地+2"], width=300)
        st.image(paths_landing_ideal, caption=["理想: 着地-2", "理想: 着地-1", "理想: 着地", "理想: 着地+1", "理想: 着地+2"], width=300)

        st.subheader("蹴り出し局面比較")
        st.image([path_push_u, path_push_i], caption=["ユーザー: 蹴り出し", "理想: 蹴り出し"], width=300)

elif page == "レポート出力":
    st.header("⑤ レポート出力")
    st.write("※この機能は今後実装予定です。")
