import os
import subprocess
import tempfile
import signal
import streamlit as st
import importlib.util

st.set_page_config(
    page_title="Model Export & Test",
    page_icon=":wrench:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check required modules
required_modules = ['ultralytics', 'cv2', 'win32gui', 'win32con']
missing_modules = [mod for mod in required_modules if importlib.util.find_spec(mod) is None]

if missing_modules:
    with st.spinner(f"Installing missing modules: {', '.join(missing_modules)}"):
        for mod in missing_modules:
            subprocess.run(["pip", "install", mod], capture_output=True)
    st.rerun()

import cv2
import win32con, win32gui

# Patch TensorRT __version__ attribute for compatibility with Ultralytics
try:
    import tensorrt
    if not hasattr(tensorrt, '__version__'):
        try:
            from importlib.metadata import version
            tensorrt.__version__ = version('tensorrt')
        except Exception:
            tensorrt.__version__ = '10.0.0'
except ImportError:
    pass

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "EXPORT"

with st.sidebar:
    tabs = ["EXPORT", "TESTS"]
    st.session_state.current_tab = st.radio(label="**Select tab**", options=tabs, horizontal=False, label_visibility="visible")

    if st.button(label="Exit", key="sidebar_exit_button"):
        os.kill(os.getpid(), signal.SIGTERM)

if st.session_state.current_tab == "EXPORT":
    from ultralytics import YOLO
    st.title(body="Model Exporter")

    # Optimization Guide
    with st.expander("💡 Optimization Guide - Click to learn more"):
        st.markdown("""
        ### 🚀 How to make your model faster:

        **1. Use ONNX Format (Recommended)**
        - ✅ 2-3x faster than PyTorch
        - ✅ Smaller file size
        - ✅ No TensorRT dependency issues

        **2. Enable FP16 (Half Precision)**
        - ✅ 2x faster inference
        - ✅ 50% smaller model size
        - ⚠️ Only ~1-2% accuracy loss

        **3. Enable Simplify**
        - ✅ Optimizes ONNX graph
        - ✅ Removes redundant operations
        - ✅ No accuracy loss

        **4. Add NMS Module**
        - ✅ Faster post-processing
        - ✅ Better for real-time use
        - ⚠️ Slightly larger file

        **5. Use Smaller Image Size**
        - 320: Fastest, good for close targets
        - 480: Balanced speed/accuracy
        - 640: Best accuracy, slower

        ### 📊 Recommended Settings:
        - **For Gaming/Aimbot**: ONNX + FP16 + Simplify + NMS + 480 size
        - **For Best Quality**: ONNX + FP32 + 640 size
        - **For Maximum FPS**: ONNX + FP16 + Simplify + NMS + 320 size
        """)

    # Model selector
    models = []
    for root, dirs, files in os.walk("./models"):
        for file in files:
            if file.endswith(".pt"):
                models.append(file)

    selected_model = st.selectbox(
        label="**Select model to export.**",
        options=models,
        key="export_selected_model_selectbox"
    )

    # Image size
    image_size = st.radio(
        label="**Select model size**",
        options=(320, 480, 640),
        help="The size of the model image must be correct.",
        key="export_image_size_radio"
    )

    # Export Format
    export_format = st.selectbox(
        label="Export Format",
        options=["onnx", "engine", "torchscript"],
        index=0,
        help="ONNX is recommended. TensorRT (engine) requires TensorRT 8.6.x",
        key="export_format"
    )

    # Optimization Level
    st.subheader("⚡ Optimization Options")

    optimization_preset = st.selectbox(
        label="Optimization Preset",
        options=["Balanced (Recommended)", "Maximum Speed", "Maximum Accuracy", "Custom"],
        index=0,
        help="Presets balance speed vs accuracy",
        key="optimization_preset"
    )

    # Set defaults based on preset
    if optimization_preset == "Maximum Speed":
        default_half = True
        default_simplify = True
        default_dynamic = False
        default_nms = True
    elif optimization_preset == "Maximum Accuracy":
        default_half = False
        default_simplify = False
        default_dynamic = True
        default_nms = False
    else:  # Balanced
        default_half = True
        default_simplify = True
        default_dynamic = False
        default_nms = True

    # Precision (only for TensorRT)
    export_half: bool = False
    export_int8: bool = False
    data_yaml_path = None
    simplify = False
    dynamic = False

    if export_format == "engine":
        st.warning("⚠️ TensorRT export requires TensorRT 8.6.x. TensorRT 10.x is NOT compatible!")

        if optimization_preset == "Custom":
            export_precision = st.selectbox(
                label="Precision",
                index=0,
                options=["half", "int8"],
                key="export_precision"
            )
        else:
            export_precision = "half" if default_half else "full"
            st.info(f"Preset uses: {export_precision} precision")

        if export_precision == "half":
            export_half = True
        elif export_precision == "int8":
            export_int8 = True

        # Configuration file path for int8 calibration.
        if export_int8:
            data_yaml_path = st.text_input(
                label="Path to dataset configuration file",
                help="See logic/game.yaml for example",
                value="logic/game.yaml"
            )

    # ONNX-specific optimizations
    if export_format == "onnx":
        st.subheader("🔧 ONNX Optimization")

        if optimization_preset == "Custom":
            export_half = st.checkbox(
                label="Half precision (FP16)",
                value=True,
                help="2x faster, 50% smaller file, minimal accuracy loss",
                key="export_onnx_half"
            )

            simplify = st.checkbox(
                label="Simplify model",
                value=True,
                help="Optimize ONNX graph for faster inference",
                key="export_onnx_simplify"
            )

            dynamic = st.checkbox(
                label="Dynamic batch size",
                value=False,
                help="Enable dynamic input sizes (slower but flexible)",
                key="export_onnx_dynamic"
            )
        else:
            export_half = default_half
            simplify = default_simplify
            dynamic = default_dynamic

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("FP16", "✅" if export_half else "❌")
            with col2:
                st.metric("Simplify", "✅" if simplify else "❌")
            with col3:
                st.metric("Dynamic", "✅" if dynamic else "❌")

    # NMS (only for ONNX and TensorRT)
    add_nms: bool = False
    if export_format in ["onnx", "engine"]:
        if optimization_preset == "Custom":
            add_nms = st.toggle(
                label="Add NMS module",
                value=True,
                disabled=True if export_int8 == True else False,
                help="Faster post-processing but slightly larger model",
                key="export_add_nms_module"
            )
        else:
            add_nms = default_nms
            st.info(f"NMS module: {'✅ Enabled' if add_nms else '❌ Disabled'}")

    export_params = {
        "format": export_format,
        "imgsz": image_size,
        "half": export_half,
        "device": 0,
    }

    if export_format in ["onnx", "engine"]:
        export_params["nms"] = add_nms

    if export_format == "onnx":
        export_params["simplify"] = simplify
        export_params["dynamic"] = dynamic

    if export_format == "engine":
        export_params["int8"] = export_int8
        if export_int8 and data_yaml_path:
            export_params["data"] = data_yaml_path

    # Show optimization summary
    st.subheader("📊 Export Summary")
    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.write("**Settings:**")
        st.write(f"- Format: {export_format.upper()}")
        st.write(f"- Image Size: {image_size}")
        st.write(f"- Precision: {'FP16' if export_half else 'FP32'}")

    with summary_col2:
        st.write("**Expected Results:**")
        if export_format == "onnx" and export_half and simplify:
            st.success("⚡ Fast inference + Small file size")
        elif export_format == "onnx":
            st.info("✓ Good balance")
        elif export_format == "engine" and export_half:
            st.success("🚀 Maximum speed (requires TensorRT 8.6)")
        else:
            st.info("✓ Standard export")

    if st.button(label="Export model", key="export_export_model_button") and selected_model is not None:
        yolo_model = YOLO(f"./models/{selected_model}")

        with st.spinner(
            text=f"Model {selected_model} exporting...",
            show_time=True):
            try:
                exported_path = yolo_model.export(**export_params)

                if exported_path:
                    st.success(f"Model {selected_model} exported! `{exported_path}`", icon="✅")
                else:
                    st.error("Error with model export. See console window for log error.")
            except Exception as e:
                st.error(f"Error with model export: {str(e)}")
                st.code(str(e))

elif st.session_state.current_tab == "TESTS":
    def test_detections(
        input_model: str = None,
        source_method="Default",
        video_source=None,
        TOPMOST=True,
        model_image_size = None,
        input_device = 0,
        input_delay = 30,
        resize_factor = 100,
        ai_conf = 0.20):
        from ultralytics import YOLO

        if input_model is None:
            return ("error", "Model not selected")

        # Apply video source
        if source_method == "Default":
            default_source_video = "media/tests/test_det.mp4"

            if os.path.exists(default_source_video):
                video_source = default_source_video
            else:
                st.error(f"Default source media for detection tests not found!")
                return
        elif source_method == "Input file":
            video_source = video_source.getvalue()

            with open("uploaded_video.mp4", "wb") as f:
                f.write(video_source)

            video_source = "uploaded_video.mp4"

        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            st.error("Error: Could not open video.")
            return

        window_name = "Detections test"
        cv2.namedWindow(window_name)

        if TOPMOST:
            debug_window_hwnd = win32gui.FindWindow(None, window_name)
            win32gui.SetWindowPos(debug_window_hwnd, win32con.HWND_TOPMOST, 100, 100, 200, 200, 0)

        model = YOLO(f'models/{input_model}', task='detect')

        while cap.isOpened():
            success, frame = cap.read()
            if success:
                result = model(frame, stream=False, show=False, imgsz=model_image_size, device=input_device, verbose=False, conf=ai_conf)

                annotated_frame = result[0].plot()

                cv2.putText(annotated_frame, "Press 'q' to quit", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

                frame_height, frame_width = frame.shape[:2]
                height = int(frame_height * resize_factor / 100)
                width = int(frame_width * resize_factor / 100)
                dim = (width, height)
                cv2.resizeWindow(window_name, dim)
                resised = cv2.resize(annotated_frame, dim, cv2.INTER_NEAREST)
                cv2.imshow(window_name, resised)
                if cv2.waitKey(input_delay) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

        if source_method == "Input file":
            try:
                os.remove("./uploaded_video.mp4")
            except: pass

        del model

    st.title("Tests")

    models = []
    for root, dirs, files in os.walk("./models"):
        for file in files:
            if file.endswith(".pt") or file.endswith(".engine") or file.endswith(".onnx"):
                models.append(file)

    # SELECT MODEL
    ai_model = st.selectbox(
        label="AI Model",
        options=models,
        key="TESTS_ai_model_selectbox",
        help="Put model to './models' path."
    )

    # SELECT MODEL IMAGE SIZE
    model_image_sizes = [320, 480, 640]
    model_size = st.selectbox(
        label="AI Model image size",
        options=model_image_sizes,
        key="TESTS_model_size_selectbox",
        index=2
    )

    # VIDEO SOURCE
    methods = ["Default", "Input file"]
    video_source_method = st.selectbox(
        label="Select video input method",
        options=methods,
        index=0,
        key="TESTS_video_source_method_selectbox"
    )

    # TOPMOST
    TOPMOST = st.toggle(
        label="Test window on top",
        value=True,
        key="tests_topmost"
    )

    # DEVICE
    test_devices = ["cpu", "0", "1", "2", "3", "4", "5"]
    device = st.selectbox(
        label="Device",
        options=test_devices,
        index=1,
        key="tests_test_devices"
    )

    if device != "cpu":
        device = int(device)

    # DELAY
    cv2_delay = st.number_input(
        label="CV2 frame wait delay",
        min_value=1,
        max_value=120,
        step=1,
        format="%u",
        value=30,
        key="TESTS_cv2_delay_number_input"
    )

    # RESIZE
    cv2_resize = st.number_input(
        label="Resize test window",
        min_value=10,
        max_value=100,
        value=80,
        step=1,
        format="%u",
        key="ESTS_cv2_resize_number_input"
    )

    # DETECTION CONF
    ai_conf = st.number_input(
        label="Minimum confidence threshold",
        min_value=0.01,
        max_value=0.99,
        step=0.01,
        format="%.2f",
        value=0.20,
        key="tests_ai_conf"
    )

    input_video = None
    if video_source_method == "Input file":
        video_source_input_file = st.file_uploader(
            label="Import video file",
            accept_multiple_files=False,
            type=(["mp4"]),
            key="TESTS_input_file_video_source_input_file"
        )

        input_video = video_source_input_file

    if st.button(label="Test detections", key="TESTS_text_detections_button"):
        if video_source_method in methods:
            if input_video == None and video_source_method == "Input file":
                st.error("Video source not found.")
            else:
                test_detections(
                    input_model=ai_model,
                    source_method=video_source_method,
                    video_source=input_video,
                    model_image_size=model_size,
                    TOPMOST=TOPMOST,
                    input_delay=cv2_delay,
                    input_device=device,
                    resize_factor=cv2_resize,
                    ai_conf=ai_conf
                )
        else:
            st.error("Select correct video input method.")
