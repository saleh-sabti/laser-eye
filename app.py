"""Local web UI tying the vision pipeline together.

Pages: dashboard, settings (camera/bed config), calibration wizard,
live detection view, and design upload/placement/export. Runs locally;
open it from a phone/tablet on the same network via http://<host-ip>:5000.
"""
import io
import math
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for, send_file, flash, jsonify

# OpenCV's internal thread pool can deadlock when cv2 calls are made from a
# Python worker thread after cv2 has been used on the main thread (camera
# reads, detection) - which is exactly what the photo-trace job does. Seen
# here: background removal finished, then cv2.findContours in the worker
# thread hung forever. Disabling OpenCV's own parallelism fixes it; the ops
# involved are fast enough single-threaded.
cv2.setNumThreads(0)

from laser_align import config, calibration, detection, align, export, photo_align, dataset, rfdetr_detect, grbl, aruco_calibration, placement
from laser_align.camera import Camera, probe_devices
from laser_align.calibration import CalibrationError
from laser_align.detection import NoReferenceFrameError, NoObjectFoundError
from laser_align.grbl import Grbl, GrblError, GrblAlarmError
from laser_align.aruco_calibration import NoMarkersConfiguredError, NotEnoughMarkersDetectedError

app = Flask(__name__)
app.secret_key = "laser-auto-align-local"  # local-only tool, no real auth surface

_camera: Camera | None = None
_camera_lock = threading.Lock()
_last_export_path: str | None = None
_last_export_kind: str | None = None   # "svg" or "png"
_last_placement_info: dict | None = None  # only set for photo (png) exports
_grbl: Grbl | None = None


def get_camera() -> Camera:
    """Get the shared Camera, creating/recreating it if settings changed.

    Guarded by a lock: without it, two requests that both find `_camera is
    None` (e.g. a page that fires a snapshot and a marker-detect at once on
    first load) each construct a `cv2.VideoCapture` on the same device index
    concurrently, which crashes the process natively with no traceback.
    `Camera.read()` has its own lock for concurrent frame grabs; this one
    only covers construction/replacement.
    """
    global _camera
    with _camera_lock:
        settings = config.load_settings()
        width, height, rotate_180 = settings["camera_width"], settings["camera_height"], settings["camera_rotate_180"]
        if (
            _camera is None
            or _camera.index != settings["camera_index"]
            or _camera.width != width
            or _camera.height != height
            or _camera.rotate_180 != rotate_180
        ):
            if _camera is not None:
                _camera.release()
            _camera = Camera(settings["camera_index"], width, height, rotate_180)
        return _camera


@app.route("/")
def index():
    settings = config.load_settings()
    return render_template(
        "index.html",
        settings=settings,
        camera_open=get_camera().is_open,
        has_calibration=calibration.load_homography() is not None,
        has_reference=detection.has_reference_frame(),
    )


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    if request.method == "POST":
        # The "Export coordinates" card is handled on its own and returns -
        # it carries a checkbox (flip_y), and checkboxes can't use the
        # .get()-fallback trick the other cards rely on (an absent checkbox
        # is indistinguishable from an unchecked one), so it must not fall
        # through into the camera block below.
        if "coord_form" in request.form:
            config.save_settings({"flip_y": "flip_y" in request.form})
            flash("Settings saved.")
            return redirect(url_for("settings_page"))

        # Each card on the page submits its own independent form - .get()
        # with a fallback to the current value means one card's submit
        # can't accidentally wipe fields that belong to another card.
        current = config.load_settings()
        new_width = int(request.form.get("camera_width", current["camera_width"]))
        new_height = int(request.form.get("camera_height", current["camera_height"]))
        new_rotate_180 = "camera_rotate_180" in request.form
        config.save_settings({
            "camera_index": int(request.form.get("camera_index", current["camera_index"])),
            "camera_width": new_width,
            "camera_height": new_height,
            "camera_rotate_180": new_rotate_180,
            "bed_width_mm": float(request.form.get("bed_width_mm", current["bed_width_mm"])),
            "bed_height_mm": float(request.form.get("bed_height_mm", current["bed_height_mm"])),
            "detection_method": request.form.get("detection_method", current["detection_method"]),
            "grbl_port": request.form.get("grbl_port", current["grbl_port"]),
            "grbl_baud": int(request.form.get("grbl_baud", current["grbl_baud"])),
        })
        resolution_changed = new_width != current["camera_width"] or new_height != current["camera_height"]
        rotation_changed = new_rotate_180 != current["camera_rotate_180"]
        if resolution_changed or rotation_changed:
            # Saved calibration/bed-area are pixel coordinates at whatever
            # resolution/orientation they were set at - silently keeping
            # them after either changes would misalign everything without
            # any obvious symptom, so clear them out and make the user
            # redo both deliberately rather than debug a mystery later.
            config.CALIBRATION_PATH.unlink(missing_ok=True)
            config.save_settings({"bed_roi_px": None})
            flash(
                "Camera resolution/orientation changed - calibration and bed area were "
                "cleared since they're tied to the old settings. Redo both."
            )
        else:
            flash("Settings saved.")
        return redirect(url_for("settings_page"))
    return render_template("settings.html", settings=config.load_settings())


@app.route("/settings/probe")
def settings_probe():
    return jsonify({"devices": probe_devices()})


def _jpeg_response(frame) -> Response:
    ok, buf = cv2.imencode(".jpg", frame)
    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.route("/calibration")
def calibration_page():
    return render_template(
        "calibration.html",
        has_reference=detection.has_reference_frame(),
        settings=config.load_settings(),
        aruco_markers=aruco_calibration.load_marker_positions(),
        aruco_min_required=aruco_calibration.MIN_MARKERS_REQUIRED,
    )


@app.route("/calibration/snapshot.jpg")
def calibration_snapshot():
    return _jpeg_response(get_camera().read())


@app.route("/calibration/capture_reference", methods=["POST"])
def calibration_capture_reference():
    detection.save_reference_frame(get_camera().read())
    flash("Empty-bed reference frame captured.")
    return redirect(url_for("calibration_page"))


@app.route("/calibration/aruco/marker/<int:marker_id>.png")
def aruco_marker_image(marker_id):
    img = aruco_calibration.generate_marker_image(marker_id)
    ok, buf = cv2.imencode(".png", img)
    return Response(buf.tobytes(), mimetype="image/png")


@app.route("/calibration/aruco/sheet.pdf")
def aruco_marker_sheet():
    paper = request.args.get("paper", "a4")
    if paper not in ("a4", "letter"):
        paper = "a4"
    return Response(
        aruco_calibration.build_marker_sheet(paper),
        mimetype="application/pdf",
        headers={"Content-Disposition": f'inline; filename="aruco_markers_{paper}.pdf"'},
    )


@app.route("/calibration/aruco/register", methods=["POST"])
def aruco_register():
    try:
        marker_id = int(request.form["marker_id"])
        x = float(request.form["machine_x_mm"])
        y = float(request.form["machine_y_mm"])
    except (KeyError, ValueError):
        flash("Enter a marker ID and its machine X/Y before registering.")
        return redirect(url_for("calibration_page"))
    aruco_calibration.save_marker_position(marker_id, x, y)
    flash(f"Marker {marker_id} registered at ({x}, {y}) mm.")
    return redirect(url_for("calibration_page"))


@app.route("/calibration/aruco/remove/<int:marker_id>", methods=["POST"])
def aruco_remove(marker_id):
    aruco_calibration.remove_marker_position(marker_id)
    flash(f"Marker {marker_id} removed.")
    return redirect(url_for("calibration_page"))


@app.route("/calibration/aruco/calibrate_now", methods=["POST"])
def aruco_calibrate_now():
    settings = config.load_settings()
    frame = get_camera().read()
    try:
        aruco_calibration.auto_calibrate(frame, settings["camera_width"], settings["camera_height"])
        flash("Auto-calibrated and set the bed area from the four markers.")
    except (NoMarkersConfiguredError, NotEnoughMarkersDetectedError, CalibrationError) as e:
        flash(f"Auto-calibration failed: {e}")
    return redirect(url_for("calibration_page"))


@app.route("/calibration/set_roi", methods=["POST"])
def calibration_set_roi():
    try:
        x0, y0, x1, y1 = (
            int(request.form["x0"]), int(request.form["y0"]),
            int(request.form["x1"]), int(request.form["y1"]),
        )
    except (KeyError, ValueError):
        flash("Click two corners on the photo before setting the bed area.")
        return redirect(url_for("calibration_page"))
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    if x1 - x0 < 20 or y1 - y0 < 20:
        flash("That bed area looks too small - click two corners further apart.")
        return redirect(url_for("calibration_page"))
    config.save_settings({"bed_roi_px": [x0, y0, x1, y1]})
    flash("Bed area set - detection will now ignore everything outside it.")
    return redirect(url_for("calibration_page"))


@app.route("/calibration/clear_roi", methods=["POST"])
def calibration_clear_roi():
    config.save_settings({"bed_roi_px": None})
    flash("Bed area cleared - detection will consider the whole frame again.")
    return redirect(url_for("calibration_page"))


@app.route("/calibration/grbl/connect", methods=["POST"])
def grbl_connect():
    global _grbl
    settings = config.load_settings()
    if _grbl is not None:
        flash("Already connected to the machine.")
        return redirect(url_for("calibration_page"))
    try:
        _grbl = Grbl(settings["grbl_port"], settings["grbl_baud"])
        flash(f"Connected to {settings['grbl_port']}. Laser forced off for safety while jogging.")
    except GrblError as e:
        flash(f"Couldn't connect: {e}")
    return redirect(url_for("calibration_page"))


@app.route("/calibration/grbl/disconnect", methods=["POST"])
def grbl_disconnect():
    global _grbl
    if _grbl is not None:
        _grbl.close()
        _grbl = None
        flash("Disconnected - LightBurn can reconnect to the machine now.")
    return redirect(url_for("calibration_page"))


def _grbl_call(fn):
    """Run a Grbl operation and turn any failure into a JSON-friendly
    result, dropping the global connection if it's actually dead (a broken
    serial link, e.g. from a limit-switch electrical glitch) instead of
    leaving the app repeating the same failing call on every future poll -
    the UI will then correctly show 'not connected' and just needs Connect
    clicked again.
    """
    global _grbl
    if _grbl is None:
        return {"ok": False, "error": "Not connected to the machine."}
    try:
        fn(_grbl)
        return {"ok": True}
    except GrblAlarmError as e:
        return {"ok": False, "error": str(e), "alarm": True}
    except GrblError as e:
        if _grbl.is_broken:
            _grbl = None
        return {"ok": False, "error": str(e)}


@app.route("/calibration/grbl/jog", methods=["POST"])
def grbl_jog():
    # AJAX endpoint (called via fetch, no page reload) - so errors must come
    # back as JSON the page can display, not a flash()+redirect nobody sees.
    dx = float(request.form.get("dx", 0))
    dy = float(request.form.get("dy", 0))
    return jsonify(_grbl_call(lambda g: g.jog_relative(dx, dy)))


@app.route("/calibration/grbl/unlock", methods=["POST"])
def grbl_unlock():
    return jsonify(_grbl_call(lambda g: g.unlock()))


@app.route("/calibration/grbl/home", methods=["POST"])
def grbl_home():
    return jsonify(_grbl_call(lambda g: g.home()))


@app.route("/calibration/grbl/goto", methods=["POST"])
def grbl_goto():
    x = float(request.form.get("x", 0))
    y = float(request.form.get("y", 0))
    return jsonify(_grbl_call(lambda g: g.jog_to(x, y)))


@app.route("/calibration/grbl/set_origin", methods=["POST"])
def grbl_set_origin():
    return jsonify(_grbl_call(lambda g: g.set_origin_here()))


@app.route("/calibration/grbl/position")
def grbl_position():
    global _grbl
    if _grbl is None:
        return jsonify({"connected": False})
    try:
        state = _grbl.get_state()
        x, y = _grbl.get_work_position()
        return jsonify({"connected": True, "x": round(x, 3), "y": round(y, 3), "state": state})
    except GrblError as e:
        if _grbl.is_broken:
            _grbl = None
            return jsonify({"connected": False, "error": str(e)})
        return jsonify({"connected": True, "error": str(e)})


@app.route("/live")
def live_page():
    return render_template(
        "live.html",
        has_reference=detection.has_reference_frame(),
        has_calibration=calibration.load_homography() is not None,
        grbl_connected=_grbl is not None,
    )


@app.route("/training")
def training_page():
    return render_template(
        "training.html",
        sample_count=dataset.sample_count(),
        rfdetr_trained=rfdetr_detect.is_trained(),
    )


@app.route("/training/preview.jpg")
def training_preview():
    frame = get_camera().read()
    det, _ = _detect_current(frame)
    if det is not None:
        frame = detection.draw_overlay(frame, det)
    return _jpeg_response(frame)


@app.route("/training/collect", methods=["POST"])
def training_collect():
    frame = get_camera().read()
    homography = calibration.load_homography()
    if homography is None:
        flash("Calibrate first - training samples need real mm coordinates too.")
        return redirect(url_for("training_page"))
    try:
        det = detection.detect_object(frame, homography)
    except (NoReferenceFrameError, NoObjectFoundError) as e:
        flash(f"Couldn't save sample: {e}")
        return redirect(url_for("training_page"))

    total = dataset.add_sample(frame, det)
    flash(f"Sample saved ({total} total). Move the piece and collect another when ready.")
    return redirect(url_for("training_page"))


def _detect_current(cam_frame):
    homography = calibration.load_homography()
    if homography is None:
        return None, "Not calibrated yet."

    settings = config.load_settings()
    roi = settings.get("bed_roi_px")
    roi = tuple(roi) if roi else None

    method = settings["detection_method"]
    if method == "rfdetr":
        try:
            det = rfdetr_detect.detect_object(cam_frame, homography)
            return det, None
        except rfdetr_detect.ModelNotTrainedError:
            pass  # not trained yet - silently fall through to classical below.
            # (this hot path runs per streamed frame, so no flash() here -
            # Settings/Dashboard show whether RF-DETR is actually trained)
        except NoObjectFoundError as e:
            return None, str(e)

    try:
        det = detection.detect_object(cam_frame, homography, roi=roi)
        return det, None
    except (NoReferenceFrameError, NoObjectFoundError) as e:
        return None, str(e)


@app.route("/live/stream")
def live_stream():
    def generate():
        while True:
            frame = get_camera().read()
            det, _ = _detect_current(frame)
            if det is not None:
                frame = detection.draw_overlay(frame, det)
            ok, buf = cv2.imencode(".jpg", frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.1)

    resp = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    return resp


PHOTO_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Hand-placement editor state (single-user local tool, module globals like
# the rest of app.py). Set when an SVG is uploaded, used by the editor page.
_place_polylines: list | None = None          # design polylines, mm, centred on origin
_place_size_mm: tuple[float, float] | None = None
_place_bed_jpg: bytes | None = None           # frozen bed photo the editor draws on
_place_detection: detection.Detection | None = None
_place_cutout = None                           # PIL image from a photo upload, traced lazily
_place_cutout_long_mm: float | None = None

# Photo processing runs on a background thread so the page can show a
# progress bar - rembg's background removal is one blocking call of a few
# to tens of seconds with no progress of its own, so the "percent" is a
# time estimate that eases toward (never past) 95% until it actually
# finishes, plus a rolling average of how long recent runs took.
_photo_job: dict | None = None
_photo_job_lock = threading.Lock()
_photo_avg_s = 8.0

_PHOTO_STAGES = [
    ("removing background", 0.05, 0.92),   # (label, pct_start, pct_end)
]


def _run_photo_job(photo_bytes: bytes, frame_jpg: bytes, det: "detection.Detection",
                   place_url: str = "/design/place") -> None:
    """Background-remove a raster upload. Only the rembg call runs here -
    the cv2 trace is done later in the request thread (`design_place`),
    because cv2.findContours in this daemon thread deadlocks once torch is
    loaded in the process (rfdetr detection method). Hands the cutout to the
    editor via module globals."""
    global _photo_job, _photo_avg_s
    global _place_cutout, _place_cutout_long_mm, _place_bed_jpg, _place_detection
    global _place_polylines, _last_export_path
    t0 = time.time()
    try:
        with _photo_job_lock:
            _photo_job.update(stage="removing background", stage_i=0, t0=t0)
        cutout = photo_align._autocrop_to_subject(photo_align.remove_background(photo_bytes))
    except Exception as e:
        with _photo_job_lock:
            _photo_job.update(state="error", error=str(e), done=True)
        return

    took = time.time() - t0
    _photo_avg_s = 0.7 * _photo_avg_s + 0.3 * max(took, 1.0)

    _place_cutout = cutout
    _place_cutout_long_mm = (max(det.length_mm, det.width_mm)
                             if det is not None else placement.DEFAULT_TRACE_LONG_MM)
    _place_detection = det
    _place_bed_jpg = frame_jpg
    _place_polylines = None
    _last_export_path = None
    with _photo_job_lock:
        _photo_job.update(state="done", done=True, took_s=round(took, 1),
                          redirect=place_url)


def _photo_job_progress() -> dict:
    """Current job status with an eased percentage derived from elapsed time
    vs the rolling average."""
    with _photo_job_lock:
        j = dict(_photo_job) if _photo_job else None
    if j is None:
        return {"state": "idle"}
    if j.get("done"):
        pct = 100 if j.get("state") == "done" else j.get("pct", 0)
        return {
            "state": j["state"], "pct": pct, "error": j.get("error"),
            "stage": j.get("stage", ""), "elapsed_s": round(time.time() - j["t0"], 1),
            "redirect": j.get("redirect"),
        }
    elapsed = time.time() - j["t0"]
    if elapsed > max(150.0, 8 * _photo_avg_s):   # watchdog - don't let the UI hang forever
        return {"state": "error", "error":
                f"Timed out after {elapsed:.0f}s - the design may be too complex, or "
                "background removal stalled. Try a simpler image."}
    lo, hi = _PHOTO_STAGES[j.get("stage_i", 0)][1], _PHOTO_STAGES[j.get("stage_i", 0)][2]
    frac = 1.0 - math.exp(-elapsed / max(_photo_avg_s, 1.0))   # eases toward 1, never reaches
    pct = round(100 * (lo + (hi - lo) * frac))
    return {"state": "working", "pct": pct, "stage": j.get("stage", ""),
            "elapsed_s": round(elapsed, 1), "est_s": round(_photo_avg_s, 1)}


@app.route("/design", methods=["GET", "POST"])
def design_page():
    global _last_export_path, _last_export_kind, _last_placement_info
    global _place_polylines, _place_size_mm, _place_bed_jpg, _place_detection, _photo_job

    if request.method == "POST":
        # The upload form submits via fetch, so answers are always JSON:
        # {redirect: ...} for an SVG (straight to the editor), or
        # {photo_job: true} for a photo (frontend then polls the progress).
        design_file = request.files.get("design_file")
        if not design_file or design_file.filename == "":
            return jsonify({"ok": False, "error": "Choose a design file first (SVG, PNG, or JPEG)."})

        ext = Path(design_file.filename).suffix.lower()
        if calibration.load_homography() is None:
            return jsonify({"ok": False, "error": "Calibrate first - designs are placed in real machine mm."})

        frame = get_camera().read()
        det, _ = _detect_current(frame)   # optional for SVG, required for photo

        if ext == ".svg":
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_in:
                design_file.save(tmp_in.name)
                svg_tmp_path = tmp_in.name
            try:
                polylines, w_mm, h_mm = placement.load_svg_mm(svg_tmp_path)
            except ValueError as e:
                return jsonify({"ok": False, "error": f"Couldn't read that SVG: {e}"})
            finally:
                Path(svg_tmp_path).unlink(missing_ok=True)

            _place_polylines = polylines
            _place_size_mm = (w_mm, h_mm)
            _place_detection = det
            ok, buf = cv2.imencode(".jpg", frame)
            _place_bed_jpg = buf.tobytes()
            _last_export_path = None
            return jsonify({"ok": True, "redirect": url_for("design_place")})

        elif ext in PHOTO_EXTENSIONS:
            if det is None:
                return jsonify({"ok": False, "error":
                    "Photo alignment needs the workpiece detected - check the reference frame and bed."})
            photo_bytes = design_file.read()
            ok, buf = cv2.imencode(".jpg", frame)
            with _photo_job_lock:
                _photo_job = {"state": "queued", "t0": time.time(), "stage": "starting", "stage_i": 0}
            # resolve the redirect URL here, in the request context - url_for()
            # blows up in the worker thread ("working outside of application context")
            place_url = url_for("design_place")
            threading.Thread(target=_run_photo_job,
                             args=(photo_bytes, buf.tobytes(), det, place_url), daemon=True).start()
            return jsonify({"ok": True, "photo_job": True})

        return jsonify({"ok": False, "error": f"Unsupported file type '{ext}' - use .svg, .png, or .jpg."})

    return render_template(
        "design.html",
        has_calibration=calibration.load_homography() is not None,
        has_placement=_place_polylines is not None,
        last_export=_last_export_path is not None,
        export_kind=_last_export_kind,
        placement=_last_placement_info,
    )


@app.route("/design/photo_status")
def design_photo_status():
    return jsonify(_photo_job_progress())


@app.route("/design/place")
def design_place():
    global _place_polylines, _place_size_mm, _place_cutout
    # A photo upload leaves a cutout to be traced - do it here, in the
    # request thread (cv2.findContours deadlocks in the upload's daemon
    # thread once torch is loaded in the process).
    if _place_polylines is None and _place_cutout is not None:
        try:
            polylines, w_mm, h_mm = placement.trace_image_to_mm(
                _place_cutout, long_side_mm=_place_cutout_long_mm or placement.DEFAULT_TRACE_LONG_MM)
        except ValueError as e:
            _place_cutout = None
            flash(f"Couldn't trace that image: {e}")
            return redirect(url_for("design_page"))
        _place_polylines, _place_size_mm, _place_cutout = polylines, (w_mm, h_mm), None

    if _place_polylines is None:
        flash("Upload a design first.")
        return redirect(url_for("design_page"))
    return render_template("design_editor.html", grbl_connected=_grbl is not None)


@app.route("/design/bed.jpg")
def design_bed():
    if _place_bed_jpg is None:
        return _jpeg_response(get_camera().read())
    resp = Response(_place_bed_jpg, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/design/state.json")
def design_state():
    if _place_polylines is None:
        return jsonify({"ok": False})
    settings = config.load_settings()
    H = calibration.load_homography()
    H_inv = np.linalg.inv(H)
    det = _place_detection
    w_mm, h_mm = _place_size_mm
    return jsonify({
        "ok": True,
        "polylines": _place_polylines,
        "design_w_mm": round(w_mm, 2), "design_h_mm": round(h_mm, 2),
        "bed_w_mm": settings["bed_width_mm"], "bed_h_mm": settings["bed_height_mm"],
        "px_to_mm": H.flatten().tolist(),
        "mm_to_px": H_inv.flatten().tolist(),
        "detection": None if det is None else {
            "contour_mm": det.contour_mm.tolist(),
            "centroid_mm": list(det.centroid_mm),
            "angle_deg": det.angle_deg,
            "length_mm": det.length_mm, "width_mm": det.width_mm,
        },
    })


@app.route("/design/rebed", methods=["POST"])
def design_rebed():
    """Re-grab the bed photo (and re-detect) without re-uploading the design."""
    global _place_bed_jpg, _place_detection
    frame = get_camera().read()
    _place_detection, _ = _detect_current(frame)
    ok, buf = cv2.imencode(".jpg", frame)
    _place_bed_jpg = buf.tobytes()
    return jsonify({"ok": True})


@app.route("/design/place/export", methods=["POST"])
def design_place_export():
    global _last_export_path, _last_export_kind, _last_placement_info
    if _place_polylines is None:
        return jsonify({"ok": False, "error": "No design loaded."})
    settings = config.load_settings()
    try:
        tx = float(request.form["tx_mm"]); ty = float(request.form["ty_mm"])
        rot = float(request.form["rot_deg"]); scale = float(request.form["scale"])
    except (KeyError, ValueError):
        return jsonify({"ok": False, "error": "Bad transform values."})
    clip = request.form.get("clip") == "1"
    flip_x = request.form.get("flip_x") == "1"

    placed = placement.place(_place_polylines, tx, ty, rot, scale, flip_x=flip_x)
    if clip and _place_detection is not None:
        placed = placement.clip_to_outline(placed, _place_detection.contour_mm)
        if not placed:
            return jsonify({"ok": False, "error": "Clipping left nothing - the design is entirely outside the piece."})

    # sanity guard: the placed geometry must be non-degenerate and land on the
    # bed. (Not a centre-equality check - place() pins the design's *un-rotated*
    # bbox centre to (tx, ty), so the axis-aligned bbox centre legitimately
    # shifts once the design is rotated or clipped; that's WYSIWYG with the
    # editor canvas, which applies the identical transform.)
    xs = [x for line in placed for x, _ in line]
    ys = [y for line in placed for _, y in line]
    bed_w, bed_h = settings["bed_width_mm"], settings["bed_height_mm"]
    if not xs or max(xs) - min(xs) < 1.0 or max(ys) - min(ys) < 1.0:
        return jsonify({"ok": False, "error":
            "Internal check failed: the placed design came out with near-zero size - check the scale."})
    if min(xs) < -1.0 or min(ys) < -1.0 or max(xs) > bed_w + 1.0 or max(ys) > bed_h + 1.0:
        return jsonify({"ok": False, "error":
            f"The placed design runs off the bed: X {min(xs):.0f}..{max(xs):.0f}, "
            f"Y {min(ys):.0f}..{max(ys):.0f} mm (bed {bed_w:.0f} x {bed_h:.0f} mm)."})

    aligned = placement.to_aligned_design(placed)
    out_path = str(Path(tempfile.gettempdir()) / "laser_align_export.svg")
    export.write_svg(out_path, aligned, settings["bed_width_mm"], settings["bed_height_mm"],
                     flip_y=settings["flip_y"])
    _last_export_path = out_path
    _last_export_kind = "svg"
    _last_placement_info = None

    return jsonify({
        "ok": True,
        "center_mm": [round(tx, 1), round(ty, 1)],
        "bounds_mm": [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)],
        "flip_y": settings["flip_y"],
    })


@app.route("/design/preview.jpg")
def design_preview():
    frame = get_camera().read()
    det, _ = _detect_current(frame)
    if det is not None:
        frame = detection.draw_overlay(frame, det)
    return _jpeg_response(frame)


@app.route("/design/export")
def design_export():
    if _last_export_path is None:
        flash("Align or place a design first.")
        return redirect(url_for("design_page"))
    if _last_export_kind == "png":
        return send_file(_last_export_path, mimetype="image/png", as_attachment=True, download_name="aligned_design.png")
    return send_file(_last_export_path, mimetype="image/svg+xml", as_attachment=True, download_name="placed_design.svg")


@app.route("/live/snapshot.jpg")
def live_snapshot():
    """One frame for Live View - polled a couple times a second rather than
    streamed, so the page only ever has one camera reader (three concurrent
    ones - stream + snapshot + marker detect - crashed the process). The
    detected outline is drawn on unless ?plain=1 (the placement check wants
    a clean frame to click)."""
    frame = get_camera().read()
    if request.args.get("plain") != "1":
        det, _ = _detect_current(frame)
        if det is not None:
            frame = detection.draw_overlay(frame, det)
    return _jpeg_response(frame)


@app.route("/live/markers.json")
def live_markers():
    """Every ArUco marker visible right now: its sub-pixel reference-corner
    pixel, the machine mm it was registered at, and the machine mm the
    current calibration maps that pixel to. Lets the placement check snap a
    click exactly onto a known corner and show the calibration error there -
    no estimating."""
    frame = get_camera().read()
    detected = aruco_calibration.detect_markers(frame)
    registered = aruco_calibration.load_marker_positions()
    homography = calibration.load_homography()

    out = []
    for mid, (px, py) in sorted(detected.items()):
        entry = {
            "id": mid,
            "corner": aruco_calibration.CORNER_NAME.get(mid, "?"),
            "px": round(px, 2), "py": round(py, 2),
        }
        if mid in registered:
            entry["reg_x"], entry["reg_y"] = registered[mid]
        if homography is not None:
            mx, my = calibration.pixels_to_mm(homography, np.array([[px, py]]))[0]
            entry["map_x"], entry["map_y"] = round(float(mx), 1), round(float(my), 1)
            if mid in registered:
                entry["err_mm"] = round(
                    float(np.hypot(mx - registered[mid][0], my - registered[mid][1])), 1
                )
        out.append(entry)
    max_err = max((m["err_mm"] for m in out if "err_mm" in m), default=None)
    return jsonify({"markers": out, "max_err_mm": max_err,
                    "calibrated": homography is not None})


@app.route("/live/pixel_to_mm", methods=["POST"])
def live_pixel_to_mm():
    """Camera pixel -> machine mm, no jogging. Lets the Live View readout
    show where a click lands in real coordinates the instant you click,
    which is the quickest read on whether calibration is any good."""
    homography = calibration.load_homography()
    if homography is None:
        return jsonify({"ok": False, "error": "Not calibrated yet."})
    try:
        px = float(request.form["px"])
        py = float(request.form["py"])
    except (KeyError, ValueError):
        return jsonify({"ok": False, "error": "Bad pixel coordinates."})
    x_mm, y_mm = calibration.pixels_to_mm(homography, np.array([[px, py]]))[0]
    x_mm, y_mm = float(x_mm), float(y_mm)
    settings = config.load_settings()
    bed_w, bed_h = settings["bed_width_mm"], settings["bed_height_mm"]
    # ~10mm grace so a click just off a corner marker doesn't read as an
    # error; "way_off" (a genuinely broken mapping) is a separate, louder flag.
    grace = 10.0
    in_bed = (-grace <= x_mm <= bed_w + grace) and (-grace <= y_mm <= bed_h + grace)
    way_off = not (-100 <= x_mm <= bed_w + 100) or not (-100 <= y_mm <= bed_h + 100)
    return jsonify({
        "ok": True, "x_mm": round(x_mm, 1), "y_mm": round(y_mm, 1),
        "in_bed": in_bed, "way_off": way_off, "bed_w": bed_w, "bed_h": bed_h,
    })


@app.route("/live/jog_to_point", methods=["POST"])
def live_jog_to_point():
    """Click a point on the live camera feed -> jog the head there. The
    honest end-to-end check that a clicked spot maps to the real machine
    position it claims (calibration is right / the coordinate convention
    holds). Reuses the calibration jog connection; never fires the laser.

    Takes camera-pixel coordinates and runs them through the same
    homography everything else uses, so no separate rectified view is
    needed.
    """
    homography = calibration.load_homography()
    if homography is None:
        return jsonify({"ok": False, "error": "Not calibrated yet - see the Calibration page."})
    try:
        px = float(request.form["px"])
        py = float(request.form["py"])
    except (KeyError, ValueError):
        return jsonify({"ok": False, "error": "Missing or bad pixel coordinates."})

    x_mm, y_mm = calibration.pixels_to_mm(homography, np.array([[px, py]]))[0]
    x_mm, y_mm = float(x_mm), float(y_mm)

    settings = config.load_settings()
    bed_w, bed_h = settings["bed_width_mm"], settings["bed_height_mm"]
    margin = 15.0
    if not (-margin <= x_mm <= bed_w + margin) or not (-margin <= y_mm <= bed_h + margin):
        return jsonify({
            "ok": False,
            "error": f"That maps to ({x_mm:.0f}, {y_mm:.0f}) mm - outside the "
                     f"{bed_w:.0f}x{bed_h:.0f}mm bed. Bad calibration, or you clicked off the bed.",
            "x_mm": round(x_mm, 1), "y_mm": round(y_mm, 1),
        })

    result = _grbl_call(lambda g: g.jog_to(x_mm, y_mm))
    result["x_mm"] = round(x_mm, 1)
    result["y_mm"] = round(y_mm, 1)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
