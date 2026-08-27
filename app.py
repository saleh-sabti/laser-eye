"""Local web UI tying the vision pipeline together.

Pages: dashboard, settings (camera/bed config), calibration wizard,
live detection view, and design upload/placement/export. Runs locally;
open it from a phone/tablet on the same network via http://<host-ip>:5000.
"""
import io
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, render_template, request, redirect, url_for, send_file, flash, jsonify

from laser_align import config, calibration, detection, align, export, photo_align, dataset, rfdetr_detect, grbl, aruco_calibration
from laser_align.camera import Camera, probe_devices
from laser_align.calibration import CalibrationError
from laser_align.detection import NoReferenceFrameError, NoObjectFoundError
from laser_align.grbl import Grbl, GrblError, GrblAlarmError
from laser_align.aruco_calibration import NoMarkersConfiguredError, NotEnoughMarkersDetectedError

app = Flask(__name__)
app.secret_key = "laser-auto-align-local"  # local-only tool, no real auth surface

_camera: Camera | None = None
_last_export_path: str | None = None
_last_export_kind: str | None = None   # "svg" or "png"
_last_placement_info: dict | None = None  # only set for photo (png) exports
_grbl: Grbl | None = None


def get_camera() -> Camera:
    global _camera
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


@app.route("/design", methods=["GET", "POST"])
def design_page():
    global _last_export_path, _last_export_kind, _last_placement_info
    settings = config.load_settings()

    if request.method == "POST":
        design_file = request.files.get("design_file")
        if not design_file or design_file.filename == "":
            flash("Choose a design file first (SVG, PNG, or JPEG).")
            return redirect(url_for("design_page"))

        ext = Path(design_file.filename).suffix.lower()
        frame = get_camera().read()
        det, err = _detect_current(frame)
        if err:
            flash(f"Detection failed: {err}")
            return redirect(url_for("design_page"))

        if ext == ".svg":
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp_in:
                design_file.save(tmp_in.name)
                svg_tmp_path = tmp_in.name
            try:
                aligned = align.align_and_clip(svg_tmp_path, det)
            except ValueError as e:
                flash(f"Alignment failed: {e}")
                return redirect(url_for("design_page"))
            finally:
                Path(svg_tmp_path).unlink(missing_ok=True)

            out_path = str(Path(tempfile.gettempdir()) / "laser_align_export.svg")
            export.write_svg(
                out_path, aligned, settings["bed_width_mm"], settings["bed_height_mm"],
                flip_y=settings["flip_y"],
            )
            _last_export_path = out_path
            _last_export_kind = "svg"
            _last_placement_info = None
            flash("Design aligned. Review below, then export.")

        elif ext in PHOTO_EXTENSIONS:
            try:
                aligned_photo = photo_align.align_photo(design_file.read(), det)
            except ValueError as e:
                flash(f"Photo alignment failed: {e}")
                return redirect(url_for("design_page"))

            out_path = str(Path(tempfile.gettempdir()) / "laser_align_export.png")
            photo_align.save_png(aligned_photo, out_path)
            _last_export_path = out_path
            _last_export_kind = "png"
            _last_placement_info = {
                "target_x_mm": round(aligned_photo.target_x_mm, 1),
                "target_y_mm": round(aligned_photo.target_y_mm, 1),
                "angle_deg": round(aligned_photo.angle_deg, 1),
                "width_mm": round(aligned_photo.width_mm, 1),
                "height_mm": round(aligned_photo.height_mm, 1),
            }
            flash(
                "Photo background removed and aligned. Rotation is already baked into "
                "the image - set your machine's job origin to the target position shown below."
            )
        else:
            flash(f"Unsupported file type '{ext}' - use .svg, .png, or .jpg.")
            return redirect(url_for("design_page"))

    return render_template(
        "design.html",
        last_export=_last_export_path is not None,
        export_kind=_last_export_kind,
        placement=_last_placement_info,
    )


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
        flash("Align a design first.")
        return redirect(url_for("design_page"))
    if _last_export_kind == "png":
        return send_file(_last_export_path, mimetype="image/png", as_attachment=True, download_name="aligned_design.png")
    return send_file(_last_export_path, mimetype="image/svg+xml", as_attachment=True, download_name="aligned_design.svg")


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
