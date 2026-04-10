"""
MDVT_gui.py — Metric Depth Video Toolbox GUI
A PySide6 GUI for converting video to side-by-side stereo 3D.

Screen 1: Project list  (~/%USERPROFILE%/mdvt_projects/)
Screen 2: Project init  (duration + import mode)
Screen 3: Project view  (scene list, player, per-scene controls, convert)
"""

from __future__ import annotations

import csv
import datetime
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

# ── Constants ────────────────────────────────────────────────────────────────

PROJECTS_DIR = Path.home() / "mdvt_projects"

DEPTH_ENGINES  = ["da3", "vda", "depthcrafter", "geometrycrafter"]
INFILL_ENGINES = ["m2svid", "normals", "stereocrafter", "stereo_dissoclusion_net", "none"]

VERSION_LABELS: List[tuple[str, str]] = [
    ("Scene (raw)",       "scene_video_file"),
    ("Depth",             "depth_video_file"),
    ("Mask",              "mask_video_file"),
    ("SBS",               "sbs"),
    ("SBS + infill mask", "sbs_infill"),
    ("Infilled (final)",  "infilled"),
]

STATUS_COLORS = {
    "done":    "#4caf50",
    "sbs":     "#2196f3",
    "depth":   "#ff9800",
    "clip":    "#ab47bc",
    "pending": "#666",
}


# ── Dark theme ───────────────────────────────────────────────────────────────

def apply_dark_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    c = QtGui.QColor
    pal.setColor(QtGui.QPalette.Window,          c(37,  37,  38))
    pal.setColor(QtGui.QPalette.WindowText,      c(220, 220, 220))
    pal.setColor(QtGui.QPalette.Base,            c(28,  28,  28))
    pal.setColor(QtGui.QPalette.AlternateBase,   c(38,  38,  40))
    pal.setColor(QtGui.QPalette.ToolTipBase,     c(255, 255, 220))
    pal.setColor(QtGui.QPalette.ToolTipText,     c(0,   0,   0))
    pal.setColor(QtGui.QPalette.Text,            c(220, 220, 220))
    pal.setColor(QtGui.QPalette.Button,          c(48,  48,  50))
    pal.setColor(QtGui.QPalette.ButtonText,      c(220, 220, 220))
    pal.setColor(QtGui.QPalette.BrightText,      c(255, 80,  80))
    pal.setColor(QtGui.QPalette.Highlight,       c(10,  132, 255))
    pal.setColor(QtGui.QPalette.HighlightedText, c(255, 255, 255))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text,       c(100, 100, 100))
    pal.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, c(100, 100, 100))
    app.setPalette(pal)
    app.setStyleSheet("""
        QToolTip { color: #000; background-color: #ffffdd; border: 1px solid #888; }
        QWidget  { font-size: 12px; }

        QPlainTextEdit, QListView, QTreeView, QTableView {
            background: #1c1c1c; border: 1px solid #3c3c3c; }

        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
            background: #2a2a2c; border: 1px solid #484848;
            padding: 3px 6px; border-radius: 4px; }

        QPushButton {
            background: #3a3a3c; border: 1px solid #555;
            padding: 6px 14px; border-radius: 6px; }
        QPushButton:hover   { background: #4a4a4e; }
        QPushButton:pressed { background: #222224; }
        QPushButton:checked { background: #0a84ff; color: #fff; border-color: #0060cc; }
        QPushButton:disabled { color: #666; border-color: #3c3c3c; }

        QLabel { color: #dcdcdc; }

        QGroupBox {
            border: 1px solid #444; border-radius: 6px;
            margin-top: 10px; padding-top: 10px; color: #aaa; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; }

        QHeaderView::section {
            background: #2e2e30; border: 1px solid #444;
            padding: 4px 6px; color: #ccc; }

        QTableWidget { gridline-color: #3c3c3c; }

        QScrollBar:vertical {
            background: #252525; width: 10px; border-radius: 5px; }
        QScrollBar::handle:vertical {
            background: #555; border-radius: 5px; min-height: 24px; }
        QScrollBar::handle:vertical:hover { background: #777; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

        QScrollBar:horizontal {
            background: #252525; height: 10px; border-radius: 5px; }
        QScrollBar::handle:horizontal {
            background: #555; border-radius: 5px; min-width: 24px; }
        QScrollBar::handle:horizontal:hover { background: #777; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

        QSplitter::handle { background: #3c3c3c; }
    """)


# ── Project helpers ──────────────────────────────────────────────────────────

def _sanitize_folder_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip() or "project"


def load_config(project_dir: Path) -> Dict[str, Any]:
    p = project_dir / "project_config.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_config(project_dir: Path, cfg: Dict[str, Any]) -> None:
    p = project_dir / "project_config.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def list_projects() -> List[Path]:
    if not PROJECTS_DIR.exists():
        return []
    return sorted(
        [d for d in PROJECTS_DIR.iterdir()
         if d.is_dir() and (d / "project_config.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )


def create_project(video_path: str) -> Path:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(video_path).stem
    base_name = _sanitize_folder_name(stem)
    proj_dir = PROJECTS_DIR / base_name
    n = 1
    while proj_dir.exists():
        proj_dir = PROJECTS_DIR / f"{base_name}_{n}"
        n += 1
    proj_dir.mkdir(parents=True)
    cfg: Dict[str, Any] = {
        "main_video_file": video_path,
        "depth_engine": "da3",
        "infill_engine": "m2svid",
        "parallel": max(1, (os.cpu_count() or 2) // 2),
        "scenes": [],
    }
    save_config(proj_dir, cfg)
    return proj_dir


def plan_paths(scene: Dict, project_dir: Path) -> Dict[str, str]:
    num  = str(scene.get("Scene Number", "0"))
    base = str(project_dir / f"scene_{num}.mkv")
    dep  = base + "_depth.mkv"
    msk  = base + "_mask.mkv"
    xfov = dep  + "_xfovs.json"
    sbs  = dep  + "_stereo.mkv"
    si   = sbs  + "_infillmask.mkv"
    inf  = sbs  + "_infilled.mkv"
    return {
        "scene_video_file": base,
        "depth_video_file": dep,
        "mask_video_file":  msk,
        "xfovs_file":       xfov,
        "sbs":              sbs,
        "sbs_infill":       si,
        "infilled":         inf,
    }


def scene_status(scene: Dict, project_dir: Path) -> str:
    p = plan_paths(scene, project_dir)
    if os.path.exists(p["infilled"]):         return "done"
    if os.path.exists(p["sbs"]):              return "sbs"
    if os.path.exists(p["depth_video_file"]): return "depth"
    if os.path.exists(p["scene_video_file"]): return "clip"
    return "pending"


def _seconds_to_timecode(s: float) -> str:
    ms = round(s * 1000)
    sc, ms = divmod(ms, 1000)
    m,  sc = divmod(sc, 60)
    h,  m  = divmod(m,  60)
    return f"{h:02d}:{m:02d}:{sc:02d}.{ms:03d}"


def get_video_info(video_path: str) -> tuple[float, float, int, int]:
    """Returns (fps, duration_seconds, width, height)."""
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return fps, frames / fps, width, height
    except Exception:
        return 25.0, 0.0, 0, 0


# ── Worker thread ────────────────────────────────────────────────────────────

class FuncWorker(QtCore.QThread):
    line = QtCore.Signal(str)
    done = QtCore.Signal(int)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn     = fn
        self.args   = args
        self.kwargs = kwargs

    def run(self):
        import subprocess as _sp
        import sys as _sys

        class _Redirect:
            def __init__(self, emit): self._emit = emit
            def write(self, s):
                if s: self._emit(s.rstrip("\n"))
            def flush(self): pass

        old_out, old_err = _sys.stdout, _sys.stderr
        _sys.stdout = _sys.stderr = _Redirect(self.line.emit)

        orig_run   = _sp.run
        orig_popen = _sp.Popen

        def _streaming_run(*args, **kwargs):
            if "stdout" in kwargs or "stderr" in kwargs:
                kwargs.setdefault("text", True)
                return orig_run(*args, **kwargs)
            env = kwargs.get("env", os.environ.copy())
            env.setdefault("PYTHONUNBUFFERED", "1")
            kwargs.update(stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, env=env)
            p = orig_popen(*args, **kwargs)
            assert p.stdout
            for ln in p.stdout:
                self.line.emit(ln.rstrip("\n"))
            rc = p.wait()
            if kwargs.get("check") and rc != 0:
                raise _sp.CalledProcessError(rc, args[0])
            return _sp.CompletedProcess(args[0], rc)

        def _streaming_popen(*args, **kwargs):
            need = "stdout" not in kwargs and "stderr" not in kwargs
            if need:
                env = kwargs.get("env", os.environ.copy())
                env.setdefault("PYTHONUNBUFFERED", "1")
                kwargs.update(stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, bufsize=1, env=env)
            p = orig_popen(*args, **kwargs)
            if need and p.stdout:
                def _reader():
                    for ln in p.stdout:
                        self.line.emit(ln.rstrip("\n"))
                threading.Thread(target=_reader, daemon=True).start()
            return p

        rc = 0
        try:
            _sp.run   = _streaming_run
            _sp.Popen = _streaming_popen
            self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.line.emit(f"ERROR: {e}")
            rc = 1
        finally:
            _sp.run   = orig_run
            _sp.Popen = orig_popen
            _sys.stdout, _sys.stderr = old_out, old_err
        self.done.emit(rc)


# ── Video player ─────────────────────────────────────────────────────────────

class VideoPlayer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            from PySide6 import QtMultimedia
            from PySide6.QtMultimediaWidgets import QVideoWidget
        except ImportError as e:
            raise RuntimeError(f"QtMultimedia not available: {e}")

        self._mm = QtMultimedia

        self.video_widget = QVideoWidget()
        self.video_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.mplayer = QtMultimedia.QMediaPlayer(self)
        self.mplayer.setVideoOutput(self.video_widget)

        self.overlay = QtWidgets.QLabel("No video")
        self.overlay.setAlignment(Qt.AlignCenter)
        self.overlay.setStyleSheet("background:#1c1c1c; color:#555; font-size:15px;")
        self.overlay.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self._stack = QtWidgets.QStackedLayout()
        self._stack.addWidget(self.video_widget)  # 0
        self._stack.addWidget(self.overlay)        # 1
        self._stack.setCurrentIndex(1)

        viewer = QtWidgets.QWidget()
        viewer.setLayout(self._stack)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self._dragging = False

        self.btn_play  = QtWidgets.QPushButton("Play")
        self.btn_play.setCheckable(True)
        self.btn_first = QtWidgets.QPushButton("⏮")
        self.btn_prev  = QtWidgets.QPushButton("◀ Frame")
        self.btn_next  = QtWidgets.QPushButton("Frame ▶")
        for btn in (self.btn_first, self.btn_prev, self.btn_next):
            btn.setFixedWidth(76)
        self.btn_play.setFixedWidth(80)

        transport = QtWidgets.QHBoxLayout()
        transport.addWidget(self.btn_first)
        transport.addWidget(self.btn_prev)
        transport.addStretch()
        transport.addWidget(self.btn_play)
        transport.addStretch()
        transport.addWidget(self.btn_next)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(viewer, 1)
        lay.addWidget(self.slider)
        lay.addLayout(transport)

        self.frame_ms = 33

        self.btn_play.toggled.connect(self._toggle_play)
        self.btn_first.clicked.connect(self._to_start)
        self.btn_prev.clicked.connect(lambda: self.step(-1))
        self.btn_next.clicked.connect(lambda: self.step(+1))
        self.mplayer.durationChanged.connect(self._on_dur)
        self.mplayer.positionChanged.connect(self._on_pos)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_dragging", True))
        self.slider.sliderReleased.connect(self._on_release)
        self.slider.valueChanged.connect(self._on_val)

    def set_source(self, path: str) -> None:
        self.mplayer.setSource(QtCore.QUrl.fromLocalFile(os.path.abspath(path)))
        self.mplayer.pause()
        self.btn_play.setChecked(False)
        self._stack.setCurrentIndex(0)

    def clear(self, msg: str = "No video") -> None:
        self.mplayer.setSource(QtCore.QUrl())
        self.overlay.setText(msg)
        self._stack.setCurrentIndex(1)

    def current_seconds(self) -> float:
        return self.mplayer.position() / 1000.0

    def _toggle_play(self, playing: bool) -> None:
        if playing:
            self.mplayer.play()
            self.btn_play.setText("Pause")
        else:
            self.mplayer.pause()
            self.btn_play.setText("Play")

    def _to_start(self) -> None:
        self.mplayer.setPosition(0)
        self.mplayer.pause()
        self.btn_play.setChecked(False)

    def step(self, n: int) -> None:
        if self.mplayer.playbackState() == self._mm.QMediaPlayer.PlayingState:
            self.mplayer.pause()
            self.btn_play.setChecked(False)
        self.mplayer.setPosition(max(0, self.mplayer.position() + n * self.frame_ms))

    def _on_dur(self, d: int) -> None:
        self.slider.setRange(0, max(0, d))

    def _on_pos(self, p: int) -> None:
        if not self._dragging:
            self.slider.blockSignals(True)
            self.slider.setValue(p)
            self.slider.blockSignals(False)

    def _on_release(self) -> None:
        self._dragging = False
        self.mplayer.setPosition(self.slider.value())

    def _on_val(self, v: int) -> None:
        if self._dragging:
            self.mplayer.setPosition(v)


# ════════════════════════════════════════════════════════════════════════════
# Screen 1 — Project List
# ════════════════════════════════════════════════════════════════════════════

class ProjectListScreen(QtWidgets.QWidget):
    project_opened  = QtCore.Signal(Path)  # open existing → Screen 3
    project_created = QtCore.Signal(Path)  # new project   → Screen 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        # Title
        title = QtWidgets.QLabel("MDVT — Projects")
        title.setStyleSheet("font-size: 24px; font-weight: bold; padding: 4px 0 12px 0;")
        title.setAlignment(Qt.AlignCenter)

        # New project button
        self.btn_new = QtWidgets.QPushButton("+ New Project")
        self.btn_new.setFixedHeight(36)
        self.btn_new.setStyleSheet(
            "background: #0a84ff; color: #fff; font-weight: bold; font-size: 13px; border-radius: 6px;")

        # Project table
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Project", "Source Video", "Last Modified"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setDefaultSectionSize(32)

        # Open button
        self.btn_open = QtWidgets.QPushButton("Open Selected Project")
        self.btn_open.setEnabled(False)
        self.btn_open.setFixedHeight(34)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.btn_new)
        top.addStretch()
        top.addWidget(self.btn_open)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(32, 24, 32, 24)
        lay.setSpacing(14)
        lay.addWidget(title)
        lay.addLayout(top)
        lay.addWidget(self.table)

        self.btn_new.clicked.connect(self._on_new)
        self.btn_open.clicked.connect(self._on_open)
        self.table.selectionModel().selectionChanged.connect(
            lambda: self.btn_open.setEnabled(bool(self.table.selectedItems())))
        self.table.doubleClicked.connect(self._on_open)

        self.refresh()

    def refresh(self) -> None:
        self.table.setRowCount(0)
        for proj in list_projects():
            cfg  = load_config(proj)
            row  = self.table.rowCount()
            self.table.insertRow(row)

            name_item = QtWidgets.QTableWidgetItem(proj.name)
            name_item.setData(Qt.UserRole, proj)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(
                cfg.get("main_video_file", "")))
            mtime = datetime.datetime.fromtimestamp(proj.stat().st_mtime)
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(
                mtime.strftime("%Y-%m-%d  %H:%M")))

    def _on_new(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Source Video", "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All Files (*)")
        if not path:
            return
        proj_dir = create_project(path)
        self.project_created.emit(proj_dir)

    def _on_open(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        item = self.table.item(rows[0].row(), 0)
        proj_dir: Path = item.data(Qt.UserRole)
        self.project_opened.emit(proj_dir)


# ════════════════════════════════════════════════════════════════════════════
# Screen 2 — Project Init
# ════════════════════════════════════════════════════════════════════════════

class ProjectInitScreen(QtWidgets.QWidget):
    import_done = QtCore.Signal(Path)  # → Screen 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_dir: Optional[Path] = None
        self._worker: Optional[FuncWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.title = QtWidgets.QLabel("Import Video")
        self.title.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.title.setAlignment(Qt.AlignCenter)

        self.lbl_video = QtWidgets.QLabel("—")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        self.lbl_video.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_video.setWordWrap(True)

        self.lbl_dur = QtWidgets.QLabel("Full video: —")
        self.lbl_dur.setAlignment(Qt.AlignCenter)
        self.lbl_dur.setStyleSheet("color: #aaa;")

        # Duration limit
        dur_grp = QtWidgets.QGroupBox("Limit conversion to (0:00 = full video)")
        dur_row = QtWidgets.QHBoxLayout(dur_grp)
        dur_row.setContentsMargins(12, 8, 12, 8)
        dur_row.addStretch()
        dur_row.addWidget(QtWidgets.QLabel("Minutes:"))
        self.spin_min = QtWidgets.QSpinBox()
        self.spin_min.setRange(0, 9999)
        self.spin_min.setFixedWidth(72)
        dur_row.addWidget(self.spin_min)
        dur_row.addWidget(QtWidgets.QLabel("Seconds:"))
        self.spin_sec = QtWidgets.QSpinBox()
        self.spin_sec.setRange(0, 59)
        self.spin_sec.setFixedWidth(60)
        dur_row.addWidget(self.spin_sec)
        dur_row.addStretch()

        # Import buttons
        self.btn_single = QtWidgets.QPushButton("Import as Single Clip")
        self.btn_single.setFixedHeight(48)
        self.btn_single.setStyleSheet("font-size: 14px;")

        self.btn_scene = QtWidgets.QPushButton("Import with Scene Detection")
        self.btn_scene.setFixedHeight(48)
        self.btn_scene.setStyleSheet(
            "font-size: 14px; font-weight: bold; "
            "background: #0a84ff; color: #fff; border-color: #0060cc;")

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addWidget(self.btn_single)
        btn_row.addWidget(self.btn_scene)

        # Console
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setFixedHeight(110)
        self.console.setPlaceholderText("Import progress will appear here…")

        card = QtWidgets.QVBoxLayout()
        card.setSpacing(14)
        card.addWidget(self.title)
        card.addWidget(self.lbl_video)
        card.addWidget(self.lbl_dur)
        card.addWidget(dur_grp)
        card.addLayout(btn_row)
        card.addWidget(self.console)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addStretch(1)
        lay.addLayout(card)
        lay.addStretch(1)
        lay.setContentsMargins(100, 40, 100, 40)

        self.btn_single.clicked.connect(lambda: self._do_import(scene_detect=False))
        self.btn_scene.clicked.connect(lambda: self._do_import(scene_detect=True))

    def load_project(self, project_dir: Path) -> None:
        self._project_dir = project_dir
        cfg = load_config(project_dir)
        video = cfg.get("main_video_file", "")
        self.title.setText(f"Import — {project_dir.name}")
        self.lbl_video.setText(video)
        self.console.clear()
        self.spin_min.setValue(0)
        self.spin_sec.setValue(0)
        self._set_busy(False)

        fps, dur, _, _ = get_video_info(video)
        if dur > 0:
            m = int(dur // 60)
            s = int(dur % 60)
            self.lbl_dur.setText(f"Full video: {m}m {s:02d}s ({dur:.1f} seconds)")
        else:
            self.lbl_dur.setText("Full video: unknown duration")

    def _set_busy(self, busy: bool) -> None:
        self.btn_single.setEnabled(not busy)
        self.btn_scene.setEnabled(not busy)

    def _log(self, msg: str) -> None:
        self.console.appendPlainText(msg)

    def _duration_limit(self) -> Optional[float]:
        total = self.spin_min.value() * 60 + self.spin_sec.value()
        return float(total) if total > 0 else None

    def _do_import(self, scene_detect: bool) -> None:
        if not self._project_dir:
            return
        cfg        = load_config(self._project_dir)
        video      = cfg.get("main_video_file", "")
        dur_limit  = self._duration_limit()
        proj_dir   = self._project_dir
        self._set_busy(True)

        def work() -> None:
            import cv2

            cap         = cv2.VideoCapture(video)
            fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_f     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            total_s     = total_f / fps

            limit_f = int(min(dur_limit * fps, total_f)) if dur_limit else total_f
            limit_s = limit_f / fps

            scenes: List[Dict] = []

            if not scene_detect:
                print("Creating single scene…")
                scenes = [{
                    "Scene Number":        "1",
                    "Start Frame":         "0",
                    "End Frame":           str(limit_f - 1),
                    "Start Time (seconds)":"0.000",
                    "End Time (seconds)":  f"{limit_s:.3f}",
                    "Start Timecode":      "00:00:00.000",
                    "End Timecode":        _seconds_to_timecode(limit_s),
                    "Length (frames)":     str(limit_f),
                    "Length (seconds)":    f"{limit_s:.3f}",
                    "Length (timecode)":   _seconds_to_timecode(limit_s),
                    "Engine":              cfg.get("depth_engine", "da3"),
                    "Infill":              cfg.get("infill_engine", "m2svid"),
                    "do_infill":           True,
                }]
            else:
                print("Running scene detection…")
                out_csv = str(proj_dir / "scenes.csv")
                subprocess.run(
                    f'scenedetect -i "{video}" -o "{proj_dir}" '
                    f'list-scenes -f scenes',
                    shell=True
                )
                # Find the output CSV (scenedetect may add suffixes)
                candidates = glob.glob(str(proj_dir / "scenes*.csv"))
                if not candidates:
                    # Also check current dir
                    stem = Path(video).stem
                    candidates = glob.glob(f"{stem}*Scenes*.csv")
                    if candidates:
                        shutil.move(candidates[0], out_csv)
                        candidates = [out_csv]
                if not candidates:
                    print("ERROR: scenedetect did not produce a CSV file.")
                    return

                csv_path = candidates[0]
                print(f"Reading scenes from: {csv_path}")

                with open(csv_path, newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader)  # skip first "Timecode List" line
                    dict_reader = csv.DictReader(f)
                    raw_rows = list(dict_reader)

                for row in raw_rows:
                    sf = int(row["Start Frame"])
                    if dur_limit and sf >= limit_f:
                        break
                    ef = min(int(row["End Frame"]), limit_f - 1)
                    lf = ef - sf + 1
                    ls = lf / fps
                    d: Dict[str, Any] = {
                        "Scene Number":         row["Scene Number"],
                        "Start Frame":          str(sf),
                        "End Frame":            str(ef),
                        "Start Time (seconds)": row["Start Time (seconds)"],
                        "End Time (seconds)":   f"{ef / fps:.3f}",
                        "Start Timecode":       row.get("Start Timecode", _seconds_to_timecode(sf / fps)),
                        "End Timecode":         _seconds_to_timecode(ef / fps),
                        "Length (frames)":      str(lf),
                        "Length (seconds)":     f"{ls:.3f}",
                        "Length (timecode)":    _seconds_to_timecode(ls),
                        "Engine":               cfg.get("depth_engine", "da3"),
                        "Infill":               cfg.get("infill_engine", "m2svid"),
                        "do_infill":            True,
                    }
                    scenes.append(d)

                print(f"Found {len(scenes)} scene(s).")

            cfg_up = load_config(proj_dir)
            cfg_up["scenes"] = scenes
            save_config(proj_dir, cfg_up)
            print(f"Saved {len(scenes)} scene(s) to project config.")

        def on_done(rc: int) -> None:
            self._set_busy(False)
            if rc == 0:
                self.import_done.emit(proj_dir)
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Import Failed",
                    "Import encountered errors — check the console output.")

        self._worker = FuncWorker(work)
        self._worker.line.connect(self._log)
        self._worker.done.connect(on_done)
        self._worker.start()


# ════════════════════════════════════════════════════════════════════════════
# Screen 3 — Project View
# ════════════════════════════════════════════════════════════════════════════

class ProjectViewScreen(QtWidgets.QWidget):
    go_home = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_dir: Optional[Path] = None
        self._cfg: Dict[str, Any]         = {}
        self._scenes: List[Dict]          = []
        self._worker: Optional[FuncWorker] = None
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # ── Top bar ──────────────────────────────────────────────────────────
        self.btn_home = QtWidgets.QPushButton("◀ Projects")
        self.btn_home.setFixedWidth(110)
        self.lbl_project = QtWidgets.QLabel("Project")
        self.lbl_project.setStyleSheet("font-size: 16px; font-weight: bold;")

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(self.btn_home)
        top_bar.addSpacing(12)
        top_bar.addWidget(self.lbl_project)
        top_bar.addStretch()

        # ── Left: scene list ─────────────────────────────────────────────────
        lbl_scenes = QtWidgets.QLabel("Scenes")
        lbl_scenes.setStyleSheet("font-weight: bold; color: #aaa; font-size: 11px;")

        self.scene_table = QtWidgets.QTableWidget(0, 5)
        self.scene_table.setHorizontalHeaderLabels(["#", "Start", "End", "Frames", "Status"])
        hh = self.scene_table.horizontalHeader()
        for col, mode in enumerate([
            QtWidgets.QHeaderView.ResizeToContents,
            QtWidgets.QHeaderView.ResizeToContents,
            QtWidgets.QHeaderView.ResizeToContents,
            QtWidgets.QHeaderView.ResizeToContents,
            QtWidgets.QHeaderView.Stretch,
        ]):
            hh.setSectionResizeMode(col, mode)
        self.scene_table.verticalHeader().setVisible(False)
        self.scene_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.scene_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.scene_table.setAlternatingRowColors(True)
        self.scene_table.setShowGrid(False)
        self.scene_table.verticalHeader().setDefaultSectionSize(28)

        left_lay = QtWidgets.QVBoxLayout()
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)
        left_lay.addWidget(lbl_scenes)
        left_lay.addWidget(self.scene_table)

        left_w = QtWidgets.QWidget()
        left_w.setLayout(left_lay)
        left_w.setMinimumWidth(260)

        # ── Right: player + per-scene controls ───────────────────────────────
        # Version selector
        self.version_combo = QtWidgets.QComboBox()
        for label, _ in VERSION_LABELS:
            self.version_combo.addItem(label)
        self.version_combo.setFixedWidth(180)

        lbl_view = QtWidgets.QLabel("View:")
        ver_row  = QtWidgets.QHBoxLayout()
        ver_row.addWidget(lbl_view)
        ver_row.addWidget(self.version_combo)
        ver_row.addStretch()

        # Player (top half of vertical splitter)
        self.player = VideoPlayer()
        self.player.video_widget.setMinimumHeight(80)
        self.player.setMinimumHeight(120)

        player_top_lay = QtWidgets.QVBoxLayout()
        player_top_lay.setContentsMargins(0, 0, 0, 0)
        player_top_lay.setSpacing(4)
        player_top_lay.addLayout(ver_row)
        player_top_lay.addWidget(self.player)

        player_top_w = QtWidgets.QWidget()
        player_top_w.setLayout(player_top_lay)

        # Scene info line
        self.lbl_scene_info = QtWidgets.QLabel("No scene selected")
        self.lbl_scene_info.setStyleSheet("color: #888; font-size: 11px;")

        # ── Per-scene settings group ──────────────────────────────────────────
        sc_grp = QtWidgets.QGroupBox("Scene Settings")
        sc_lay = QtWidgets.QGridLayout(sc_grp)
        sc_lay.setVerticalSpacing(8)
        sc_lay.setHorizontalSpacing(12)

        sc_lay.addWidget(QtWidgets.QLabel("Depth Engine:"), 0, 0)
        self.combo_depth = QtWidgets.QComboBox()
        self.combo_depth.addItems(DEPTH_ENGINES)
        sc_lay.addWidget(self.combo_depth, 0, 1)

        sc_lay.addWidget(QtWidgets.QLabel("Infill Engine:"), 0, 2)
        self.combo_infill = QtWidgets.QComboBox()
        self.combo_infill.addItems(INFILL_ENGINES)
        sc_lay.addWidget(self.combo_infill, 0, 3)

        self.chk_infill = QtWidgets.QCheckBox("Enable infill for this scene")
        self.chk_infill.setChecked(True)
        sc_lay.addWidget(self.chk_infill, 1, 0, 1, 2)

        self.btn_split = QtWidgets.QPushButton("✂  Split at Current Frame")
        sc_lay.addWidget(self.btn_split, 1, 2, 1, 2)

        # Convert-this-scene button
        self.btn_convert_scene = QtWidgets.QPushButton("▶  Convert This Scene")
        self.btn_convert_scene.setFixedHeight(34)
        self.btn_convert_scene.setStyleSheet("font-weight: bold;")

        # Controls (bottom half of vertical splitter)
        controls_lay = QtWidgets.QVBoxLayout()
        controls_lay.setContentsMargins(0, 4, 0, 0)
        controls_lay.setSpacing(6)
        controls_lay.addWidget(self.lbl_scene_info)
        controls_lay.addWidget(sc_grp)
        controls_lay.addWidget(self.btn_convert_scene)
        controls_lay.addStretch()

        controls_w = QtWidgets.QWidget()
        controls_w.setLayout(controls_lay)
        controls_w.setMinimumHeight(120)

        # Vertical splitter: player on top, controls on bottom
        v_splitter = QtWidgets.QSplitter(Qt.Vertical)
        v_splitter.addWidget(player_top_w)
        v_splitter.addWidget(controls_w)
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 0)
        v_splitter.setSizes([420, 160])
        v_splitter.setChildrenCollapsible(False)

        right_w = v_splitter

        # ── Splitter ──────────────────────────────────────────────────────────
        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])
        splitter.setChildrenCollapsible(False)

        # ── Global settings bar ───────────────────────────────────────────────
        glob_grp = QtWidgets.QGroupBox("Global Settings")
        g_lay = QtWidgets.QGridLayout(glob_grp)
        g_lay.setHorizontalSpacing(10)
        g_lay.setVerticalSpacing(6)

        g_lay.addWidget(QtWidgets.QLabel("Default Depth Engine:"), 0, 0)
        self.combo_global_depth = QtWidgets.QComboBox()
        self.combo_global_depth.addItems(DEPTH_ENGINES)
        g_lay.addWidget(self.combo_global_depth, 0, 1)

        g_lay.addWidget(QtWidgets.QLabel("Default Infill Engine:"), 0, 2)
        self.combo_global_infill = QtWidgets.QComboBox()
        self.combo_global_infill.addItems(INFILL_ENGINES)
        g_lay.addWidget(self.combo_global_infill, 0, 3)

        g_lay.addWidget(QtWidgets.QLabel("Parallel jobs:"), 1, 0)
        self.spin_parallel = QtWidgets.QSpinBox()
        self.spin_parallel.setRange(1, 64)
        self.spin_parallel.setValue(max(1, (os.cpu_count() or 2) // 2))
        self.spin_parallel.setFixedWidth(70)
        g_lay.addWidget(self.spin_parallel, 1, 1)

        self.btn_apply_global = QtWidgets.QPushButton("Apply Engines to All Scenes")
        g_lay.addWidget(self.btn_apply_global, 1, 2, 1, 2)

        # Convert-all button
        self.btn_convert_all = QtWidgets.QPushButton("▶▶  Convert All Scenes")
        self.btn_convert_all.setFixedHeight(44)
        self.btn_convert_all.setFixedWidth(220)
        self.btn_convert_all.setStyleSheet(
            "background: #0a84ff; color: #fff; "
            "font-size: 14px; font-weight: bold; border-color: #0060cc;")

        bottom_bar = QtWidgets.QHBoxLayout()
        bottom_bar.addWidget(glob_grp, 1)
        bottom_bar.addSpacing(8)
        bottom_bar.addWidget(self.btn_convert_all, 0, Qt.AlignBottom)

        # ── Console ───────────────────────────────────────────────────────────
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        self.console.setPlaceholderText("Console output…")

        # ── Master layout ─────────────────────────────────────────────────────
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        lay.addLayout(top_bar)
        lay.addWidget(splitter, 1)
        lay.addLayout(bottom_bar)
        lay.addWidget(self.console)

        # ── Wire signals ──────────────────────────────────────────────────────
        self.btn_home.clicked.connect(self.go_home)
        self.scene_table.currentCellChanged.connect(lambda row, *_: self._on_scene_selected(row))
        self.version_combo.currentIndexChanged.connect(self._update_player)
        self.combo_depth.currentTextChanged.connect(self._on_depth_changed)
        self.combo_infill.currentTextChanged.connect(self._on_infill_changed)
        self.chk_infill.toggled.connect(self._on_infill_toggled)
        self.btn_split.clicked.connect(self._on_split)
        self.btn_convert_scene.clicked.connect(self._on_convert_scene)
        self.btn_convert_all.clicked.connect(self._on_convert_all)
        self.btn_apply_global.clicked.connect(self._on_apply_global)
        self.combo_global_depth.currentTextChanged.connect(self._save_global)
        self.combo_global_infill.currentTextChanged.connect(self._save_global)
        self.spin_parallel.valueChanged.connect(self._save_global)

    # ── Loading ───────────────────────────────────────────────────────────────

    def load_project(self, project_dir: Path) -> None:
        self._project_dir = project_dir
        self._reload()

    def _reload(self, select_row: int = 0) -> None:
        self._cfg    = load_config(self._project_dir)
        self._scenes = self._cfg.get("scenes", [])

        self.lbl_project.setText(self._project_dir.name)

        # Global controls (no signals while loading)
        for widget in (self.combo_global_depth, self.combo_global_infill, self.spin_parallel):
            widget.blockSignals(True)
        self.combo_global_depth.setCurrentText(self._cfg.get("depth_engine", "da3"))
        self.combo_global_infill.setCurrentText(self._cfg.get("infill_engine", "m2svid"))
        self.spin_parallel.setValue(self._cfg.get("parallel", 2))
        for widget in (self.combo_global_depth, self.combo_global_infill, self.spin_parallel):
            widget.blockSignals(False)

        # Scene table
        self.scene_table.blockSignals(True)
        self.scene_table.setRowCount(0)
        for s in self._scenes:
            r = self.scene_table.rowCount()
            self.scene_table.insertRow(r)
            self.scene_table.setItem(r, 0, self._cell(s.get("Scene Number", str(r + 1))))
            self.scene_table.setItem(r, 1, self._cell(s.get("Start Timecode", "")))
            self.scene_table.setItem(r, 2, self._cell(s.get("End Timecode", "")))
            self.scene_table.setItem(r, 3, self._cell(s.get("Length (frames)", "")))

            status = scene_status(s, self._project_dir)
            st_item = QtWidgets.QTableWidgetItem(status)
            st_item.setForeground(QtGui.QColor(STATUS_COLORS.get(status, "#888")))
            self.scene_table.setItem(r, 4, st_item)
        self.scene_table.blockSignals(False)

        if self._scenes:
            self.scene_table.setCurrentCell(min(select_row, len(self._scenes) - 1), 0)
        else:
            self.player.clear("No scenes — import video first")

    @staticmethod
    def _cell(text: str) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        return item

    # ── Scene selection ───────────────────────────────────────────────────────

    def _current_scene(self) -> Optional[Dict]:
        row = self.scene_table.currentRow()
        if 0 <= row < len(self._scenes):
            return self._scenes[row]
        return None

    def _on_scene_selected(self, row: int) -> None:
        s = self._scenes[row] if 0 <= row < len(self._scenes) else None
        if s is None:
            self.player.clear()
            self.lbl_scene_info.setText("No scene selected")
            return

        # Update per-scene controls without triggering saves
        for w in (self.combo_depth, self.combo_infill, self.chk_infill):
            w.blockSignals(True)
        self.combo_depth.setCurrentText(s.get("Engine", self._cfg.get("depth_engine", "da3")))
        self.combo_infill.setCurrentText(s.get("Infill", self._cfg.get("infill_engine", "m2svid")))
        self.chk_infill.setChecked(s.get("do_infill", True))
        for w in (self.combo_depth, self.combo_infill, self.chk_infill):
            w.blockSignals(False)

        status = scene_status(s, self._project_dir)
        self.lbl_scene_info.setText(
            f"#{s.get('Scene Number','')}   "
            f"{s.get('Start Timecode','')} → {s.get('End Timecode','')}   "
            f"| {s.get('Length (frames)','')} frames   | {status}"
        )
        self._update_player()

    def _update_player(self) -> None:
        s = self._current_scene()
        if s is None:
            self.player.clear("No scene selected")
            return
        paths = plan_paths(s, self._project_dir)
        idx   = self.version_combo.currentIndex()
        label, key = VERSION_LABELS[idx]
        path  = paths.get(key, "")
        if path and os.path.exists(path):
            self.player.set_source(path)
        else:
            self.player.clear(f"{label} not generated yet")

    # ── Per-scene settings ────────────────────────────────────────────────────

    def _on_depth_changed(self, eng: str) -> None:
        s = self._current_scene()
        if s: s["Engine"] = eng; self._save_scenes()

    def _on_infill_changed(self, eng: str) -> None:
        s = self._current_scene()
        if s: s["Infill"] = eng; self._save_scenes()

    def _on_infill_toggled(self, checked: bool) -> None:
        s = self._current_scene()
        if s: s["do_infill"] = checked; self._save_scenes()

    def _save_scenes(self) -> None:
        self._cfg["scenes"] = self._scenes
        save_config(self._project_dir, self._cfg)

    def _save_global(self) -> None:
        self._cfg["depth_engine"]  = self.combo_global_depth.currentText()
        self._cfg["infill_engine"] = self.combo_global_infill.currentText()
        self._cfg["parallel"]      = self.spin_parallel.value()
        save_config(self._project_dir, self._cfg)

    def _on_apply_global(self) -> None:
        gd = self.combo_global_depth.currentText()
        gi = self.combo_global_infill.currentText()
        for s in self._scenes:
            s["Engine"] = gd
            s["Infill"] = gi
        self._save_scenes()
        self._log(f"Applied depth={gd}, infill={gi} to all {len(self._scenes)} scenes.")
        self._reload(select_row=self.scene_table.currentRow())

    # ── Split scene ───────────────────────────────────────────────────────────

    def _on_split(self) -> None:
        row = self.scene_table.currentRow()
        if row < 0 or row >= len(self._scenes):
            return
        s  = self._scenes[row]
        sf = int(s["Start Frame"]);           ef = int(s["End Frame"])
        ss = float(s["Start Time (seconds)"]); es = float(s["End Time (seconds)"])
        spf = (es - ss) / (ef - sf) if ef != sf else 0.0

        cur = self.player.current_seconds()
        cur = min(max(cur, ss + 1e-6), es - 1e-6)
        split_f = int(sf + (cur - ss) / spf) if spf > 0 else (sf + ef) // 2

        if split_f <= sf or split_f >= ef:
            self._log("Split ignored — cursor is too close to scene edges.")
            return

        def chunk(csf: int, cef: int) -> Dict:
            css = ss + (csf - sf) * spf
            ces = ss + (cef - sf) * spf
            lf  = cef - csf + 1
            d   = s.copy()
            d.update(
                {"Start Frame":          str(csf),
                 "End Frame":            str(cef),
                 "Start Time (seconds)": f"{css:.3f}",
                 "End Time (seconds)":   f"{ces:.3f}",
                 "Start Timecode":       _seconds_to_timecode(css),
                 "End Timecode":         _seconds_to_timecode(ces),
                 "Length (frames)":      str(lf),
                 "Length (seconds)":     f"{max(0.0, ces - css):.3f}",
                 "Length (timecode)":    _seconds_to_timecode(max(0.0, ces - css))})
            return d

        self._scenes[row] = chunk(sf, split_f)
        self._scenes.insert(row + 1, chunk(split_f + 1, ef))

        for i, sc in enumerate(self._scenes):
            sc["Scene Number"] = str(i + 1)

        self._save_scenes()
        self._log(f"Split scene #{s['Scene Number']} at frame {split_f}.")
        self._reload(select_row=row)

    # ── Conversion ────────────────────────────────────────────────────────────

    def _busy(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def _build_args(self, scene: Dict) -> Any:
        """Build lightweight args namespace for pipeline functions."""
        class _Args: pass
        a = _Args()
        a.color_video      = self._cfg["main_video_file"]
        a.output_dir       = str(self._project_dir)
        a.parallel         = self._cfg.get("parallel", 2)
        a.no_render        = False
        a.max_scene_frames = 999999

        infill_eng = scene.get("Infill", self._cfg.get("infill_engine", "m2svid"))
        do_infill  = scene.get("do_infill", True)
        # When infill is disabled per-scene, treat as 'none' so step6 is skipped
        a.infill_engine = infill_eng if do_infill else "none"
        return a

    def _on_convert_scene(self) -> None:
        if self._busy():
            QtWidgets.QMessageBox.information(self, "Busy", "A job is already running.")
            return
        row = self.scene_table.currentRow()
        if row < 0:
            return

        scene_snap   = self._scenes[row].copy()
        project_dir  = self._project_dir
        pipeline_path = str(Path(__file__).parent / "movie_2_3D.py")
        a = self._build_args(scene_snap)

        def work() -> None:
            import importlib.util, cv2
            spec = importlib.util.spec_from_file_location("_pipeline", pipeline_path)
            pl   = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pl)  # type: ignore

            sd = scene_snap.copy()
            sd.update(plan_paths(sd, project_dir))
            sd["infill"]   = sd.get("do_infill", True)
            sd["finished"] = os.path.exists(sd["sbs"]) or os.path.exists(sd["infilled"])

            cap = cv2.VideoCapture(a.color_video)
            fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fr  = cap.get(cv2.CAP_PROP_FPS)

            # Seek to the scene's start frame so step1 reads the right frames
            start_frame = int(sd.get("Start Frame", 0))
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            pl.step1_create_scene_videos(cap, [sd], fr, fw, fh)
            pl.step2_estimate_depth(a, [sd])
            pl.step3_generate_masks(a, [sd])
            pl.step4_find_convergence([sd])
            pl.step5_render_sbs(a, [sd])
            if a.infill_engine != "none":
                pl.step6_infill_and_collect(a, [sd])

        self._run_worker(work, f"Converting scene #{scene_snap.get('Scene Number', row + 1)}")

    def _on_convert_all(self) -> None:
        if self._busy():
            QtWidgets.QMessageBox.information(self, "Busy", "A job is already running.")
            return
        if not self._scenes:
            return

        scenes_snap   = [s.copy() for s in self._scenes]
        project_dir   = self._project_dir
        cfg_snap      = self._cfg.copy()
        pipeline_path = str(Path(__file__).parent / "movie_2_3D.py")

        # Pre-compute per-scene args outside the thread (reads from self is fine here)
        per_scene_args = [self._build_args(s) for s in scenes_snap]

        def work() -> None:
            import importlib.util, cv2
            spec = importlib.util.spec_from_file_location("_pipeline", pipeline_path)
            pl   = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(pl)  # type: ignore

            cap = cv2.VideoCapture(cfg_snap["main_video_file"])
            fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fr  = cap.get(cv2.CAP_PROP_FPS)

            # Build full scene list with file paths
            full_scenes = []
            for s in scenes_snap:
                sd = s.copy()
                sd.update(plan_paths(sd, project_dir))
                sd["infill"]   = sd.get("do_infill", True)
                sd["finished"] = os.path.exists(sd["sbs"]) or os.path.exists(sd["infilled"])
                full_scenes.append(sd)

            # Step 1: create all clips in one pass (efficient — sequential read)
            pl.step1_create_scene_videos(cap, full_scenes, fr, fw, fh)

            # Steps 2–6: per scene (respects per-scene engine settings)
            for sd, a in zip(full_scenes, per_scene_args):
                num = sd.get("Scene Number", "?")
                print(f"\n── Scene {num} ──")
                pl.step2_estimate_depth(a, [sd])
                pl.step3_generate_masks(a, [sd])
                pl.step4_find_convergence([sd])
                pl.step5_render_sbs(a, [sd])
                if a.infill_engine != "none":
                    pl.step6_infill_and_collect(a, [sd])

            # Step 7: concat + mux
            video_files = []
            for sd in full_scenes:
                if os.path.exists(sd["infilled"]):
                    video_files.append(sd["infilled"])
                elif os.path.exists(sd["sbs"]):
                    video_files.append(sd["sbs"])

            if video_files:
                class _A7: pass
                a7 = _A7()
                a7.color_video = cfg_snap["main_video_file"]
                a7.output_dir  = str(project_dir)
                pl.step7_concat_and_mux(a7, video_files)
            else:
                print("No output scenes to concatenate.")

        self._run_worker(work, "Converting all scenes")

    def _run_worker(self, fn, title: str) -> None:
        self._log(f"▶  {title}")
        self._worker = FuncWorker(fn)
        self._worker.line.connect(self._log)
        self._worker.done.connect(self._on_job_done)
        self._worker.start()
        # Disable convert buttons while running
        self.btn_convert_scene.setEnabled(False)
        self.btn_convert_all.setEnabled(False)

    def _on_job_done(self, rc: int) -> None:
        self.btn_convert_scene.setEnabled(True)
        self.btn_convert_all.setEnabled(True)
        sel = self.scene_table.currentRow()
        self._log(f"◆  Finished (exit code {rc})")
        self._reload(select_row=max(0, sel))

    def _log(self, msg: str) -> None:
        self.console.appendPlainText(msg)
        self.console.verticalScrollBar().setValue(
            self.console.verticalScrollBar().maximum())


# ════════════════════════════════════════════════════════════════════════════
# Main window — stacked navigation
# ════════════════════════════════════════════════════════════════════════════

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MDVT — Metric Depth Video Toolbox")
        self.resize(1300, 840)
        self.setMinimumSize(900, 600)

        self.stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.stack)

        self.screen_list = ProjectListScreen()
        self.screen_init = ProjectInitScreen()
        self.screen_view = ProjectViewScreen()

        self.stack.addWidget(self.screen_list)  # index 0
        self.stack.addWidget(self.screen_init)  # index 1
        self.stack.addWidget(self.screen_view)  # index 2

        self.screen_list.project_opened.connect(self._open_project)
        self.screen_list.project_created.connect(self._new_project)
        self.screen_init.import_done.connect(self._open_project)
        self.screen_view.go_home.connect(self._go_home)

        self.stack.setCurrentIndex(0)

    def _open_project(self, proj_dir: Path) -> None:
        self.screen_view.load_project(proj_dir)
        self.stack.setCurrentIndex(2)

    def _new_project(self, proj_dir: Path) -> None:
        self.screen_init.load_project(proj_dir)
        self.stack.setCurrentIndex(1)

    def _go_home(self) -> None:
        self.screen_list.refresh()
        self.stack.setCurrentIndex(0)


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication(sys.argv)
    apply_dark_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
