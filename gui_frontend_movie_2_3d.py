"""
gui_frontend.py — Optional PySide6 GUI for metric_depth_video_toolbox

How to hook up:
1) In your main script (e.g., movie_2_3D.py), add:
   parser.add_argument('--gui', action='store_true',
                       help='Launch the PySide6 GUI (optional)')

2) In main():
   if args.gui:
       from gui_frontend import run_gui
       run_gui(args, main_script_path=__file__)
       return

Run:
   python movie_2_3D.py --gui
"""

from __future__ import annotations

import os
import sys
import csv
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ----------------------------
# CSV/scene helpers (no Qt)
# ----------------------------

@dataclass
class SceneRow:
    data: Dict[str, str]
    paths: Dict[str, str] = field(default_factory=dict)

    @property
    def number(self) -> int:
        return int(self.data['Scene Number'])

    @number.setter
    def number(self, v: int) -> None:
        self.data['Scene Number'] = str(v)

    def len_frames(self) -> int:
        return int(self.data['Length (frames)'])

    def start_tc(self) -> str:
        return self.data.get('Start Timecode', '')

    def end_tc(self) -> str:
        return self.data.get('End Timecode', '')

    def engine(self) -> str:
        return self.data.get('Engine', '') or 'da3'

    def set_engine(self, eng: str) -> None:
        self.data['Engine'] = eng

    def display_name(self) -> str:
        return f"#{self.number:03d}  {self.start_tc()} → {self.end_tc()}  ({self.len_frames()} fr)"


def _plan_paths_for_scene(scene: Dict[str, str], output_dir: str) -> Dict[str, str]:
    num = str(scene['Scene Number'])
    base = os.path.join(output_dir, f'scene_{num}.mkv')
    depth = base + "_depth.mkv"
    mask = base + "_mask.mkv"
    xfovs = depth + "_xfovs.json"
    sbs = depth + "_stereo.mkv"
    sbs_infill = sbs + "_infillmask.mkv"
    infilled = sbs + "_infilled.mkv"
    return {
        'scene_video_file': base,
        'depth_video_file': depth,
        'mask_video_file': mask,
        'xfovs_file': xfovs,
        'sbs': sbs,
        'sbs_infill': sbs_infill,
        'infilled': infilled,
    }


def _write_full_scene_csv(scene_csv_path: str, rows: List[SceneRow], csv_delimiter: str) -> None:
    fieldnames = [
        'Scene Number','Start Time (seconds)','Start Timecode','Start Frame',
        'End Time (seconds)','End Timecode','End Frame', 'Length (frames)',
        'Length (seconds)','Length (timecode)', 'Engine','Infill'
    ]
    with open(scene_csv_path, 'w', newline='') as f:
        f.write('Timecode List\n')   # first line ignored by loader
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=csv_delimiter)
        writer.writeheader()
        for r in rows:
            out = {k: r.data.get(k, '') for k in fieldnames}
            writer.writerow(out)


def _renumber_and_rename(scene_rows: List[SceneRow], start_index: int, output_dir: str) -> None:
    old_to_new: Dict[int, int] = {}
    for i in range(start_index, len(scene_rows)):
        old = scene_rows[i].number
        new = i + 1
        if new != old:
            old_to_new[old] = new
        scene_rows[i].number = new

    def _paths(num: int) -> Dict[str, str]:
        d = {'Scene Number': str(num)}
        return _plan_paths_for_scene(d, output_dir)

    for old in sorted(old_to_new.keys(), reverse=True):
        new = old_to_new[old]
        oldp = _paths(old)
        newp = _paths(new)
        for key in oldp:
            src = oldp[key]
            dst = newp[key]
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)


def _split_scene_at_frame(scene_rows: List[SceneRow], idx: int, split_frame: int) -> bool:
    """Split scene at an absolute frame index (split_frame belongs to the FIRST part).
       Returns True if a split happened, False if split was invalid (near edges)."""
    s = scene_rows[idx].data
    sf = int(s['Start Frame']); ef = int(s['End Frame'])
    if split_frame <= sf or split_frame >= ef:
        return False  # nothing to do (avoid empty halves)

    ss = float(s['Start Time (seconds)']); es = float(s['End Time (seconds)'])

    def tc(seconds: float) -> str:
        ms = round(seconds * 1000)
        sec, ms = divmod(ms, 1000)
        m, sec = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"

    spf = (es - ss) / (ef - sf) if ef != sf else 0.0

    def make_chunk(chunk_sf: int, chunk_ef: int) -> Dict[str, str]:
        chunk_ss = ss + (chunk_sf - sf) * spf
        chunk_es = ss + (chunk_ef - sf) * spf
        d = s.copy()
        d['Start Frame'] = str(chunk_sf)
        d['Start Time (seconds)'] = f"{chunk_ss:.3f}"
        d['Start Timecode'] = tc(chunk_ss)
        d['End Frame'] = str(chunk_ef)
        d['End Time (seconds)'] = f"{chunk_es:.3f}"
        d['End Timecode'] = tc(chunk_es)
        length_frames = chunk_ef - chunk_sf + 1
        d['Length (frames)'] = str(length_frames)
        d['Length (seconds)'] = f"{max(0.0, chunk_es - chunk_ss):.3f}"
        d['Length (timecode)'] = tc(max(0.0, chunk_es - chunk_ss))
        return d

    first = SceneRow(make_chunk(sf, split_frame))
    second = SceneRow(make_chunk(split_frame + 1, ef))
    scene_rows[idx] = first
    scene_rows.insert(idx + 1, second)
    return True


# ----------------------------
# GUI (PySide6 imported lazily)
# ----------------------------

def run_gui(cli_args, main_script_path: str) -> None:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt

    app = QtWidgets.QApplication(sys.argv)

    # --- Dark theme (Fusion palette) ---
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(37, 37, 38))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 48))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtGui.QColor(255, 255, 220))
    palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor(0, 0, 0))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 45, 48))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.BrightText, QtGui.QColor(255, 0, 0))
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(10, 132, 255))
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
    app.setPalette(palette)
    app.setStyleSheet(
        "QToolTip { color: #000; background-color: #ffffdd; border: 1px solid #666; }"
        "QWidget { font-size: 12px; }"
        "QPlainTextEdit, QListView, QTreeView, QTableView { background: #1e1e1e; border: 1px solid #3c3c3c; }"
        "QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { background: #2a2a2a; border: 1px solid #3c3c3c; }"
        "QPushButton { background: #2f2f2f; border: 1px solid #3c3c3c; padding: 6px 10px; border-radius: 6px; }"
        "QPushButton:checked { background: #3a3a3a; }"
        "QLabel { color: #ddd; }"
    )

    # Multimedia imports
    try:
        from PySide6 import QtMultimedia
        from PySide6.QtMultimediaWidgets import QVideoWidget
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            None,
            'QtMultimedia not available',
            f"{e}\nInstall PySide6 multimedia components (pip install PySide6) and OS media codecs."
        )
        return

    # Import pipeline (the caller script) so we can reuse helpers
    import importlib.util
    spec = importlib.util.spec_from_file_location("_pipeline_mod", main_script_path)
    pipeline = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(pipeline)  # type: ignore

    # ---------- Worker that runs Python functions and streams prints ----------
    class FuncWorker(QtCore.QThread):
        line = QtCore.Signal(str)
        done = QtCore.Signal(int)
        def __init__(self, fn, *args, **kwargs):
            super().__init__()
            self.fn = fn
            self.args = args
            self.kwargs = kwargs
        def run(self):
            import subprocess, os, threading, sys as _sys

            # Redirect Python prints to the GUI console
            class _Redirect:
                def __init__(self, emit): self.emit = emit
                def write(self, s):
                    if s:
                        self.emit(s.rstrip("\n"))
                def flush(self):
                    pass

            old_out, old_err = _sys.stdout, _sys.stderr
            _sys.stdout = _sys.stderr = _Redirect(self.line.emit)

            # Monkey-patch subprocess.run and subprocess.Popen to live-stream output
            orig_run   = subprocess.run
            orig_popen = subprocess.Popen

            def streaming_run(*args, **kwargs):
                text = kwargs.pop('text', True)
                if 'stdout' in kwargs or 'stderr' in kwargs:
                    kwargs['text'] = text
                    return orig_run(*args, **kwargs)
                kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                env = kwargs.get('env', os.environ.copy())
                env.setdefault('PYTHONUNBUFFERED', '1')
                kwargs['env'] = env
                p = orig_popen(*args, **kwargs)
                assert p.stdout is not None
                for line in p.stdout:
                    self.line.emit(line.rstrip('\n'))
                rc = p.wait()
                if kwargs.get('check') and rc != 0:
                    raise subprocess.CalledProcessError(rc, args[0])
                return subprocess.CompletedProcess(args=args[0], returncode=rc)

            def streaming_popen(*args, **kwargs):
                need_stream = ('stdout' not in kwargs and 'stderr' not in kwargs)
                if need_stream:
                    kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    env = kwargs.get('env', os.environ.copy())
                    env.setdefault('PYTHONUNBUFFERED', '1')
                    kwargs['env'] = env
                p = orig_popen(*args, **kwargs)
                if need_stream and p.stdout:
                    def reader():
                        for line in p.stdout:
                            self.line.emit(line.rstrip('\n'))
                    threading.Thread(target=reader, daemon=True).start()
                return p

            rc = 0
            try:
                subprocess.run   = streaming_run   # stream run()
                subprocess.Popen = streaming_popen # stream Popen()
                self.fn(*self.args, **self.kwargs)
            except Exception as e:
                self.line.emit(f"ERROR: {e}")
                rc = 1
            finally:
                subprocess.run   = orig_run
                subprocess.Popen = orig_popen
                _sys.stdout, _sys.stderr = old_out, old_err
            self.done.emit(rc)


    # ---------- Video player ----------
    class VideoPlayer(QtWidgets.QWidget):
        """QMediaPlayer + stacked viewer: either the video or an overlay message."""
        def __init__(self):
            super().__init__()
            # Video widget
            self.video_widget = QVideoWidget()
            self.video_widget.setMinimumHeight(420)
            self.video_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.mplayer = QtMultimedia.QMediaPlayer(self)
            self.mplayer.setVideoOutput(self.video_widget)

            # Overlay message (separate page)
            self.overlay_label = QtWidgets.QLabel("No video file exists for this scene yet")
            self.overlay_label.setAlignment(Qt.AlignCenter)
            self.overlay_label.setWordWrap(True)
            self.overlay_label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
            self.overlay_label.setStyleSheet(
                "background-color: #1e1e1e; color: #ddd; font-size: 16px; padding: 16px;"
            )

            # Stacked container to switch between video and overlay
            self.viewer_stack = QtWidgets.QStackedLayout()
            self.viewer_stack.addWidget(self.video_widget)      # index 0
            self.viewer_stack.addWidget(self.overlay_label)     # index 1
            self.viewer_container = QtWidgets.QWidget()
            self.viewer_container.setLayout(self.viewer_stack)
            self.viewer_stack.setCurrentIndex(1)  # start with overlay until a video is set

            self.frame_ms = 33  # ~30fps fallback
            self.frame_ms = 33  # ~30fps fallback

            # Scrub slider (milliseconds)
            self.slider = QtWidgets.QSlider(Qt.Horizontal)
            self.slider.setRange(0, 0)
            self.slider.setSingleStep(33)
            self.slider.setPageStep(1000)
            self._slider_user_drag = False

            # Transport
            self.btn_play = QtWidgets.QPushButton('Play')
            self.btn_play.setCheckable(True)
            self.btn_stop = QtWidgets.QPushButton('⏮ First')
            self.btn_prev = QtWidgets.QPushButton('◀︎ Frame')
            self.btn_next = QtWidgets.QPushButton('Frame ▶︎')

            ctr = QtWidgets.QHBoxLayout()
            ctr.addWidget(self.btn_stop)
            ctr.addWidget(self.btn_prev)
            ctr.addStretch(1)
            ctr.addWidget(self.btn_play)
            ctr.addStretch(1)
            ctr.addWidget(self.btn_next)

            lay = QtWidgets.QVBoxLayout(self)
            lay.addWidget(self.viewer_container)
            lay.addWidget(self.slider)
            lay.addLayout(ctr)

            # Signals
            self.btn_play.toggled.connect(self._toggle_play)
            self.btn_stop.clicked.connect(self._to_start)
            self.btn_prev.clicked.connect(lambda: self.step(-1))
            self.btn_next.clicked.connect(lambda: self.step(+1))

            self.mplayer.durationChanged.connect(self._on_duration_changed)
            self.mplayer.positionChanged.connect(self._on_position_changed)
            self.slider.sliderPressed.connect(self._on_slider_pressed)
            self.slider.sliderReleased.connect(self._on_slider_released)
            self.slider.valueChanged.connect(self._on_slider_value_changed)

        def set_source(self, path: str):
            url = QtCore.QUrl.fromLocalFile(os.path.abspath(path))
            self.mplayer.setSource(url)
            self.mplayer.pause()
            self.btn_play.setChecked(False)
            self.slider.setRange(0, 0)
            self.slider.setValue(0)
            # show the video page when a valid source is set
            self.viewer_stack.setCurrentIndex(0)

        def _toggle_play(self, playing: bool):
            if playing:
                self.mplayer.play()
                self.btn_play.setText('Pause')
            else:
                self.mplayer.pause()
                self.btn_play.setText('Play')

        def _to_start(self):
            self.mplayer.pause()
            self.mplayer.setPosition(0)
            self.btn_play.setChecked(False)

        def step(self, n: int):
            if self.mplayer.playbackState() == QtMultimedia.QMediaPlayer.PlayingState:
                self.mplayer.pause()
                self.btn_play.setChecked(False)
            pos = self.mplayer.position()
            new_pos = max(0, pos + n * self.frame_ms)
            self.mplayer.setPosition(new_pos)

        def _on_duration_changed(self, dur_ms: int):
            self.slider.setRange(0, max(0, int(dur_ms)))
        def _on_position_changed(self, pos_ms: int):
            if not self._slider_user_drag:
                self.slider.blockSignals(True)
                self.slider.setValue(int(pos_ms))
                self.slider.blockSignals(False)
        def _on_slider_pressed(self):
            self._slider_user_drag = True
        def _on_slider_released(self):
            self._slider_user_drag = False
            self.mplayer.setPosition(self.slider.value())
        def _on_slider_value_changed(self, v: int):
            if self._slider_user_drag:
                self.mplayer.setPosition(v)

        def current_seconds(self) -> float:
            return self.mplayer.position() / 1000.0

    # ---------- Main Window ----------
    class MainWindow(QtWidgets.QMainWindow):
        def __init__(self, args):
            super().__init__()
            self.setWindowTitle('Metric Depth Video Toolbox — GUI')
            self.resize(1320, 820)
            self.args = args
            self.scene_csv: Optional[str] = getattr(args, 'scene_file', None)
            self.csv_delimiter: str = getattr(args, 'csv_delimiter', ',')
            self.output_dir: str = getattr(args, 'output_dir', 'output')
            self.color_video: Optional[str] = getattr(args, 'color_video', None)
            self.end_scene: int = getattr(args, 'end_scene', -1)
            self.process_worker: Optional[FuncWorker] = None
            # Track previous engine per selected row to allow cancel/revert
            self._last_engine_by_row: Dict[int, str] = {}
            # Remember selection across jobs so UI doesn't jump to row 0
            self._selected_row_before_job: Optional[int] = None
            # Keep full scene list (all scenes), and a visible subset honoring end_scene
            self.scenes_all: List[SceneRow] = []

            # Widgets
            self.scene_list = QtWidgets.QListWidget()
            self.scene_list.setMinimumWidth(320)
            self.player = VideoPlayer()

            # Top toolbar row ABOVE the video player
            self.btn_process_all = QtWidgets.QPushButton('Process All Scenes')

            # Right-side controls
            self.version_combo = QtWidgets.QComboBox()
            self.engine_combo = QtWidgets.QComboBox()
            self.engine_combo.addItems(['da3', 'vda', 'depthcrafter', 'geometrycrafter'])
            self.btn_split = QtWidgets.QPushButton('Split scene before current frame')
            self.btn_convert_scene = QtWidgets.QPushButton('Convert This Scene')

            # Console
            self.console = QtWidgets.QPlainTextEdit()
            self.console.setReadOnly(True)

            # Left panel (scene list)
            left = QtWidgets.QVBoxLayout()
            lbl_left = QtWidgets.QLabel('Scenes')
            lbl_left.setStyleSheet('font-size: 14px; font-weight: 600;')
            left.addWidget(lbl_left)
            left.addWidget(self.scene_list, 1)
            leftw = QtWidgets.QWidget(); leftw.setLayout(left)
            leftw.setMinimumWidth(320)            # Controls row: actions on the left, dropdowns on the right
            form = QtWidgets.QGridLayout()
            rr = 0
            form.addWidget(QtWidgets.QLabel('Version to view:'), rr, 0)
            form.addWidget(self.version_combo, rr, 1)
            rr += 1
            form.addWidget(QtWidgets.QLabel('Engine:'), rr, 0)
            form.addWidget(self.engine_combo, rr, 1)

            actions = QtWidgets.QHBoxLayout()
            actions.addWidget(self.btn_split)
            actions.addWidget(self.btn_convert_scene)

            actionsw = QtWidgets.QWidget(); actionsw.setLayout(actions)
            formw = QtWidgets.QWidget(); formw.setLayout(form)

            controls_row = QtWidgets.QHBoxLayout()
            controls_row.addWidget(actionsw)
            controls_row.addStretch(1)
            controls_row.addWidget(formw)

            # Right column layout
            right = QtWidgets.QVBoxLayout()

            # Toolbar row (Process All Scenes) ABOVE player
            tools = QtWidgets.QHBoxLayout()
            tools.addWidget(self.btn_process_all)
            tools.addStretch(1)

            right.addLayout(tools)
            right.addWidget(self.player, 10)
            right.addLayout(controls_row)
            right.addWidget(QtWidgets.QLabel('Console output'))
            right.addWidget(self.console, 6)
            rightw = QtWidgets.QWidget(); rightw.setLayout(right)

            # Splitter
            splitter = QtWidgets.QSplitter()
            splitter.setOrientation(Qt.Horizontal)
            splitter.addWidget(leftw)
            splitter.addWidget(rightw)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([360, 960])
            splitter.setChildrenCollapsible(False)
            self.setCentralWidget(splitter)

            # Signals
            self.scene_list.currentRowChanged.connect(self._on_select_scene)
            self.version_combo.currentIndexChanged.connect(self._update_player_source)
            self.engine_combo.currentTextChanged.connect(self._on_engine_changed)
            self.btn_split.clicked.connect(self._on_split_clicked)
            self.btn_convert_scene.clicked.connect(self._on_convert_scene)
            self.btn_process_all.clicked.connect(self._on_process_all)

            self._bootstrap()

        # ---- bootstrap & data ----
        def _bootstrap(self):
            # Pick video if not provided
            if not self.color_video:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Pick a video file')
                if not path:
                    QtWidgets.QMessageBox.warning(self, 'No file', 'No video selected. Exiting.')
                    QtWidgets.QApplication.quit(); return
                self.color_video = path

            os.makedirs(self.output_dir, exist_ok=True)
            self.args.color_video = self.color_video
            self.args.output_dir = self.output_dir

            # Ensure scene file exists (mutates args.scene_file)
            try:
                pipeline.ensure_scene_file(self.args)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, 'Scene detect failed', str(e))
                return
            self.scene_csv = self.args.scene_file

            # Ask for end_scene if not given
            if self.end_scene is None or int(self.end_scene) < 0:
                val, ok = QtWidgets.QInputDialog.getInt(self, 'End scene', 'Stop after scene # (0 = all):', 0, 0)
                if ok:
                    self.end_scene = -1 if val == 0 else val
                    self.args.end_scene = self.end_scene

            # Load scenes and plan paths
            self._load_scenes()

            # If any scene clip is missing, ask to pre-create them now (up to end_scene)
            missing_any = any(not os.path.exists(s.paths['scene_video_file']) for s in self.scenes)
            if missing_any:
                resp = QtWidgets.QMessageBox.question(
                    self,
                    'Create per-scene clips?',
                    'The original video will be split into per-scene files. \nDo you want to create the per-scene clip files now (up to your End Scene setting)?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes
                )
                if resp == QtWidgets.QMessageBox.Yes:
                    def do_step1_only():
                        # Open input once
                        cap, frame_w, frame_h, frame_rate = pipeline.open_input_video(self.color_video)
                        # Build the subset up to and including end_scene; if end_scene <= 0, use all
                        limit = self.end_scene if (self.end_scene and self.end_scene > 0) else len(self.scenes_all)
                        # Ensure limit does not exceed actual scenes
                        limit = min(limit, len(self.scenes_all))
                        scene_list = [sr.data for sr in self.scenes_all[:limit]]
                        pipeline.step1_create_scene_videos(cap, scene_list, frame_rate, frame_w, frame_h)
                    self._spawn_worker(FuncWorker, do_step1_only, title='Creating per-scene clips')

            self._refresh_version_combo()

        def _load_scenes(self):
            # Load ALL scenes from CSV and plan paths for ALL (ignore end_scene here)
            self.scenes_all = []
            split_all = pipeline.load_and_split_scenes(self.scene_csv, self.csv_delimiter,
                                                       getattr(self.args, 'max_scene_frames', 1500))
            planned_all = pipeline.plan_scene_files(split_all, self.output_dir, -1)
            for d in planned_all:
                sr = SceneRow(d)
                sr.paths = _plan_paths_for_scene(d, self.output_dir)
                self.scenes_all.append(sr)

            # Now compute the visible subset based on end_scene
            if self.end_scene and self.end_scene > 0:
                self.scenes = self.scenes_all[:self.end_scene]
            else:
                self.scenes = self.scenes_all

            # Populate UI list from visible subset
            self.scene_list.clear()
            for s in self.scenes:
                self.scene_list.addItem(s.display_name())
            if self.scenes:
                self.scene_list.setCurrentRow(0)

        # ---- UI actions ----
        def _on_select_scene(self, row: int):
            if row < 0 or row >= len(self.scenes):
                return
            s = self.scenes[row]
            # remember last engine for this row
            self._last_engine_by_row[row] = s.engine()
            self.engine_combo.blockSignals(True)
            self.engine_combo.setCurrentText(s.engine())
            self.engine_combo.blockSignals(False)
            self._refresh_version_combo()
            self._update_player_source()

        def _refresh_version_combo(self):
            self.version_combo.clear()
            items = [
                ('Scene (raw)', 'scene_video_file'),
                ('Depth', 'depth_video_file'),
                ('Mask', 'mask_video_file'),
                ('SBS', 'sbs'),
                ('SBS + infill mask', 'sbs_infill'),
                ('Infilled (final)', 'infilled'),
            ]
            row = self.scene_list.currentRow()
            s = self.scenes[row] if 0 <= row < len(self.scenes) else None
            for label, key in items:
                idx = self.version_combo.count()
                self.version_combo.addItem(label, userData=key)
                exists = bool(s and os.path.exists(s.paths.get(key, '')))
                model = self.version_combo.model()
                item = None
                if hasattr(model, "item"):
                    try:
                        item = model.item(idx)
                    except Exception:
                        item = None
                if item is not None:
                    flags = item.flags()
                    if not exists and key != 'scene_video_file':
                        item.setFlags(flags & ~Qt.ItemFlag.ItemIsEnabled)
                    else:
                        item.setFlags(flags | Qt.ItemFlag.ItemIsEnabled)

        def _current_scene_and_version_path(self) -> Optional[str]:
            row = self.scene_list.currentRow()
            if row < 0:
                return None
            s = self.scenes[row]
            key = self.version_combo.currentData()
            p = s.paths.get(key)
            if key != 'scene_video_file' and (not p or not os.path.exists(p)):
                return s.paths['scene_video_file']
            return p

        def _update_player_source(self):
            p = self._current_scene_and_version_path()
            if p and os.path.exists(p):
                try:
                    self.player.set_source(p)
                    # switch to video view
                    self.player.viewer_stack.setCurrentIndex(0)
                except Exception as e:
                    self._log(f"Viewer error: {e}")
                    self.player.overlay_label.setText("Unable to load this video")
                    self.player.viewer_stack.setCurrentIndex(1)
            else:
                # No file yet for this scene/version
                self.player.mplayer.setSource(QtCore.QUrl())  # clear player
                self.player.overlay_label.setText("No video file exists for this scene yet")
                self.player.viewer_stack.setCurrentIndex(1)

        def _on_engine_changed(self, eng: str):
            row = self.scene_list.currentRow()
            if row < 0:
                return
            current_engine = self.scenes[row].engine()
            new_engine = eng
            if new_engine == current_engine:
                return
            # Check for existing products of previous engine
            paths = self.scenes[row].paths
            # Also consider FOV and convergence files
            conv_path = None
            if paths.get('depth_video_file'):
                conv_path = paths.get('depth_video_file') + "_convergence_depths.json"
            candidates = [
                paths.get('depth_video_file'),
                paths.get('xfovs_file'),
                conv_path,
                paths.get('sbs'),
                paths.get('sbs_infill'),
                paths.get('infilled'),
            ]
            existing = [p for p in candidates if p and os.path.exists(p)]
            if existing:
                msg = ("Changing the engine will delete existing files produced by the previous engine for this scene:"
                       + "\n".join(existing)
                       + "\n\nDo you want to delete these files and switch engine?")
                resp = QtWidgets.QMessageBox.question(
                    self,
                    'Change engine?',
                    msg,
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No
                )
                if resp != QtWidgets.QMessageBox.Yes:
                    # revert selection
                    self.engine_combo.blockSignals(True)
                    self.engine_combo.setCurrentText(current_engine)
                    self.engine_combo.blockSignals(False)
                    return
                # delete files
                for p in existing:
                    try:
                        os.remove(p)
                        self._log(f'Deleted: {p}')
                    except Exception as e:
                        self._log(f'Failed to delete {p}: {e}')
            # apply change
            self.scenes[row].set_engine(new_engine)
            self._persist_scenes()
            # remember last engine
            self._last_engine_by_row[row] = new_engine
            # refresh full state to avoid stale UI after engine change
            self._reload_after_edit(select_row=row)
            self._update_player_source()

        def _on_split_clicked(self):
            row = self.scene_list.currentRow()
            if row < 0:
                return
            # Operate on the full list; visible subset shares objects at same indices
            s = self.scenes_all[row].data
            # Map current player time to an absolute frame index in this scene
            ss = float(s['Start Time (seconds)']); es = float(s['End Time (seconds)'])
            sf = int(s['Start Frame']);           ef = int(s['End Frame'])
            spf = (es - ss) / (ef - sf) if ef != sf else 0.0
            cur_sec = self.player.current_seconds()
            cur_sec = min(max(cur_sec, ss + 1e-6), es - 1e-6)
            split_frame = int(sf + (cur_sec - ss) / spf) if spf > 0 else (sf + ef) // 2

            if not _split_scene_at_frame(self.scenes_all, row, split_frame):
                self._log("Split ignored (cursor too close to scene edges).")
                return

            _renumber_and_rename(self.scenes_all, row, self.output_dir)
            self._persist_scenes()
            self._reload_after_edit(select_row=row)

        def _on_convert_scene(self):
            row = self.scene_list.currentRow()
            if row < 0:
                return
            sr = self.scenes[row]

            def do_convert_one_scene():
                # Prepare a shallow args clone
                class A: pass
                a = A()
                # mirror fields used by pipeline
                a.color_video = self.color_video
                a.output_dir = self.output_dir
                a.no_render = getattr(self.args, 'no_render', False)
                a.parallel = getattr(self.args, 'parallel', 1)
                a.max_scene_frames = getattr(self.args, 'max_scene_frames', 1500)
                a.no_infill = getattr(self.args, 'no_infill', False)
                a.csv_delimiter = self.csv_delimiter
                a.scene_file = self.scene_csv
                a.end_scene = 1  # we pass only one scene anyway

                # Open video and run step-by-step on JUST this scene
                cap, frame_w, frame_h, frame_rate = pipeline.open_input_video(self.color_video)

                # Important: ensure scene dict is fresh/planned
                scene_dict = sr.data.copy()
                scene_dict.update(_plan_paths_for_scene(scene_dict, self.output_dir))
                scene_list = [scene_dict]

                # Step 1
                pipeline.step1_create_scene_videos(cap, scene_list, frame_rate, frame_w, frame_h)
                # Step 2
                pipeline.step2_estimate_depth(a, scene_list)
                # Step 3
                pipeline.step3_generate_masks(a, scene_list)
                # Step 4
                pipeline.step4_find_convergence(scene_list)
                # Step 5
                if not a.no_render:
                    pipeline.step5_render_sbs(a, scene_list)
                # Step 6 (infill; we don't concat in step 7 for single scene)
                pipeline.step6_infill_and_collect(a, scene_list)
                # Optionally validate this one scene
                # (Commented to avoid hard stop on mismatch; uncomment if desired)
                # assert pipeline.validate_video_lengths(scene_list), "Length mismatch on final video for scene"

            self._spawn_worker(FuncWorker, do_convert_one_scene, title=f"Converting scene #{sr.number}")

        def _on_process_all(self):
            # Run the main pipeline functions inside the process thread
            def do_all():
                # Mirror main() with our args
                a = self.args
                pipeline.ensure_output_dir(a.output_dir)
                pipeline.ensure_scene_file(a)
                cap, frame_w, frame_h, frame_rate = pipeline.open_input_video(a.color_video)
                scenes = pipeline.load_and_split_scenes(a.scene_file, a.csv_delimiter, a.max_scene_frames)
                scene_files = pipeline.plan_scene_files(scenes, a.output_dir, a.end_scene)
                pipeline.step1_create_scene_videos(cap, scene_files, frame_rate, frame_w, frame_h)
                pipeline.step2_estimate_depth(a, scene_files)
                pipeline.step3_generate_masks(a, scene_files)
                pipeline.step4_find_convergence(scene_files)
                if not a.no_render:
                    pipeline.step5_render_sbs(a, scene_files)
                    videos = pipeline.step6_infill_and_collect(a, scene_files)
                    assert pipeline.validate_video_lengths(scene_files), "Something was wrong with one of the video files"
                    pipeline.step7_concat_and_mux(a, videos)

            self._spawn_worker(FuncWorker, do_all, title='Processing all scenes')

        # ---- helpers ----
        def _spawn_worker(self, worker_cls, fn, title: str):
            if self.process_worker and self.process_worker.isRunning():
                QtWidgets.QMessageBox.information(self, 'Busy', 'A job is already running. Please wait for it to finish.')
                return
            # remember current selection so we can restore after reload
            self._selected_row_before_job = self.scene_list.currentRow()
            self._log(f"▶ {title}")
            self.process_worker = worker_cls(fn)
            self.process_worker.line.connect(self._log)
            self.process_worker.done.connect(self._job_done)
            self.process_worker.start()

        def _job_done(self, rc: int):
            self._log(f"◆ Job finished with code {rc}")
            # restore selection if we had one
            sel = self._selected_row_before_job
            self._selected_row_before_job = None
            if sel is not None and 0 <= sel < self.scene_list.count():
                self._reload_after_edit(select_row=sel)
            else:
                self._reload_after_edit()

        def _reload_after_edit(self, select_row: Optional[int] = None):
            self._load_scenes()
            self._refresh_version_combo()
            if select_row is not None and 0 <= select_row < self.scene_list.count():
                self.scene_list.setCurrentRow(select_row)

        def _persist_scenes(self):
            # Always write ALL scenes to CSV, not just the visible subset
            _write_full_scene_csv(self.scene_csv, self.scenes_all, self.csv_delimiter)
            self._log('Saved scenes CSV (all scenes) with updated numbering/engines.')

        def _log(self, msg: str):
            self.console.appendPlainText(msg)

    w = MainWindow(cli_args)
    w.show()
    sys.exit(app.exec())
