#!/usr/bin/env python3
"""
pexels_downloader.py
Single-file PyQt5 app to search + download images from the Pexels API.

Requirements:
    pip install PyQt5 requests

Before running, set your PEXELS_API_KEY environment variable:
    export PEXELS_API_KEY="your_key_here"   # Linux / macOS
    setx PEXELS_API_KEY "your_key_here"     # Windows (then restart terminal)

Author: ChatGPT (single-file version)
"""

import os
import sys
import requests
from PyQt5 import QtCore, QtWidgets, QtGui

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
MAX_PER_PAGE = 80

class DownloaderThread(QtCore.QThread):
    progress_changed = QtCore.pyqtSignal(int)
    status_changed = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(bool, str)

    def __init__(self, api_key, query, count, folder, parent=None):
        super().__init__(parent)
        self.api_key = "weNk0JBE1bENuYpmGQDhuvljDUZeqxdwyqGx5rGBIWnJZ2d2SSy7hcRL"
        self.query = query.strip()
        self.count = int(count)
        self.folder = folder

    def run(self):
        if not self.api_key:
            self.finished_signal.emit(False, "PEXELS_API_KEY not set in environment.")
            return
        if not self.query:
            self.finished_signal.emit(False, "Empty search query.")
            return
        if self.count <= 0:
            self.finished_signal.emit(False, "Number of images must be > 0.")
            return
        if not os.path.isdir(self.folder):
            self.finished_signal.emit(False, "Destination folder does not exist.")
            return

        headers = {"Authorization": self.api_key}
        downloaded = 0
        page = 1
        total_to_get = self.count
        self.status_changed.emit(f"Searching for \"{self.query}\"...")

        try:
            while downloaded < total_to_get:
                per_page = min(MAX_PER_PAGE, total_to_get - downloaded)
                params = {"query": self.query, "per_page": per_page, "page": page}
                resp = requests.get(PEXELS_SEARCH_URL, headers=headers, params=params, timeout=20)
                if resp.status_code == 401:
                    self.finished_signal.emit(False, "Unauthorized: check your API key.")
                    return
                if resp.status_code != 200:
                    self.finished_signal.emit(False, f"Pexels API error: {resp.status_code} {resp.text}")
                    return

                data = resp.json()
                photos = data.get("photos", [])
                if not photos:
                    if downloaded == 0:
                        self.finished_signal.emit(False, "No photos found for that query.")
                        return
                    else:
                        self.status_changed.emit("No more photos available.")
                        break

                for p in photos:
                    if downloaded >= total_to_get:
                        break

                    src = p.get("src", {})
                    img_url = src.get("original") or src.get("large2x") or src.get("large") or src.get("medium")
                    if not img_url:
                        continue

                    photo_id = p.get("id", "img")
                    ext = os.path.splitext(img_url.split("?")[0])[1] or ".jpg"
                    safe_query = "".join(ch if ch.isalnum() else "_" for ch in self.query)[:40]
                    fname = f"{safe_query}_{downloaded+1:04d}_{photo_id}{ext}"
                    out_path = os.path.join(self.folder, fname)

                    self.status_changed.emit(f"Downloading {downloaded+1}/{total_to_get} ...")
                    try:
                        dl = requests.get(img_url, stream=True, timeout=30)
                        if dl.status_code == 200:
                            with open(out_path, "wb") as f:
                                for chunk in dl.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                        else:
                            self.status_changed.emit(f"Warning: failed to download image {photo_id}.")
                            continue
                    except Exception as ex:
                        self.status_changed.emit(f"Warning: error downloading {img_url}: {ex}")
                        continue

                    downloaded += 1
                    percent = int((downloaded / total_to_get) * 100)
                    self.progress_changed.emit(percent)

                page += 1

            if downloaded == 0:
                self.finished_signal.emit(False, "No images were downloaded.")
            else:
                self.finished_signal.emit(True, f"Downloaded {downloaded} image(s) to: {self.folder}")
        except Exception as e:
            self.finished_signal.emit(False, f"Unexpected error: {e}")


class PexelsDownloaderApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pexels API Downloader")
        self.setMinimumSize(520, 260)
        self._build_ui()
        self.downloader_thread = None

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        row_q = QtWidgets.QHBoxLayout()
        row_q.addWidget(QtWidgets.QLabel("Search keywords:"))
        self.query_edit = QtWidgets.QLineEdit()
        self.query_edit.setPlaceholderText("e.g. cats, mountains, space wallpaper")
        row_q.addWidget(self.query_edit)
        layout.addLayout(row_q)

        row_controls = QtWidgets.QHBoxLayout()
        row_controls.addWidget(QtWidgets.QLabel("Number of images:"))
        self.spin_count = QtWidgets.QSpinBox()
        self.spin_count.setRange(1, 1000)
        self.spin_count.setValue(10)
        row_controls.addWidget(self.spin_count)

        row_controls.addSpacing(10)
        row_controls.addWidget(QtWidgets.QLabel("Destination:"))
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setReadOnly(True)
        row_controls.addWidget(self.folder_edit)

        self.btn_browse = QtWidgets.QPushButton("Choose...")
        self.btn_browse.clicked.connect(self.choose_folder)
        row_controls.addWidget(self.btn_browse)
        layout.addLayout(row_controls)

        row_action = QtWidgets.QHBoxLayout()
        self.btn_download = QtWidgets.QPushButton("Download")
        self.btn_download.clicked.connect(self.on_download)
        row_action.addWidget(self.btn_download)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        self.progress.setFormat("%p%")
        row_action.addWidget(self.progress)
        layout.addLayout(row_action)

        layout.addWidget(QtWidgets.QLabel("Status:"))
        self.status_text = QtWidgets.QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(110)
        layout.addWidget(self.status_text)

        note = QtWidgets.QLabel("Make sure PEXELS_API_KEY is set in your environment.")
        note.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(note)

    def choose_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select destination folder", os.path.expanduser("~"))
        if folder:
            self.folder_edit.setText(folder)

    def append_status(self, text):
        self.status_text.append(text)
        self.status_text.moveCursor(QtGui.QTextCursor.End)

    def on_download(self):
        if self.downloader_thread and self.downloader_thread.isRunning():
            QtWidgets.QMessageBox.information(self, "Please wait", "A download is already in progress.")
            return

        api_key = os.environ.get("PEXELS_API_KEY", "")
        query = self.query_edit.text()
        count = self.spin_count.value()
        folder = self.folder_edit.text().strip() or os.path.expanduser("~")

        if not os.path.isdir(folder):
            try:
                os.makedirs(folder, exist_ok=True)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Folder error", f"Unable to create/access folder: {e}")
                return

        self.progress.setValue(0)
        self.status_text.clear()
        self.append_status("Starting...")

        self.downloader_thread = DownloaderThread(api_key, query, count, folder)
        self.downloader_thread.progress_changed.connect(self.progress.setValue)
        self.downloader_thread.status_changed.connect(self.append_status)
        self.downloader_thread.finished_signal.connect(self.on_finished)
        self.btn_download.setEnabled(False)
        self.downloader_thread.start()

    def on_finished(self, success, message):
        self.append_status(message)
        self.btn_download.setEnabled(True)
        if success:
            QtWidgets.QMessageBox.information(self, "Download complete", message)
        else:
            QtWidgets.QMessageBox.warning(self, "Download failed", message)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PexelsDownloaderApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
