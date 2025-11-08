#!/usr/bin/env python3
"""
pexels_downloader.py
Single-file PyQt5 app to search + download images from the Pexels API with multi-threading.

Requirements:
    pip install PyQt5 requests

Before running, set your PEXELS_API_KEY environment variable:
    export PEXELS_API_KEY="your_key_here"   # Linux / macOS
    setx PEXELS_API_KEY "your_key_here"     # Windows (then restart terminal)

Author: ChatGPT (single-file version with multi-threading)
"""

import os
import sys
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from PyQt5 import QtCore, QtWidgets, QtGui

PEXELS_SEARCH_URL = "https://api.pexels.com/v1/search"
MAX_PER_PAGE = 80
MAX_WORKERS = 10  # Number of concurrent download threads


class DownloaderThread(QtCore.QThread):
    progress_changed = QtCore.pyqtSignal(int)
    status_changed = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(bool, str)

    def __init__(self, api_key, query, count, folder, max_workers=MAX_WORKERS, parent=None):
        super().__init__(parent)
        self.api_key = "weNk0JBE1bENuYpmGQDhuvljDUZeqxdwyqGx5rGBIWnJZ2d2SSy7hcRL"
        self.query = query.strip()
        self.count = int(count)
        self.folder = folder
        self.max_workers = max_workers
        self.downloaded_count = 0
        self.lock = Lock()

    def download_single_image(self, img_data):
        """Download a single image - designed to run in thread pool"""
        img_url, out_path, index, total = img_data
        try:
            dl = requests.get(img_url, stream=True, timeout=30)
            if dl.status_code == 200:
                with open(out_path, "wb") as f:
                    for chunk in dl.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                with self.lock:
                    self.downloaded_count += 1
                    percent = int((self.downloaded_count / total) * 100)
                    self.progress_changed.emit(percent)
                    self.status_changed.emit(
                        f"Downloaded {self.downloaded_count}/{total}: {os.path.basename(out_path)}"
                    )

                return True, out_path
            else:
                return False, f"HTTP {dl.status_code} for {img_url}"
        except Exception as ex:
            return False, f"Error downloading {img_url}: {ex}"

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
        page = 1
        total_to_get = self.count
        self.status_changed.emit(f"Searching for \"{self.query}\"...")

        # Collect all image URLs first
        images_to_download = []

        try:
            while len(images_to_download) < total_to_get:
                per_page = min(MAX_PER_PAGE, total_to_get - len(images_to_download))
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
                    if len(images_to_download) == 0:
                        self.finished_signal.emit(False, "No photos found for that query.")
                        return
                    else:
                        self.status_changed.emit("No more photos available.")
                        break

                for p in photos:
                    if len(images_to_download) >= total_to_get:
                        break

                    src = p.get("src", {})
                    img_url = src.get("original") or src.get("large2x") or src.get("large") or src.get("medium")
                    if not img_url:
                        continue

                    photo_id = p.get("id", "img")
                    ext = os.path.splitext(img_url.split("?")[0])[1] or ".jpg"
                    safe_query = "".join(ch if ch.isalnum() else "_" for ch in self.query)[:40]
                    fname = f"{safe_query}_{len(images_to_download)+1:04d}_{photo_id}{ext}"
                    out_path = os.path.join(self.folder, fname)

                    images_to_download.append((img_url, out_path, len(images_to_download) + 1, total_to_get))

                page += 1

            if not images_to_download:
                self.finished_signal.emit(False, "No images found to download.")
                return

            # Download all images concurrently using ThreadPoolExecutor
            self.status_changed.emit(
                f"Starting concurrent download of {len(images_to_download)} images with {self.max_workers} threads..."
            )

            successful = 0
            failed = 0

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_img = {
                    executor.submit(self.download_single_image, img_data): img_data for img_data in images_to_download
                }

                for future in as_completed(future_to_img):
                    success, msg = future.result()
                    if success:
                        successful += 1
                    else:
                        failed += 1
                        self.status_changed.emit(f"Warning: {msg}")

            if successful == 0:
                self.finished_signal.emit(False, "No images were downloaded successfully.")
            else:
                summary = f"Downloaded {successful} image(s) to: {self.folder}"
                if failed > 0:
                    summary += f" ({failed} failed)"
                self.finished_signal.emit(True, summary)

        except Exception as e:
            self.finished_signal.emit(False, f"Unexpected error: {e}")


class PexelsDownloaderApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pexels API Downloader (Multi-threaded)")
        self.setMinimumSize(520, 320)
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

        # Add thread count control
        row_threads = QtWidgets.QHBoxLayout()
        row_threads.addWidget(QtWidgets.QLabel("Concurrent threads:"))
        self.spin_threads = QtWidgets.QSpinBox()
        self.spin_threads.setRange(1, 20)
        self.spin_threads.setValue(MAX_WORKERS)
        self.spin_threads.setToolTip("Number of simultaneous downloads (higher = faster, but may overload)")
        row_threads.addWidget(self.spin_threads)
        row_threads.addStretch()
        layout.addLayout(row_threads)

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

        note = QtWidgets.QLabel("Make sure PEXELS_API_KEY is set in your environment. Multi-threaded for faster downloads!")
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
        max_workers = self.spin_threads.value()

        if not os.path.isdir(folder):
            try:
                os.makedirs(folder, exist_ok=True)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Folder error", f"Unable to create/access folder: {e}")
                return

        self.progress.setValue(0)
        self.status_text.clear()
        self.append_status(f"Starting with {max_workers} concurrent threads...")

        self.downloader_thread = DownloaderThread(api_key, query, count, folder, max_workers)
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
