# dashboard.py
# Tkinter-based user dashboard for the Illegal Parking Detection System (v2).
# Provides video upload, live feed, zone controls, violation log,
# stats, CSV export, and NEW: night mode, confidence slider, analytics
# charts, heatmap toggle, evidence folder access.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import csv
import datetime
import os
import subprocess
import platform

from roi  import ROIManager
from main import run_detection, get_first_frame

# Try to import matplotlib for analytics charts
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for embedding in Tkinter
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[Dashboard] matplotlib not found — analytics charts disabled.")


# ── Color palette (dark Catppuccin-inspired) ──────────────────────────────────
BG_BASE    = "#1e1e2e"
BG_SURFACE = "#181825"
BG_MANTLE  = "#11111b"
BG_OVERLAY = "#313244"
FG_TEXT    = "#cdd6f4"
FG_SUBTLE  = "#a6adc8"
FG_MUTED   = "#6c7086"
ACCENT_BLUE   = "#89b4fa"
ACCENT_GREEN  = "#a6e3a1"
ACCENT_RED    = "#f38ba8"
ACCENT_YELLOW = "#f9e2af"
ACCENT_PEACH  = "#fab387"
ACCENT_MAUVE  = "#cba6f7"


class Dashboard:
    """
    Main application window (v2 — Upgraded).

    Layout:
        [Left panel — controls] | [Center — live video] | [Right panel — stats + log + charts]
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Illegal Parking Detection System v2.0")
        self.root.geometry("1380x800")
        self.root.minsize(1200, 700)
        self.root.configure(bg=BG_BASE)

        # State
        self.video_path       = None
        self.zones            = []
        self.stop_flag        = [False]
        self.detection_thread = None
        self.violations_log   = []       # List of violation dicts for CSV export
        self._latest_stats    = {}       # Latest stats from detection loop

        self._build_ui()

    # ══════════════════════════════════════════════════════════════════════════
    #  UI Construction
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self._build_title_bar()

        content = tk.Frame(self.root, bg=BG_BASE)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self._build_left_panel(content)
        self._build_center_panel(content)
        self._build_right_panel(content)

    def _build_title_bar(self):
        bar = tk.Frame(self.root, bg=BG_MANTLE, height=52)
        bar.pack(fill=tk.X)

        tk.Label(
            bar,
            text="  🚗 Illegal Parking Detection System  v2.0",
            font=("Helvetica", 15, "bold"),
            bg=BG_MANTLE, fg=FG_TEXT,
        ).pack(side=tk.LEFT, pady=12)

        # Version badge
        tk.Label(
            bar, text=" UPGRADED ",
            font=("Helvetica", 8, "bold"),
            bg=ACCENT_MAUVE, fg=BG_MANTLE,
        ).pack(side=tk.LEFT, padx=6, pady=16)

        self.status_label = tk.Label(
            bar, text="● Idle",
            font=("Helvetica", 10), bg=BG_MANTLE, fg=FG_MUTED,
        )
        self.status_label.pack(side=tk.RIGHT, padx=18)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self, parent):
        panel = tk.Frame(parent, bg=BG_SURFACE, width=250)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), pady=5)
        panel.pack_propagate(False)

        # ── Video Input ──────────────────────────────────────────────────────
        self._section(panel, "📁  Video Input")
        self._btn(panel, "Upload Video", self._upload_video, bg=BG_OVERLAY, fg=FG_TEXT)
        self.video_name_label = self._info_label(panel, "No video selected")

        # ── No-Parking Zones ─────────────────────────────────────────────────
        self._section(panel, "🚫  No-Parking Zones")
        self._btn(panel, "Define Zones on Frame", self._define_zones, bg=BG_OVERLAY, fg=FG_TEXT)
        self.zone_info_label = self._info_label(panel, "0 zones defined")

        # ── Time Threshold ───────────────────────────────────────────────────
        self._section(panel, "⏱  Time Threshold (seconds)")

        self.threshold_var = tk.IntVar(value=30)
        slider = ttk.Scale(
            panel, from_=5, to=120, orient=tk.HORIZONTAL,
            variable=self.threshold_var, command=self._on_threshold_change,
        )
        slider.pack(fill=tk.X, padx=12, pady=(2, 0))

        self.threshold_val_label = tk.Label(
            panel, text="30 s", bg=BG_SURFACE,
            fg=ACCENT_GREEN, font=("Helvetica", 10, "bold"),
        )
        self.threshold_val_label.pack(anchor=tk.W, padx=14)

        # ── Confidence Threshold (NEW) ───────────────────────────────────────
        self._section(panel, "🎯  Detection Confidence")

        self.confidence_var = tk.DoubleVar(value=0.40)
        conf_slider = ttk.Scale(
            panel, from_=0.20, to=0.90, orient=tk.HORIZONTAL,
            variable=self.confidence_var, command=self._on_confidence_change,
        )
        conf_slider.pack(fill=tk.X, padx=12, pady=(2, 0))

        self.confidence_val_label = tk.Label(
            panel, text="0.40", bg=BG_SURFACE,
            fg=ACCENT_PEACH, font=("Helvetica", 10, "bold"),
        )
        self.confidence_val_label.pack(anchor=tk.W, padx=14)

        # ── DIP Enhancements (NEW) ───────────────────────────────────────────
        self._section(panel, "🔧  DIP Enhancements")

        self.night_mode_var = tk.BooleanVar(value=False)
        night_check = tk.Checkbutton(
            panel, text="  Night Mode (CLAHE)", variable=self.night_mode_var,
            bg=BG_SURFACE, fg=FG_TEXT, selectcolor=BG_OVERLAY,
            activebackground=BG_SURFACE, activeforeground=FG_TEXT,
            font=("Helvetica", 9),
        )
        night_check.pack(anchor=tk.W, padx=12, pady=(2, 0))

        self.heatmap_var = tk.BooleanVar(value=False)
        heat_check = tk.Checkbutton(
            panel, text="  Show Violation Heatmap", variable=self.heatmap_var,
            bg=BG_SURFACE, fg=FG_TEXT, selectcolor=BG_OVERLAY,
            activebackground=BG_SURFACE, activeforeground=FG_TEXT,
            font=("Helvetica", 9),
        )
        heat_check.pack(anchor=tk.W, padx=12, pady=(2, 0))

        # ── Controls ─────────────────────────────────────────────────────────
        self._section(panel, "▶  Controls")

        self.start_btn = tk.Button(
            panel, text="▶  Start Detection",
            command=self._start_detection, relief=tk.FLAT, cursor="hand2",
            bg=ACCENT_GREEN, fg=BG_MANTLE,
            font=("Helvetica", 10, "bold"), pady=7,
        )
        self.start_btn.pack(fill=tk.X, padx=12, pady=(3, 2))

        self.stop_btn = tk.Button(
            panel, text="■  Stop Detection",
            command=self._stop_detection, relief=tk.FLAT, cursor="hand2",
            bg=ACCENT_RED, fg=BG_MANTLE,
            font=("Helvetica", 10, "bold"), pady=7,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(fill=tk.X, padx=12, pady=2)

        # ── Export ───────────────────────────────────────────────────────────
        self._section(panel, "💾  Export & Evidence")
        self._btn(panel, "Save Report as CSV", self._save_report, bg=BG_OVERLAY, fg=FG_TEXT)
        self._btn(panel, "Open Evidence Folder", self._open_evidence, bg=BG_OVERLAY, fg=FG_TEXT)

        # spacer
        tk.Frame(panel, bg=BG_SURFACE).pack(fill=tk.BOTH, expand=True)

        tk.Label(
            panel,
            text="Digital Image Processing Project\nSmart Transportation — v2.0",
            font=("Helvetica", 7), bg=BG_SURFACE, fg=FG_MUTED, justify=tk.CENTER,
        ).pack(pady=8)

    # ── Center panel (video feed) ─────────────────────────────────────────────

    def _build_center_panel(self, parent):
        panel = tk.Frame(parent, bg=BG_BASE)
        panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(
            panel, text="Live Video Feed",
            font=("Helvetica", 10, "bold"), bg=BG_BASE, fg=FG_SUBTLE,
        ).pack(anchor=tk.W, pady=(5, 3))

        self.video_label = tk.Label(
            panel, bg=BG_MANTLE,
            text="No video loaded.\n\nUpload a video file to begin.",
            fg=FG_MUTED, font=("Helvetica", 12),
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

    # ── Right panel (stats + log + charts) ────────────────────────────────────

    def _build_right_panel(self, parent):
        panel = tk.Frame(parent, bg=BG_SURFACE, width=310)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0), pady=5)
        panel.pack_propagate(False)

        # ── Statistics ────────────────────────────────────────────────────────
        self._section(panel, "📊  Statistics")

        stats_box = tk.Frame(panel, bg=BG_OVERLAY, relief=tk.FLAT)
        stats_box.pack(fill=tk.X, padx=10, pady=(0, 6))

        self.stat_detected    = self._stat_row(stats_box, "Vehicles Detected",  "0", ACCENT_BLUE)
        self.stat_stationary  = self._stat_row(stats_box, "Stationary",         "0", ACCENT_YELLOW)
        self.stat_violations  = self._stat_row(stats_box, "Total Violations",   "0", ACCENT_RED)
        self.stat_zones       = self._stat_row(stats_box, "Active Zones",       "0", ACCENT_PEACH)

        # ── Violation Log ────────────────────────────────────────────────────
        self._section(panel, "📋  Violation Log")

        tree_frame = tk.Frame(panel, bg=BG_SURFACE)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))

        cols = ("ID", "Type", "Zone", "Time", "Duration")
        self.log_tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings", height=12,
        )

        # Style the treeview
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Treeview",
            background=BG_OVERLAY, foreground=FG_TEXT,
            fieldbackground=BG_OVERLAY, rowheight=23, font=("Helvetica", 8),
        )
        style.configure(
            "Treeview.Heading",
            background=BG_MANTLE, foreground=FG_SUBTLE,
            font=("Helvetica", 8, "bold"),
        )
        style.map("Treeview", background=[("selected", "#585b70")])

        col_widths = {"ID": 36, "Type": 68, "Zone": 52, "Time": 62, "Duration": 58}
        for col in cols:
            self.log_tree.heading(col, text=col)
            self.log_tree.column(col, width=col_widths[col], anchor=tk.CENTER)

        sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.log_tree.yview)
        self.log_tree.configure(yscrollcommand=sb.set)
        self.log_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # ── Analytics Charts (NEW) ───────────────────────────────────────────
        self._section(panel, "📈  Analytics")
        self._build_charts(panel)

    def _build_charts(self, parent):
        """Build embedded matplotlib charts for violation analytics."""
        self.chart_frame = tk.Frame(parent, bg=BG_SURFACE)
        self.chart_frame.pack(fill=tk.X, padx=10, pady=(0, 8))

        if not HAS_MATPLOTLIB:
            tk.Label(
                self.chart_frame,
                text="Install matplotlib for charts:\npip install matplotlib",
                bg=BG_SURFACE, fg=FG_MUTED, font=("Helvetica", 8),
                justify=tk.CENTER,
            ).pack(pady=10)
            self.fig = None
            return

        # Create matplotlib figure with dark theme
        self.fig = Figure(figsize=(3.0, 2.8), dpi=85, facecolor=BG_SURFACE)
        self.fig.subplots_adjust(hspace=0.6, top=0.92, bottom=0.15, left=0.2, right=0.95)

        # Two subplots: timeline + zone bar chart
        self.ax_timeline = self.fig.add_subplot(211)
        self.ax_zones    = self.fig.add_subplot(212)

        self._style_axes(self.ax_timeline, "Violations Over Time")
        self._style_axes(self.ax_zones, "Violations Per Zone")

        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.X)
        self.chart_canvas.draw()

    def _style_axes(self, ax, title):
        """Apply dark theme to a matplotlib axes."""
        ax.set_facecolor(BG_OVERLAY)
        ax.set_title(title, fontsize=8, color=FG_SUBTLE, pad=4)
        ax.tick_params(colors=FG_MUTED, labelsize=6)
        for spine in ax.spines.values():
            spine.set_color(FG_MUTED)

    # ══════════════════════════════════════════════════════════════════════════
    #  Widget helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _section(self, parent, text):
        """Divider label for panel sections."""
        tk.Label(
            parent, text=text,
            font=("Helvetica", 9, "bold"), bg=BG_SURFACE, fg=ACCENT_BLUE, anchor=tk.W,
        ).pack(fill=tk.X, padx=8, pady=(12, 3))

    def _btn(self, parent, text, command, bg=BG_OVERLAY, fg=FG_TEXT):
        """Standard panel button."""
        tk.Button(
            parent, text=text, command=command, relief=tk.FLAT, cursor="hand2",
            bg=bg, fg=fg, font=("Helvetica", 9), pady=6, activebackground="#45475a",
        ).pack(fill=tk.X, padx=12, pady=2)

    def _info_label(self, parent, text):
        """Small grey informational label."""
        lbl = tk.Label(
            parent, text=text, bg=BG_SURFACE, fg=FG_MUTED,
            font=("Helvetica", 8), wraplength=220, justify=tk.LEFT,
        )
        lbl.pack(anchor=tk.W, padx=14, pady=(0, 2))
        return lbl

    def _stat_row(self, parent, label, value, value_color):
        """One statistic row returning the value label widget."""
        row = tk.Frame(parent, bg=BG_OVERLAY)
        row.pack(fill=tk.X, padx=4, pady=2)
        tk.Label(row, text=label, bg=BG_OVERLAY, fg=FG_SUBTLE,
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=6)
        val_lbl = tk.Label(row, text=value, bg=BG_OVERLAY, fg=value_color,
                           font=("Helvetica", 9, "bold"))
        val_lbl.pack(side=tk.RIGHT, padx=6)
        return val_lbl

    # ══════════════════════════════════════════════════════════════════════════
    #  Event handlers
    # ══════════════════════════════════════════════════════════════════════════

    def _on_threshold_change(self, val):
        self.threshold_val_label.config(text=f"{int(float(val))} s")

    def _on_confidence_change(self, val):
        self.confidence_val_label.config(text=f"{float(val):.2f}")

    def _upload_video(self):
        path = filedialog.askopenfilename(
            title="Select CCTV Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.video_path = path
        name = os.path.basename(path)
        self.video_name_label.config(text=name, fg=FG_TEXT)

        # Show a preview of the first frame
        frame = get_first_frame(path)
        if frame is not None:
            self._display_frame(frame)
        else:
            messagebox.showerror("Error", "Could not read the video file.")

    def _define_zones(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please upload a video first.")
            return

        frame = get_first_frame(self.video_path)
        if frame is None:
            messagebox.showerror("Error", "Could not read video frame.")
            return

        self._set_status("● Defining Zones...", ACCENT_YELLOW)

        def _run_zone_definition():
            roi_manager = ROIManager()
            zones = roi_manager.define_zones_interactive(frame)
            # Schedule UI update back on the main thread
            self.root.after(0, lambda: self._on_zones_defined(zones))

        threading.Thread(target=_run_zone_definition, daemon=True).start()

    def _on_zones_defined(self, zones):
        """Called on the main thread after zone definition completes."""
        self.zones = zones
        count = len(self.zones)
        self.zone_info_label.config(
            text=f"{count} zone(s) defined",
            fg=ACCENT_GREEN if count > 0 else FG_MUTED,
        )
        self.stat_zones.config(text=str(count))
        self._set_status("● Idle", FG_MUTED)

    def _start_detection(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please upload a video first.")
            return
        if not self.zones:
            messagebox.showwarning("No Zones", "Please define at least one no-parking zone.")
            return

        # Clear previous session data
        self.violations_log = []
        self._latest_stats = {}
        for row in self.log_tree.get_children():
            self.log_tree.delete(row)
        self.stat_detected.config(text="0")
        self.stat_stationary.config(text="0")
        self.stat_violations.config(text="0")

        # Reset charts
        if self.fig:
            self.ax_timeline.clear()
            self.ax_zones.clear()
            self._style_axes(self.ax_timeline, "Violations Over Time")
            self._style_axes(self.ax_zones, "Violations Per Zone")
            self.chart_canvas.draw()

        self.stop_flag = [False]
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._set_status("● Running", ACCENT_GREEN)

        output_path  = os.path.splitext(self.video_path)[0] + "_output.mp4"
        threshold    = self.threshold_var.get()
        night_mode   = self.night_mode_var.get()
        confidence   = self.confidence_var.get()
        show_heatmap = self.heatmap_var.get()

        self.detection_thread = threading.Thread(
            target=run_detection,
            kwargs=dict(
                video_path           = self.video_path,
                zones                = self.zones,
                threshold            = threshold,
                output_path          = output_path,
                frame_callback       = self._on_frame,
                log_callback         = self._on_violation,
                stop_flag            = self.stop_flag,
                stats_callback       = self._on_stats,
                night_mode           = night_mode,
                confidence_threshold = confidence,
                show_heatmap         = show_heatmap,
            ),
            daemon=True,
        )
        self.detection_thread.start()
        self.root.after(300, self._poll_thread)

    def _stop_detection(self):
        self.stop_flag[0] = True
        self._set_status("● Stopped", ACCENT_RED)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _save_report(self):
        if not self.violations_log:
            messagebox.showinfo("No Data", "No violations have been recorded yet.")
            return

        default_name = f"parking_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=default_name,
            title="Save Violation Report",
        )
        if not path:
            return

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["vehicle_id", "vehicle_type", "zone", "timestamp", "duration"],
            )
            writer.writeheader()
            writer.writerows(self.violations_log)

        messagebox.showinfo("Saved", f"Report saved to:\n{path}")

    def _open_evidence(self):
        """Open the evidence folder in the system file explorer."""
        evidence_dir = os.path.abspath("evidence")
        if not os.path.exists(evidence_dir):
            os.makedirs(evidence_dir, exist_ok=True)

        if platform.system() == "Windows":
            os.startfile(evidence_dir)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", evidence_dir])
        else:
            subprocess.Popen(["xdg-open", evidence_dir])

    # ══════════════════════════════════════════════════════════════════════════
    #  Callbacks from detection thread (always schedule on main thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _on_frame(self, frame):
        """Receive an annotated frame and display it on the canvas."""
        self.root.after(0, lambda f=frame: self._display_frame(f))

    def _on_violation(self, viol_info):
        """Receive a new violation dict and add it to the log."""
        self.violations_log.append(viol_info)
        self.root.after(0, lambda v=viol_info: self._add_log_row(v))

    def _on_stats(self, stats):
        """Receive updated statistics and refresh the stats panel."""
        self._latest_stats = stats
        self.root.after(0, lambda s=stats: self._update_stats(s))

    # ══════════════════════════════════════════════════════════════════════════
    #  UI update helpers (must run on main thread)
    # ══════════════════════════════════════════════════════════════════════════

    def _display_frame(self, frame):
        """Resize a BGR frame and show it on the video label widget."""
        try:
            lbl_w = self.video_label.winfo_width()  or 800
            lbl_h = self.video_label.winfo_height() or 520
            h, w  = frame.shape[:2]
            scale = min(lbl_w / w, lbl_h / h)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

            resized = cv2.resize(frame, (new_w, new_h))
            rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img     = Image.fromarray(rgb)
            imgtk   = ImageTk.PhotoImage(image=img)

            self.video_label.config(image=imgtk, text="")
            self.video_label.image = imgtk   # prevent garbage collection
        except Exception:
            pass

    def _add_log_row(self, v):
        """Insert a violation row at the top of the treeview."""
        self.log_tree.insert(
            "", 0,
            values=(
                v["vehicle_id"],
                v["vehicle_type"].capitalize(),
                v["zone"],
                v["timestamp"],
                f"{v['duration']:.0f}s",
            ),
            tags=("violation",),
        )
        self.log_tree.tag_configure("violation", foreground=ACCENT_RED)

    def _update_stats(self, stats):
        self.stat_detected.config(text=str(stats.get("total_detected", 0)))
        self.stat_stationary.config(text=str(stats.get("stationary", 0)))
        self.stat_violations.config(text=str(stats.get("total_violations", 0)))
        self.stat_zones.config(text=str(stats.get("active_zones", 0)))

        # Update charts every 60 frames worth of stats
        if self.fig and stats.get("total_violations", 0) > 0:
            self._update_charts(stats)

    def _update_charts(self, stats):
        """Refresh the matplotlib analytics charts."""
        if not self.fig:
            return

        try:
            # ── Timeline chart ────────────────────────────────────────────
            timeline = stats.get("timeline", [])
            if timeline:
                self.ax_timeline.clear()
                self._style_axes(self.ax_timeline, "Violations Over Time")

                # Plot cumulative violations over time
                if len(timeline) > 0:
                    start_t = timeline[0][0]
                    times = [(t - start_t) for t, _ in timeline]
                    counts = list(range(1, len(times) + 1))
                    self.ax_timeline.plot(times, counts, color="#f38ba8",
                                         linewidth=1.5, marker="o", markersize=3)
                    self.ax_timeline.set_xlabel("Time (s)", fontsize=6, color=FG_MUTED)
                    self.ax_timeline.set_ylabel("Count", fontsize=6, color=FG_MUTED)

            # ── Zone bar chart ────────────────────────────────────────────
            zone_stats = stats.get("zone_stats", {})
            if zone_stats:
                self.ax_zones.clear()
                self._style_axes(self.ax_zones, "Violations Per Zone")

                zones = list(zone_stats.keys())
                counts = list(zone_stats.values())
                colors = [ACCENT_RED, ACCENT_PEACH, ACCENT_YELLOW, ACCENT_MAUVE,
                          ACCENT_BLUE, ACCENT_GREEN]
                bar_colors = [colors[i % len(colors)] for i in range(len(zones))]

                self.ax_zones.bar(zones, counts, color=bar_colors, width=0.5)
                self.ax_zones.set_ylabel("Count", fontsize=6, color=FG_MUTED)

            self.chart_canvas.draw_idle()

        except Exception:
            pass

    def _poll_thread(self):
        """Check whether the detection thread has finished."""
        if self.detection_thread and self.detection_thread.is_alive():
            self.root.after(500, self._poll_thread)
        else:
            if not self.stop_flag[0]:
                self._set_status("● Finished", ACCENT_BLUE)
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)

    def _set_status(self, text, color):
        self.status_label.config(text=text, fg=color)


# ──────────────────────────────────────────────────────────────────────────────
#  Public launcher (called from main.py)
# ──────────────────────────────────────────────────────────────────────────────

def launch_dashboard():
    root = tk.Tk()
    Dashboard(root)
    root.mainloop()
