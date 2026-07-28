"""Launch Shoulder Gear Designer."""

from __future__ import annotations

import sys
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox

from literature_pitch_gui import LiteraturePitchGUI


def main() -> None:
    """Create, foreground, and run the Tk application.

    Startup information is printed to PyCharm's Run pane. Unexpected startup
    errors are also saved beside ``main.py`` so failures cannot appear silent.
    """

    project_directory = Path(__file__).resolve().parent
    error_log = project_directory / "startup_error.log"
    print(f"Starting Shoulder Gear Designer with {sys.executable}", flush=True)

    try:
        root = tk.Tk()
        root.geometry("1200x760")
        LiteraturePitchGUI(root)

        # macOS can place a new Tk window behind PyCharm. Raise it temporarily,
        # then remove the always-on-top flag so it behaves like a normal window.
        root.deiconify()
        root.lift()
        root.attributes("-topmost", True)
        root.after(750, lambda: root.attributes("-topmost", False))
        root.after(100, root.focus_force)
        root.update_idletasks()
        print("GUI initialized successfully.", flush=True)
        root.mainloop()
    except Exception as error:
        details = "".join(traceback.format_exception(error))
        error_log.write_text(details, encoding="utf-8")
        print(details, file=sys.stderr, flush=True)
        try:
            messagebox.showerror(
                "Shoulder Gear Designer startup failed",
                f"{error}\n\nDetails were saved to:\n{error_log}",
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
