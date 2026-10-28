"""Execute training cells in a Jupyter kernel. No model fit code lives here."""
from pathlib import Path
import sys
import os
runtime = Path(__file__).resolve().parents[1] / ".runtime"
runtime.mkdir(exist_ok=True)
os.environ["IPYTHONDIR"] = str(runtime / "ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime / "jupyter")
os.environ["MPLCONFIGDIR"] = str(runtime / "matplotlib")
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
import nbformat

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/01_train_maintenance.ipynb"
notebook = nbformat.read(path, as_version=4)

if "--offline-shell" in sys.argv:
    # Provisional offline cell evaluation; not a full Jupyter server session.
    import os
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    os.chdir(root)
    from traitlets.config import Config
    config = Config()
    config.HistoryManager.hist_file = ":memory:"
    shell = InteractiveShell.instance(config=config)
    try:
        for count, cell in enumerate((c for c in notebook.cells if c.cell_type == "code"), 1):
            print(f"Offline IPython shell: executing cell {count}", flush=True)
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            cell.execution_count = count
            cell.outputs = []
            if captured.stdout:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output("display_data", data=output.data, metadata=output.metadata))
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(f"Notebook cell {count} failed") from (result.error_before_exec or result.error_in_exec)
    finally:
        InteractiveShell.clear_instance()
    notebook.metadata["execution_backend"] = "Offline IPython cell evaluation; rerun in Jupyter for assignment"
else:
    from nbclient import NotebookClient
    NotebookClient(notebook, timeout=180, startup_timeout=30, kernel_name=os.getenv("NOTEBOOK_KERNEL", "python3"),
                   resources={"metadata": {"path": str(root)}}).execute()
    notebook.metadata["execution_backend"] = "Jupyter kernel via nbclient"
nbformat.write(notebook, path)
print(f"Executed notebook: {path}")
