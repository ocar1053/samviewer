from pathlib import Path
import sys

from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.web import cli as streamlit_cli


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from samviewer.app import main  # noqa: E402


if __name__ == "__main__":
    if get_script_run_ctx() is None:
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(streamlit_cli.main())

    main()
