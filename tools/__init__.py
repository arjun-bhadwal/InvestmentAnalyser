"""
Tool registration — importing each module triggers @mcp.tool() decoration.
"""
from tools import portfolio      # noqa: F401
from tools import market_data    # noqa: F401
from tools import analysis       # noqa: F401
from tools import macro          # noqa: F401
from tools import news           # noqa: F401
from tools import risk           # noqa: F401
from tools import insider        # noqa: F401
