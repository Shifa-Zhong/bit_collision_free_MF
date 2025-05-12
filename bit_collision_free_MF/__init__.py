"""
bit_collision_free_MF
=====================
Generate molecular fingerprints guaranteed to be free of bit‑collisions.

本顶层模块保持“轻量”——只暴露版本号，并用**懒加载**方式把真正依赖
NumPy / RDKit 的实现延后到第一次调用时再导入，
从而让 setuptools 在构建阶段能够顺利读取 __version__ 而不会缺库。
"""

from importlib import import_module
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# 版本号：放在一个没有第三方依赖的独立文件 `__version__.py` 中
# ---------------------------------------------------------------------------
from .__version__ import __version__   # noqa: F401  (供外部和构建系统读取)

# ---------------------------------------------------------------------------
# 懒加载包装器
# ---------------------------------------------------------------------------
def _fp():
    """按需导入内部 `fingerprint` 子模块。"""
    return import_module(".fingerprint", __name__)


def CollisionFreeMorganFP(*args, **kwargs):  # type: ignore[N802]
    """Lazy proxy to fingerprint.CollisionFreeMorganFP"""
    return _fp().CollisionFreeMorganFP(*args, **kwargs)


def generate_fingerprints(*args, **kwargs):
    """Lazy proxy to fingerprint.generate_fingerprints"""
    return _fp().generate_fingerprints(*args, **kwargs)


def save_fingerprints(*args, **kwargs):
    """Lazy proxy to fingerprint.save_fingerprints"""
    return _fp().save_fingerprints(*args, **kwargs)


# ---------------------------------------------------------------------------
# 兼顾类型检查：静态分析工具看到完整符号表
# ---------------------------------------------------------------------------
if TYPE_CHECKING:  # pragma: no cover
    from .fingerprint import (  # noqa: F401
        CollisionFreeMorganFP as _CollisionFreeMorganFP,
        generate_fingerprints as _generate_fingerprints,
        save_fingerprints as _save_fingerprints,
    )

# 对外公开 API
__all__ = [
    "CollisionFreeMorganFP",
    "generate_fingerprints",
    "save_fingerprints",
    "__version__",
]
