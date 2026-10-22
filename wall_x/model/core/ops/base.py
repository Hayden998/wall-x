"""Base class for operator proxies with lazy backend resolution."""

import importlib
import logging
import threading

logger = logging.getLogger(__name__)

# Class name -> attribute on the ``xcompute`` package. When ``xcompute`` is
# installed (the accelerated kernel package the reference serving stack uses),
# these operators resolve to its kernels; otherwise they fall back to the inline
# CUDA kernels or pure PyTorch. Keeping the names in one table means the ops
# modules themselves stay backend-agnostic.
_XCOMPUTE_NAMES = {
    "AdaRMSNormOp": "adarmsnorm",
    "AddRMSNormOp": "add_rmsnorm",
    "GatedRMSNormOp": "gated_rmsnorm",
    "GetRopeIndexOp": "get_rope_index",
    "GetWindowIndexOp": "get_window_index",
    "LayerNormOp": "layernorm",
    "MRoPEOp": "m_rope",
    "PermuteOp": "permute",
    "PosEmbedInterpOp": "pos_embed_interp",
    "Qwen35GetRopeIndexOp": "qwen35_get_rope_index",
    "RMSNormOp": "rmsnorm",
    "RoPEOp": "rope",
    "RotPosEmbOp": "rot_pos_emb",
    "SiluAndMulOp": "silu_and_mul",
    "TwoExpertPackOp": "two_expert_pack",
    "TwoExpertUnpackOp": "two_expert_unpack",
    "UnpermuteOp": "unpermute",
}


_ACCEL_MODULE_NAMES = ("xcompute", "external_accel")


def _load_accel_module():
    """Import the accelerated-kernel package under whichever name it ships as."""
    for name in _ACCEL_MODULE_NAMES:
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


class OpsProxy:
    """Callable proxy that lazily resolves to the best available backend.

    Fallback order:
      Level 1: xcompute kernel, when the optional package is installed
      Level 2: CUDA inline kernel, when available
      Level 3: Pure PyTorch fallback
    """

    def __init__(self):
        self._resolved_fn = None
        self._backend = None
        self._resolve_lock = threading.Lock()

    def _resolve(self):
        """Resolve the best available backend. Called once on first use."""
        if self._resolved_fn is not None:
            return
        with self._resolve_lock:
            if self._resolved_fn is not None:
                return

            xcompute_fn = self._probe_xcompute()
            if xcompute_fn is not None:
                self._backend = "xcompute"
                self._resolved_fn = xcompute_fn
                logger.info("%s: using xcompute kernel", self.__class__.__name__)
                return

            cuda_fn = self._get_cuda_kernel()
            if cuda_fn is not None:
                self._backend = "cuda_inline"
                self._resolved_fn = cuda_fn
                return

            pytorch_fn = self._get_pytorch_fallback()
            if pytorch_fn is None:
                raise RuntimeError(
                    f"{self.__class__.__name__}: no available operator backend. "
                    f"Available: {self.available_backends()}"
                )
            self._backend = "pytorch"
            self._resolved_fn = pytorch_fn
            logger.info("%s: using PyTorch fallback", self.__class__.__name__)

    def _get_cuda_kernel(self):
        """Override in subclasses to provide a CUDA inline kernel."""
        return None

    @property
    def _xcompute_name(self):
        """Kernel attribute for this operator, or None to skip the fused backend.

        A subclass may declare ``_external_accel_name``; otherwise the central
        table is used.
        """
        name = getattr(self, "_external_accel_name", None)
        if name:
            return name
        return _XCOMPUTE_NAMES.get(type(self).__name__)

    def _probe_external_accel(self):
        """Raw lookup of this op's kernel in the accelerated package."""
        name = self._xcompute_name
        if not name:
            return None
        module = _load_accel_module()
        if module is None:
            return None
        return getattr(module, name, None)

    def _probe_xcompute(self):
        """Return the kernel for this op, or None when unavailable.

        Subclasses wrap the raw kernel by overriding ``_probe_external_accel``
        (signature compatibility); dispatch through it so those wrappers run.
        """
        return self._probe_external_accel()

    def has_xcompute(self) -> bool:
        """Public capability query: is an xcompute kernel available for this op?"""
        return self._probe_xcompute() is not None

    def _pytorch_fallback(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} has no PyTorch fallback")

    def _get_pytorch_fallback(self):
        """Return the PyTorch fallback only when a subclass implements one."""
        if type(self)._pytorch_fallback is OpsProxy._pytorch_fallback:
            return None
        return self._pytorch_fallback

    def __call__(self, *args, **kwargs):
        if self._resolved_fn is None:
            self._resolve()
        return self._resolved_fn(*args, **kwargs)

    @property
    def backend(self) -> str:
        if self._resolved_fn is None:
            self._resolve()
        return self._backend

    def has_external_accel(self) -> bool:
        """Public capability query: is an accelerated kernel available for this op?"""
        return self._probe_xcompute() is not None or self._get_cuda_kernel() is not None

    def available_backends(self):
        """Return available backend names for this operator."""
        backends = []
        if self._probe_xcompute() is not None:
            backends.append("xcompute")
        if self._get_cuda_kernel() is not None:
            backends.append("cuda_inline")
        if self._get_pytorch_fallback() is not None:
            backends.append("pytorch")
        return backends

    def _get_backend_fn(self, backend):
        """Return the callable for a specific backend, or None if unavailable."""
        if backend == "pytorch":
            return self._get_pytorch_fallback()
        if backend == "xcompute":
            return self._probe_xcompute()
        if backend == "cuda_inline":
            return self._get_cuda_kernel()
        raise ValueError(f"Unknown backend: {backend}")

    def call_with_backend(self, backend, *args, **kwargs):
        """Call this operator using a specific backend."""
        fn = self._get_backend_fn(backend)
        if fn is None:
            raise RuntimeError(
                f"{self.__class__.__name__}: backend '{backend}' not available. "
                f"Available: {self.available_backends()}"
            )
        return fn(*args, **kwargs)

    def __repr__(self):
        backend = self._backend or "unresolved"
        return f"<{self.__class__.__name__} backend={backend}>"
