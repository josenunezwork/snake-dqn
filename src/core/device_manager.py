"""Centralized device management with M1 optimization."""

import os
from typing import Optional

import torch


class DeviceManager:
    """Singleton device manager for consistent device handling across the codebase.

    Provides centralized device management with support for test overrides.

    Usage:
        # Normal usage (auto-detects best device)
        device = DeviceManager.get_device()

        # Testing usage (override device)
        DeviceManager.override_device(torch.device('cpu'))
        # ... run tests ...
        DeviceManager.reset_for_testing()
    """

    _instance: Optional["DeviceManager"] = None
    _device: Optional[torch.device] = None
    _initialized: bool = False
    _silent: bool = False  # Suppress print messages after first init
    _override_device: Optional[torch.device] = None  # For testing

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize device manager with optimal settings."""
        if not self._initialized:
            self._detect_and_configure_device()
            self._initialized = True
            DeviceManager._silent = True  # Suppress future prints

    def _detect_and_configure_device(self) -> None:
        """Detect best available device and configure optimizations."""
        env_device = os.getenv("SNAKE_DQN_DEVICE", "")
        requested_device = (env_device or "").strip().lower()
        if requested_device and requested_device != "auto":
            if requested_device == "cpu":
                self._device = torch.device("cpu")
                self._device_type = "cpu"
                if not DeviceManager._silent:
                    print("🖥️  Using CPU (forced)")
            elif requested_device == "mps":
                if torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                    self._device_type = "mps"
                    # M1-specific optimizations
                    torch.set_num_threads(4)  # M1 works best with 4 threads
                    if not DeviceManager._silent:
                        print("🚀 Using M1 Metal Performance Shaders (MPS)")
                        print("💡 M1-optimized: 4 threads, unified memory")
                else:
                    if not DeviceManager._silent:
                        print(
                            "⚠️  SNAKE_DQN_DEVICE=mps requested but MPS is unavailable; "
                            "falling back to auto-detection."
                        )
            elif requested_device == "cuda":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                    self._device_type = "cuda"
                    if not DeviceManager._silent:
                        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
                else:
                    if not DeviceManager._silent:
                        print(
                            "⚠️  SNAKE_DQN_DEVICE=cuda requested but CUDA is unavailable; "
                            "falling back to auto-detection."
                        )
            else:
                if not DeviceManager._silent:
                    print(
                        f"⚠️  Unrecognized SNAKE_DQN_DEVICE='{env_device}'; "
                        "falling back to auto-detection."
                    )

            if self._device is not None:
                return
        # Auto-detection: prefer CUDA, then CPU. MPS is intentionally NOT
        # auto-selected: for this small 58->512->256 model, MPS kernel-launch /
        # transfer overhead makes it ~5x slower than CPU on Apple Silicon.
        # Request it explicitly with --device mps / SNAKE_DQN_DEVICE=mps.
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            self._device_type = "cuda"
            if not DeviceManager._silent:
                print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            self._device = torch.device("cpu")
            self._device_type = "cpu"
            if not DeviceManager._silent:
                if torch.backends.mps.is_available():
                    print(
                        "🖥️  Using CPU (MPS available but slower for this small model; "
                        "use --device mps or SNAKE_DQN_DEVICE=mps to force MPS)"
                    )
                else:
                    print("🖥️  Using CPU")

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        return self._device

    @property
    def device_type(self) -> str:
        """Get the device type as string."""
        return self._device_type

    @property
    def is_mps(self) -> bool:
        """Check if using MPS (M1)."""
        return self._device_type == "mps"

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA."""
        return self._device_type == "cuda"

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU."""
        return self._device_type == "cpu"

    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to the managed device."""
        return tensor.to(self._device)

    @classmethod
    def get_device(cls) -> torch.device:
        """
        Get the current device.

        If an override device is set (for testing), returns that instead
        of the auto-detected device.

        This is the preferred way to get the device in policies and models.

        Returns:
            torch.device: The optimal device for the current system,
                          or the override device if set.
        """
        if cls._override_device is not None:
            return cls._override_device
        instance = cls()
        return instance.device

    @classmethod
    def override_device(cls, device: Optional[torch.device]) -> None:
        """
        Override the device for testing purposes.

        Call with None to clear the override and return to auto-detection.

        Args:
            device: The device to use, or None to clear override
        """
        cls._override_device = device

    @classmethod
    def reset_for_testing(cls) -> None:
        """
        Reset the singleton state for clean test isolation.

        This clears the singleton instance, initialized flag, and any
        device override. Use this in test teardown to ensure clean state.
        """
        cls._instance = None
        cls._device = None
        cls._initialized = False
        cls._silent = False
        cls._override_device = None

    @classmethod
    def is_override_active(cls) -> bool:
        """Check if a device override is currently active."""
        return cls._override_device is not None

    def __repr__(self) -> str:
        override_str = f", override={self._override_device}" if self._override_device else ""
        return f"DeviceManager(device={self._device}, type={self._device_type}{override_str})"
