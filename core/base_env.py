"""General abstract base classes and observation utilities for benchmark environments.

Public API
----------
``ExtendedTimeStep``
    A :class:`typing.NamedTuple` that extends the standard ``dm_env.TimeStep``
    with an explicit ``action`` field.  Used as the canonical return type of
    :meth:`BaseEnv.reset` and :meth:`BaseEnv.step` when ``output_mode="timestep"``.

``PixelStack``
    Accumulates rendered pixel frames (RGB / depth / segmentation / normal)
    into a fixed-length sliding window and returns a stacked array ready for
    the agent's convolutional encoder.

``ActionStack``
    Maintains a rolling history of applied actions.  When ``output_mode``
    is ``"timestep"``, :class:`BaseEnv` appends each action automatically and
    exposes it through :attr:`ExtendedTimeStep.action`.

``BaseEnv``
    Abstract :class:`gymnasium.Env` subclass that centralises the
    ``output_mode`` switch and optional action-stack integration.
    Concrete environments implement the private ``_reset()`` / ``_step()``
    hooks; the public ``reset()`` / ``step()`` apply the requested output
    format transparently.

``TimeStepWrapper``
    Drop-in :class:`gymnasium.Wrapper` that converts any existing Gymnasium
    environment to return :class:`ExtendedTimeStep` objects, without requiring
    the ``acme`` library.
"""

import json
import os

import cv2
import numpy as np
import gymnasium as gym

from abc import ABC, abstractmethod
from collections import deque
from dm_env import specs, StepType
from gymnasium import spaces
from typing import Any, NamedTuple, Optional, Dict, List
from numpy.typing import NDArray



# ---------------------------------------------------------------------------
# ExtendedTimeStep
# ---------------------------------------------------------------------------

class ExtendedTimeStep(NamedTuple):
    """Standard ``dm_env.TimeStep`` augmented with an explicit ``action`` field.

    This is the canonical timestep type for agents that condition on action
    history (e.g. DrQv2, TACO).  The ``observation`` field carries only the
    raw pixel / state observation; the action history lives in ``action``.

    Supports both attribute access and string-keyed ``[]`` access so that
    :class:`~agents.drqv2.replay_buffer.ReplayBufferStorage` can look up
    fields by spec name.
    """

    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self) -> bool:
        return self.step_type == StepType.FIRST

    def mid(self) -> bool:
        return self.step_type == StepType.MID

    def last(self) -> bool:
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        return tuple.__getitem__(self, attr)

    def replace(self, **kwargs) -> "ExtendedTimeStep":
        return self._replace(**kwargs)


# ---------------------------------------------------------------------------
# PixelStack
# ---------------------------------------------------------------------------

class PixelStack:
    """Accumulates rendered pixel frames into a fixed-length sliding window.

    Frames are stored in a :class:`collections.deque` of length ``size``; on
    the first frame the deque is filled with copies of that frame so that the
    stack is never under-populated.

    Parameters
    ----------
    dtype:
        List of channel types to track, e.g. ``["rgb", "depth"]``.
        Valid keys: ``"rgb"``, ``"depth"``, ``"segmentation"``, ``"normal"``.
    pre_aug:
        ``(H, W)`` target shape for each frame (``cv2.resize`` with ``INTER_AREA``).
    size:
        Number of consecutive frames to stack.
    """

    _CHANNELS = ("rgb", "depth", "segmentation", "normal")

    def __init__(self, dtype: list, pre_aug: list, size: int) -> None:
        self.dtype = {k: k in dtype for k in self._CHANNELS}
        self._dtype = {k: v for k, v in self.dtype.items() if v}
        self._size = size
        self._pre_aug = pre_aug
        self._stack: dict = {}
        self._reset()

    def _reset(self) -> None:
        for k in self._dtype:
            self._stack[k] = deque([], maxlen=self._size)

    def append(self, obs: tuple, is_reset: bool) -> None:
        """Push a new rendered frame tuple onto the stack.

        Parameters
        ----------
        obs:
            ``(rgb, depth, segmentation, normal)`` tuple as returned by the
            Genesis camera render call.  Entries that are not tracked are
            ignored.
        is_reset:
            When ``True``, clears the deque before appending so the history
            does not bleed across episodes.
        """
        if is_reset:
            self._reset()

        raw = dict(zip(self._CHANNELS, obs))
        h, w = self._pre_aug
        for k in self._dtype:
            # spec pipeline-pixels: redimensiona a imagem inteira (o np.resize só copiava os primeiros 3·H·W valores,
            # ~0,8% de uma câmera 1280×720) e passa de (H, W, 3) para (3, H, W)
            img = np.asarray(raw[k])
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=-1)
            img = cv2.resize(img[..., :3].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            img = img.transpose(2, 0, 1)
            if k in ("rgb", "segmentation"):
                img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img = np.ascontiguousarray(img, dtype=np.float32)
            self._stack[k].append(img)
            if len(self._stack[k]) == 1:
                self._stack[k] = deque(list(self._stack[k]) * self._size, maxlen=self._size)

    def built_stack(self) -> dict:
        """Return a dict mapping channel name → stacked array (concatenated on axis 0)."""
        return {k: np.concatenate(list(v), axis=0) for k, v in self._stack.items()}


# ---------------------------------------------------------------------------
# ActionStack
# ---------------------------------------------------------------------------

class ActionStack:
    """Maintains a rolling history of the last ``shape[0]`` applied actions.

    Parameters
    ----------
    shape:
        ``(stack_size, action_dim)`` — number of frames to keep and the
        dimensionality of a single action vector.

    The stack is initialised with zero vectors and is always full, so
    :meth:`build_stack` is safe to call immediately after construction or
    after :meth:`reset`.
    """

    def __init__(self, shape: tuple) -> None:
        self._shape = shape
        self.reset()

    def reset(self) -> None:
        """Fill the history with zero-action vectors."""
        self._stack: deque = deque(
            [np.zeros(self._shape[1], dtype=np.float32)] * self._shape[0],
            maxlen=self._shape[0],
        )

    def append(self, action: np.ndarray) -> None:
        """Push a new action onto the history, dropping the oldest."""
        self._stack.append(np.asarray(action, dtype=np.float32).flatten())

    def build_stack(self) -> np.ndarray:
        """Return a flattened ``(stack_size * action_dim,)`` action history array."""
        return np.concatenate(list(self._stack), axis=0)


# ---------------------------------------------------------------------------
# BaseEnv
# ---------------------------------------------------------------------------

class BaseEnv(gym.Env, ABC):
    """Abstract base for all environments in this benchmark.

    Subclasses implement ``_reset()`` and ``_step()`` with any simulator.
    Two orthogonal options control the public interface:

    ``output_mode``
        ``"gym"`` (default)
            - ``reset()`` → ``(obs, info)``
            - ``step()``  → ``(obs, reward, terminated, truncated, info)``

        ``"timestep"``
            - ``reset()`` → ``(ExtendedTimeStep, info)``
            - ``step()``  → ``(ExtendedTimeStep, info)``

        The ``observation`` field of :class:`ExtendedTimeStep` carries only
        the raw pixel / state observation.  The ``action`` field carries the
        action (history) so the two concerns are cleanly separated.

    ``action_stack_shape``
        When provided as ``(stack_size, action_dim)``, an :class:`ActionStack`
        is created and maintained automatically:

        - The stack is reset to zeros on every :meth:`reset` call.
        - Each action passed to :meth:`step` is appended **after** the physics
          step returns, so ``ExtendedTimeStep.action`` at time *t* contains
          ``[a_{t-k+1}, …, a_t]``.

        In ``"gym"`` mode the stack is still maintained; access it directly
        via ``env._action_stack.build_stack()``.
    """

    metadata: dict = {}

    def __init__(
        self,
        output_mode: str = "gym",
        action_stack_shape: Optional[tuple] = None,
    ) -> None:
        if output_mode not in ("gym", "timestep"):
            raise ValueError(
                f"output_mode must be 'gym' or 'timestep', got '{output_mode!r}'"
            )
        self._output_mode = output_mode
        self._action_stack: Optional[ActionStack] = (
            ActionStack(action_stack_shape) if action_stack_shape is not None else None
        )

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _reset(self) -> tuple[Any, dict]:
        """Simulator-specific reset.

        Returns:
            obs:  Initial observation (matches ``observation_space``).
            info: Auxiliary diagnostic dictionary.
        """

    @abstractmethod
    def _step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Simulator-specific step.

        Returns:
            obs, reward, terminated, truncated, info
        """

    # ------------------------------------------------------------------
    # Public gym.Env interface
    # ------------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        obs, info = self._reset()
        if self._action_stack is not None:
            self._action_stack.reset()
        if self._output_mode == "timestep":
            action = (
                self._action_stack.build_stack()
                if self._action_stack is not None
                else np.zeros(self.action_space.shape, dtype=np.float32).flatten()
            )
            return ExtendedTimeStep(
                step_type=StepType.FIRST,
                reward=0.0,
                discount=1.0,
                observation=obs,
                action=action,
            ), info
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self._step(action)
        if self._action_stack is not None:
            self._action_stack.append(_action_to_numpy(action))
        if self._output_mode == "timestep":
            reward = float(reward.item() if hasattr(reward, "item") else reward)
            action_arr = (
                self._action_stack.build_stack()
                if self._action_stack is not None
                else _action_to_numpy(action)
            )
            if terminated:
                step_type, discount = StepType.LAST, 0.0
            elif truncated:
                step_type, discount = StepType.LAST, 1.0
            else:
                step_type, discount = StepType.MID, 1.0
            return ExtendedTimeStep(
                step_type=step_type,
                reward=reward,
                discount=discount,
                observation=obs,
                action=action_arr,
            ), info
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # dm_env spec helpers (relevant when output_mode == "timestep")
    # ------------------------------------------------------------------

    def observation_spec(self) -> specs.Array:
        """dm_env-compatible observation spec, derived from ``observation_space``."""
        return _space_to_spec(self.observation_space, name="observation")

    def action_spec(self) -> specs.Array:
        """dm_env-compatible action spec, derived from ``action_space``."""
        return _space_to_spec(self.action_space, name="action")

    @property
    def output_mode(self) -> str:
        return self._output_mode


# ---------------------------------------------------------------------------
# TimeStepWrapper
# ---------------------------------------------------------------------------

class TimeStepWrapper(gym.Wrapper):
    """Wraps any :class:`gymnasium.Env` to return :class:`ExtendedTimeStep`.

    Converts the standard Gymnasium 5-tuple interface to the benchmark's
    extended timestep format without requiring the ``acme`` library.  The
    ``action`` field of each :class:`ExtendedTimeStep` holds the raw action
    (no stacking — add an :class:`ActionStack` inside the wrapped env or use
    :class:`BaseEnv` if you need history).

    Usage::

        env = gym.make("CartPole-v1")
        env = TimeStepWrapper(env)

        ts, info = env.reset()       # ExtendedTimeStep (StepType.FIRST)
        ts, info = env.step(action)  # ExtendedTimeStep (StepType.MID or LAST)
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._reset_next_step = True
        self._zero_action = np.zeros(env.action_space.shape, dtype=np.float32).flatten()

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[ExtendedTimeStep, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._reset_next_step = False
        return ExtendedTimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=obs,
            action=self._zero_action,
        ), info

    def step(self, action: Any) -> tuple[ExtendedTimeStep, dict]:
        if self._reset_next_step:
            return self.reset()

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._reset_next_step = terminated or truncated

        reward = float(reward)
        action_arr = _action_to_numpy(action)
        if terminated:
            step_type, discount = StepType.LAST, 0.0
        elif truncated:
            step_type, discount = StepType.LAST, 1.0
        else:
            step_type, discount = StepType.MID, 1.0
        return ExtendedTimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=obs,
            action=action_arr,
        ), info

    def observation_spec(self) -> specs.Array:
        return _space_to_spec(self.observation_space, name="observation")

    def action_spec(self) -> specs.Array:
        return _space_to_spec(self.action_space, name="action")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _action_to_numpy(action: Any) -> np.ndarray:
    """Convert any action type to a flat ``float32`` numpy array."""
    if hasattr(action, "detach"):  # torch.Tensor
        return action.detach().cpu().numpy().flatten()
    return np.asarray(action, dtype=np.float32).flatten()


def _space_to_spec(space: gym.Space, name: Optional[str] = None):
    """Convert a :class:`gymnasium.Space` to a ``dm_env`` spec or nested specs."""
    if isinstance(space, spaces.Discrete):
        return specs.DiscreteArray(
            num_values=int(space.n), dtype=space.dtype, name=name
        )

    if isinstance(space, spaces.Box):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=space.low,
            maximum=space.high,
            name=name,
        )

    if isinstance(space, spaces.MultiBinary):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=0.0,
            maximum=1.0,
            name=name,
        )

    if isinstance(space, spaces.MultiDiscrete):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=np.zeros(space.shape),
            maximum=space.nvec - 1,
            name=name,
        )

    if isinstance(space, spaces.Tuple):
        return tuple(_space_to_spec(s, name) for s in space.spaces)

    if isinstance(space, spaces.Dict):
        return {key: _space_to_spec(value, key) for key, value in space.spaces.items()}

    raise ValueError(f"Unsupported gym space type: {type(space)}")


class BiguaGymEnv(BaseEnv):
    """Abstract base for BiguaGym-simulator environments.

    Parameters
    ----------
    seed:
        RNG seed forwarded to the underlying environment on each reset.
    env_params:
        Environment-level config dict (episode length, clip_actions,
        timestep flag, etc.).  Stored as ``self.env_cfg``.
    obs_params:
        Observation config dict (type, stack, pre_aug_shape, etc.).
        Stored as ``self.obs_cfg``.
    output_mode:
        ``"gym"`` (default) or ``"timestep"`` — see :class:`BaseEnv`.
    show_viewer:
        Open an interactive viewer window.
    action_stack_shape:
        Optional ``(stack_size, action_dim)`` for action-history stacking.
    render_mode:
        ``"rgb_array"`` to enable :meth:`render` (compatible with
        ``gym.wrappers.RecordVideo``).  ``None`` disables rendering.
    """

    metadata: dict = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        seed: int,
        env_params: dict,
        obs_params: dict,
        output_mode: str = "gym",
        show_viewer: bool = False,
        action_stack_shape: Optional[tuple] = None,
        render_mode: Optional[str] = None,
    ) -> None:

        super().__init__(output_mode=output_mode, action_stack_shape=action_stack_shape)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']!r}, "
                f"got {render_mode!r}"
            )

        self.seed = seed
        self.env_cfg = env_params
        self.obs_cfg = obs_params
        self.show_viewer = show_viewer
        self.render_mode = render_mode

        self.rng = np.random.default_rng(seed=seed)

        self._last_render_frame: Optional[NDArray] = None
        self._recording: bool = False
        self._record_writer = None  # cv2.VideoWriter, created lazily on first frame
        self._record_path: Optional[str] = None
        self._record_fps: int = 20

        self._env = self._build_env()
        self._init_spaces()

    # ------------------------------------------------------------------
    # Subclass contract — construction hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_env(self) -> gym.Env:
        """Instantiate and return the underlying BiguaSim environment.

        Example::

            return biguasim.make(scenario_cfg=cfg, show_viewport=False)
        """

    @abstractmethod
    def _init_spaces(self) -> None:
        """Set ``self.observation_space`` and ``self.action_space``.

        Called after ``_build_env()``, so ``self._env`` is available.

        Example (state-based)::

            self.observation_space = self._env.observation_space
            self.action_space      = self._env.action_space
        """

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _agent_id(self, env_cfg : dict, agent_name : str) -> int :
        for i in range(len(env_cfg['agents'])):
            agent = env_cfg['agents'][i]
            if agent['agent_name'] == agent_name:
                return i
    
    def _load_config(self, file_name : str) -> Dict[str, Any]:
        with open(file_name, "r") as f:
            return json.load(f)
    # ------------------------------------------------------------------
    # BaseEnv contract
    # ------------------------------------------------------------------

    @abstractmethod
    def _wrap_state(self, state : dict) -> NDArray:
        """Process the complete BiguaSim state; return ``obs``."""

    @abstractmethod
    def _reward(self) -> float | NDArray:
        """Process the environment reward; return ``reward``."""

    @abstractmethod
    def _reset(self) -> tuple[Any, dict]:
        """Execute the simulation resets; return ``(obs, info)``."""

    @abstractmethod
    def _step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Execute one physics step; return ``(obs, reward, terminated, truncated, info)``."""

    # ------------------------------------------------------------------
    # Rendering and recording
    # ------------------------------------------------------------------

    def render(self) -> Optional[NDArray]:
        """Return the latest RGB frame from ``CameraView`` as a uint8 array.

        Compatible with ``gym.wrappers.RecordVideo`` when the env is
        constructed with ``render_mode="rgb_array"``.  Returns ``None``
        if no frame has been captured yet or ``render_mode`` is not set.
        """
        if self.render_mode != "rgb_array" or self._last_render_frame is None:
            return None
        frame = self._last_render_frame
        if self._recording:
            self._write_frame(frame)
        return frame

    def start_recording(self, path: str, fps: int = 20) -> None:
        """Begin writing rendered frames to a video file with ``cv2.VideoWriter``.

        Parameters
        ----------
        path:
            Output file path (e.g. ``"episode.mp4"``).
        fps:
            Frames per second of the output video.
        """
        import cv2  # noqa: F401 — validate import early
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._record_path = path
        self._record_fps = fps
        self._record_writer = None  # created lazily on first frame (need frame shape)
        self._recording = True

    def _write_frame(self, frame: NDArray) -> None:
        import cv2
        if self._record_writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._record_writer = cv2.VideoWriter(
                self._record_path, fourcc, self._record_fps, (w, h)
            )
        # cv2 expects BGR
        self._record_writer.write(frame)

    def stop_recording(self) -> None:
        """Flush and close the video file.

        Does nothing if recording was never started.
        """
        if not self._recording:
            return
        self._recording = False
        if self._record_writer is not None:
            self._record_writer.release()
            self._record_writer = None

    def close(self) -> None:
        if self._recording:
            self.stop_recording()
        del self._env



