import biguasim

import numpy as np

from typing import List
from numpy.typing import NDArray
from operator import itemgetter
from gymnasium import spaces
from pathlib import Path
from scipy.interpolate import splprep, splev

from .base_env import BiguaGymEnv, PixelStack
from .space import DATA_REGISTRY

CONFIG = f'{Path(__file__).resolve().parent.parent}/config'

DOMAIN = {
    'aereo' : ['DjiMatrice'],
    'surface' : ['BlueBoat'],
    'underwater' : ['BlueROV2', 'BlueROVHeavy', 'TorpedoAUV'],
    'multi-domain' : ['Hydrone']
}


class HoverEnv(BiguaGymEnv):
    """3-D hover task: reach and hold a randomly sampled target position.

    The target is drawn uniformly from a ±10 m box around the spawn location.
    Reward combines exponential proximity, distance-improvement smoothing,
    attitude stability (roll/pitch), and a yaw-rate penalty. Episodes terminate
    on roll/pitch > 15°, out-of-bounds, or on-target success. A coarse
    curriculum via ``target_factor`` resamples the target only every N successes.
    """

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        render_mode: str = None,
        reward_version: str = "v1",
    ) -> None:
        output_mode = "timestep" if timestep else "gym"

        # Versão da recompensa (spec recompensas): "v0" = original, "v1" = penalidade de término,
        # bônus de sucesso fixo e Trajectory paga por avanço.
        if reward_version not in ("v0", "v1"):
            raise ValueError(f"reward_version inválido: {reward_version!r} (use 'v0' ou 'v1')")
        self._reward_version = reward_version

        self._agent_type = agent_type
        self._location = location
        self._rotation = rotation
        self._batch_size = batch_size
        self._control_abstraction = control_abstraction
        self._observation_type = observation_type
        self._action_repeat = action_stack
        self._target_factor = target_factor
        self._timestep = timestep

        env_params, obs_params = self._build_params()

        keys = tuple(obs_params.keys())
        # itemgetter with a single arg returns a scalar, not a tuple — wrap so
        # downstream code can always treat the result as an iterable of values.
        _raw_getter = itemgetter(*keys)
        if len(keys) == 1:
            self._getter = lambda d: (_raw_getter(d),)
        else:
            self._getter = _raw_getter

        self._dynamics = None
        self._last_norm = None
        self._episode_steps = 0
        self._on_target = False
        self._on_target_buf = 0  # cross-episode success counter (drives target_factor curriculum)
        self._target = None
        self._target_list = []

        # action_stack_shape requires the underlying env's action dim, which is
        # only known after _build_env(). Initialize without it, then inject.
        super().__init__(seed, env_params, obs_params, output_mode, show_viewer, None, render_mode)

        if timestep:
            from .base_env import ActionStack
            self._action_stack = ActionStack(
                (action_stack, int(self._env.action_space.shape[0]))
            )

    @property
    def max_episode_steps(self) -> int:
        return 400  # BiguaGym default

    def _build_params(self):
        env_params: dict = self._load_config(f"{CONFIG}/state.json")

        _id = self._agent_id(env_params, 'robot')

        env_params['agents'][_id]['agent_type'] = self._agent_type
        env_params['agents'][_id]['control_abstraction'] = self._control_abstraction
        env_params['agents'][_id]['location'] = self._location
        env_params['agents'][_id]['rotation'] = self._rotation
        env_params['agents'][_id]['dynamics']['batch_size'] = self._batch_size

        loc = np.asarray(self._location, dtype=np.float32)
        self._bounds = np.array([loc - 10.0, loc + 10.0])

        # Keep z-min above ground.
        self._bounds[0, 2] = max(float(self._bounds[0, 2]), 0.1)
        self._bounds[1, 2] = max(float(loc[2]), 2.0 * self._bounds[0, 2])

        if isinstance(self._observation_type, str):
            self._observation_type = [self._observation_type]

        obs_params = {
            obs_type: np.zeros(DATA_REGISTRY[obs_type]).ravel().shape
            for obs_type in self._observation_type
        }

        return env_params.copy(), obs_params.copy()
    

    def _build_env(self):
        return biguasim.make(scenario_cfg=self.env_cfg, show_viewport=self.show_viewer)

    def _wrap_state(self, state: dict) -> NDArray:
        raw = state.get('CameraView')
        if raw is not None:
            self._last_render_frame = np.asarray(raw, dtype=np.uint8)[:, :, 0:3]
        return np.concatenate([
            np.asarray(v, dtype=np.float32).ravel()
            for v in self._getter(state)
        ])

    @staticmethod
    def _unwrap_state(state: dict) -> dict:
        # PATCH TEMPORÁRIO (testes-ambientes): biguasim 1.0.0 devolve
        # {'robot': [ {sensores} ], 't': t}; o restante do código espera o dict plano de sensores.
        # Estado já plano passa sem mudança.
        robot = state.get('robot')
        if isinstance(robot, list) and robot and isinstance(robot[0], dict):
            return robot[0]
        return state

    def _set_dynamics(self, state: dict) -> None:
        # RPYDynamicsSensor reports roll/pitch/yaw (indices 15:) in degrees;
        # convert here so every downstream radian comparison/trig call is correct.
        dynamics = np.asarray(state['RPYDynamicsSensor'], dtype=np.float64).ravel()
        dynamics[15:] = np.radians(dynamics[15:])
        self._dynamics = dynamics

    def _reset(self):
        state = self._unwrap_state(self._env.reset())
        self._set_dynamics(state)
        self._episode_steps = 0
        self._last_norm = None
        self._on_target = False

        if self._on_target_buf % self._target_factor == 0:
            self._target : NDArray = self.rng.uniform(low=self._bounds[0], high=self._bounds[1])
            self._target_list = self._target.tolist()

        return self._wrap_state(state), {}

    def _init_spaces(self) -> None:
        total_dim = sum(int(np.prod(shape)) for shape in self.obs_cfg.values())
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32,
        )
        # PATCH TEMPORÁRIO (testes-ambientes): com cmd_motor_speeds a dinâmica usa a ação direto
        # como velocidade dos rotores, então o Box vem dos parâmetros do modelo de dinâmica
        # (nº de rotores e limites), não do esquema Unreal do agente (ex.: BlueBoat → [vx, vy, vz]).
        dyn = getattr(self._env, '_dynamics_dict', {}).get('robot')
        dyn_params = getattr(dyn, 'params', None) or {}
        if self._control_abstraction == 'cmd_motor_speeds' and dyn_params.get('rotor_pos'):
            self.action_space = spaces.Box(
                low=float(dyn_params['rotor_speed_min']),
                high=float(dyn_params['rotor_speed_max']),
                shape=(len(dyn_params['rotor_pos']),),
                dtype=np.float32,
            )
            return
        self.action_space = spaces.Box(
            low=float(self._env.action_space.get_low()[0]),
            high=float(self._env.action_space.get_high()[0]),
            shape=(int(self._env.action_space.shape[0]),),
            dtype=np.float32,
        )

    def _reward(self):
        target = self._target
        pos = np.asarray(self._dynamics[6:9])
        ang_vel = np.asarray(self._dynamics[12:15])
        rpy = np.asarray(self._dynamics[15:])

        norm = np.linalg.norm(target - pos)

        self._on_target = abs(norm) <= 0.15
        if self._on_target:
            self._on_target_buf += 1

        norm_reward = np.exp(-2.0 * norm)

        if self._last_norm is None:
            self._last_norm = norm

        smooth_reward = 3 * (self._last_norm - norm)
        self._last_norm = norm

        rpy_magnitude = np.linalg.norm(rpy[:2])
        rpy_ratio = np.clip(rpy_magnitude / np.radians(35), 0.0, 1.0)
        stable_reward = -(rpy_ratio ** 2) * 3.0

        yaw_rate = abs(ang_vel[2])
        spin_reward = -yaw_rate * 0.2

        return norm_reward + smooth_reward + stable_reward + spin_reward
    
    def _env_constraints(self) -> None:
        self._env.draw_point(self._target_list)

    def _step(self, action: NDArray) -> tuple:
        self._env_constraints()
        self._episode_steps += self._action_repeat

        flat = np.asarray(action, dtype=np.float32).ravel()
        if self._batch_size > 1:
            action_arg = np.tile(flat, (self._batch_size, 1)).tolist()
        else:
            action_arg = flat.tolist()

        state = self._unwrap_state(self._env.step(action_arg, action_repeat=self._action_repeat))
        self._set_dynamics(state)
        obs = self._wrap_state(state)

        pos = np.asarray(self._dynamics[6:9])
        r, p, _ = np.asarray(self._dynamics[15:])

        truncated = self._episode_steps >= self.max_episode_steps

        out_of_bounds = bool(
            np.any((pos < self._bounds[0]) | (pos > self._bounds[1]))
        )
        terminated = bool(
            (abs(r) > np.radians(15))
            or (abs(p) > np.radians(15))
            or out_of_bounds
            or self._on_target
        )

        reward = 3.0 * abs(self._reward()) if self._on_target else self._reward()
        info = {"reached_goals": int(self._on_target_buf)}
        return obs, float(reward), terminated, truncated, info

    def update_target_factor(self, factor: int) -> None:
        self._target_factor = abs(int(factor))



class LandEnv(HoverEnv):
    """Precision landing task: descend to a ground-level target on a landing pad.

    Extends ``HoverEnv`` by constraining the target's z to ground level and
    visualising a 1 m × 1 m landing pad via ``draw_box``. The reward adds
    velocity-aware terms: an impact penalty for excess descent speed and a drift
    penalty for horizontal speed near the pad, both scaled by a proximity weight
    that sharpens as the vehicle approaches. Hard landings (fast descent within
    0.5 m) terminate the episode with a −5 penalty.
    """

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        render_mode: str = None,
        reward_version: str = "v1",
    ) -> None:

        super().__init__(seed, agent_type, control_abstraction, location, rotation, batch_size, observation_type,
                         show_viewer, timestep, action_stack, target_factor, render_mode, reward_version=reward_version)
        
    @property
    def max_episode_steps(self) -> int:
        return 300  # BiguaGym default
    
    def _reset(self):
        state = self._unwrap_state(self._env.reset())
        self._set_dynamics(state)
        self._episode_steps = 0
        self._last_norm = None
        self._on_target = False

        if self._on_target_buf % self._target_factor == 0:
            low = self._bounds[0].copy()
            high = self._bounds[1].copy()
            high[2] = low[2]
            self._target : NDArray = self.rng.uniform(low=low, high=high)
            self._target_list = self._target.tolist()

        return self._wrap_state(state), {}

    def _env_constraints(self) -> None:
        self._env.draw_box(self._target_list, extent=[1.0, 1.0, 0.1], thickness=50)

    def _reward(self):
        target = self._target
        pos = np.asarray(self._dynamics[6:9])
        vel = np.asarray(self._dynamics[3:6])
        ang_vel = np.asarray(self._dynamics[12:15])
        rpy = np.asarray(self._dynamics[15:])

        norm = np.linalg.norm(target - pos)
        self._on_target = bool(norm <= 0.15)
        if self._on_target:
            self._on_target_buf += 1

        norm_reward = np.exp(-2.0 * norm)

        if self._last_norm is None:
            self._last_norm = norm
        smooth_reward = 3.0 * (self._last_norm - norm)
        self._last_norm = norm

        # Proximity weight: 1.0 at pad level, decays with altitude above target.
        # Controls how aggressively landing-specific penalties apply.
        height_above_target = max(float(pos[2] - target[2]), 0.0)
        proximity_weight = float(np.exp(-height_above_target / 3.0))

        # Attitude stability — tolerance tightens (35°→10°) and penalty grows
        # (weight 3→7) as the vehicle descends toward the pad.
        rpy_magnitude = np.linalg.norm(rpy[:2])
        angle_threshold = np.radians(35.0 - 25.0 * proximity_weight)
        rpy_ratio = np.clip(rpy_magnitude / angle_threshold, 0.0, 1.0)
        stable_reward = -(rpy_ratio ** 2) * (3.0 + 4.0 * proximity_weight)

        yaw_rate = abs(ang_vel[2])
        spin_reward = -yaw_rate * 0.2

        # Impact penalty — excess downward velocity above a safe descent rate.
        # Penalises hard falls; allows gentle 0.3 m/s approaches without cost.
        safe_descent = 0.3  # m/s
        descent_speed = max(float(-vel[2]), 0.0)
        excess_descent = max(descent_speed - safe_descent, 0.0)
        impact_penalty = -excess_descent * (1.0 + 3.0 * proximity_weight) * 0.8

        # Drift penalty — penalises horizontal speed near the pad to prevent
        # side-impacts and sliding landings.
        safe_horiz = 0.3  # m/s
        horiz_speed = float(np.linalg.norm(vel[:2]))
        excess_horiz = max(horiz_speed - safe_horiz, 0.0)
        drift_penalty = -excess_horiz * proximity_weight * 0.5

        return (norm_reward + smooth_reward + stable_reward
                + spin_reward + impact_penalty + drift_penalty)

    def _step(self, action: NDArray) -> tuple:
        self._env_constraints()
        self._episode_steps += self._action_repeat

        flat = np.asarray(action, dtype=np.float32).ravel()
        if self._batch_size > 1:
            action_arg = np.tile(flat, (self._batch_size, 1)).tolist()
        else:
            action_arg = flat.tolist()

        state = self._unwrap_state(self._env.step(action_arg, action_repeat=self._action_repeat))
        self._set_dynamics(state)
        obs = self._wrap_state(state)

        pos = np.asarray(self._dynamics[6:9])
        vel = np.asarray(self._dynamics[3:6])
        r, p, _ = np.asarray(self._dynamics[15:])

        truncated = self._episode_steps >= self.max_episode_steps

        out_of_bounds = bool(
            np.any((pos < self._bounds[0]) | (pos > self._bounds[1]))
        )

        # Hard-landing detection: fast descent while close to the pad.
        height_above_target = max(float(pos[2] - self._target[2]), 0.0)
        proximity_weight = float(np.exp(-height_above_target / 3.0))
        descent_speed = max(float(-vel[2]), 0.0)
        hard_landing = bool(proximity_weight > 0.5 and descent_speed > 2.0)

        terminated = bool(
            (abs(r) > np.radians(15))
            or (abs(p) > np.radians(15))
            or out_of_bounds
            or self._on_target
            or hard_landing
        )

        reward = self._reward()
        if self._on_target:
            reward = 3.0 * abs(reward)
        elif hard_landing:
            reward = -5.0

        info = {"reached_goals": int(self._on_target_buf), "hard_landing": hard_landing}
        return obs, float(reward), terminated, truncated, info


class DockEnv(LandEnv):
    """Underwater / sub-surface docking task: descend to a negative-z target.

    Mirrors ``LandEnv`` but inverts the z operating envelope so the agent must
    reach a target below the surface (z < 0). Bounds are tightened to keep the
    vehicle in the sub-surface domain, and out-of-bounds logic checks the z
    axis independently from x/y. Hard-docking detection (fast descent while
    near the dock) terminates with a −5 penalty, matching ``LandEnv``'s
    hard-landing behaviour.
    """

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        render_mode: str = None,
        reward_version: str = "v1",
    ) -> None:

        super().__init__(seed, agent_type, control_abstraction, location, rotation, batch_size, observation_type,
                         show_viewer, timestep, action_stack, target_factor, render_mode, reward_version=reward_version)
        
    
    def _build_params(self):
        env_params, obs_params = super()._build_params()

        loc = np.asarray(self._location, dtype=np.float32)
        self._bounds = np.array([loc - 10.0, loc + 10.0])

        # Keep z-min above ground.
        self._bounds[0, 2] = min(float(self._bounds[0, 2] + 10), -0.2)
        self._bounds[1, 2] = -10

        return env_params, obs_params
    
    def _reset(self):
        state = self._unwrap_state(self._env.reset())
        self._set_dynamics(state)
        self._episode_steps = 0
        self._last_norm = None
        self._on_target = False

        if self._on_target_buf % self._target_factor == 0:
            low = self._bounds[0].copy()
            high = self._bounds[1].copy()
            low[2] = high[2]
            self._target : NDArray = self.rng.uniform(low=low, high=high)
            self._target_list = self._target.tolist()

        return self._wrap_state(state), {}
    
    def _step(self, action: NDArray) -> tuple:
        self._env_constraints()
        self._episode_steps += self._action_repeat

        flat = np.asarray(action, dtype=np.float32).ravel()
        if self._batch_size > 1:
            action_arg = np.tile(flat, (self._batch_size, 1)).tolist()
        else:
            action_arg = flat.tolist()

        state = self._unwrap_state(self._env.step(action_arg, action_repeat=self._action_repeat))
        self._set_dynamics(state)
        obs = self._wrap_state(state)

        pos = np.asarray(self._dynamics[6:9])
        vel = np.asarray(self._dynamics[3:6])
        r, p, _ = np.asarray(self._dynamics[15:])

        truncated = self._episode_steps >= self.max_episode_steps

        out_of_bounds = bool(
            np.any((pos[:2] < self._bounds[0,:2]) | (pos[:2] > self._bounds[1,:2]) | (pos[2] > self._bounds[0,2]) | (pos[2] < self._bounds[1,2]))
        )

        # Hard-docking detection: fast descent while close to the pad.
        height_above_target = max(float(pos[2] - self._target[2]), 0.0)
        proximity_weight = float(np.exp(-height_above_target / 3.0))
        descent_speed = max(float(-vel[2]), 0.0)
        hard_docking = bool(proximity_weight > 0.5 and descent_speed > 2.0)

        terminated = bool(
            (abs(r) > np.radians(15))
            or (abs(p) > np.radians(15))
            or out_of_bounds
            or self._on_target
            or hard_docking
        )

        reward = self._reward()
        if self._on_target:
            reward = 3.0 * abs(reward)
        elif hard_docking:
            reward = -5.0

        info = {"reached_goals": int(self._on_target_buf), "hard_docking": hard_docking}
        return obs, float(reward), terminated, truncated, info
    
class NavEnv(HoverEnv):
    """Domain-aware navigation base: sets z bounds per vehicle type.

    Extends ``HoverEnv`` with operating-envelope constraints derived from the
    agent's domain (aerial, surface, or underwater). Aerial vehicles are kept
    above ground (z ≥ 0); surface vehicles are clamped to ±0.5 m around the
    waterline; underwater vehicles operate in the −20 m to −0.1 m range.
    Intended as a base for path-following tasks such as ``TrajectoryEnv``.
    """

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        render_mode: str = None,
        reward_version: str = "v1",
    ) -> None:

        super().__init__(seed, agent_type, control_abstraction, location, rotation, batch_size, observation_type,
                         show_viewer, timestep, action_stack, target_factor, render_mode, reward_version=reward_version)
        
    
    def _build_params(self):
        env_params, obs_params = super()._build_params()

        loc = np.asarray(self._location, dtype=np.float32)
        self._bounds = np.array([loc - 10.0, loc + 10.0])

        
        if self._agent_type in DOMAIN['aereo']:
            self._bounds[0, 2] = 0.0

        elif self._agent_type in DOMAIN['surface']:
            self._bounds[0, 2] = -1
            self._bounds[1, 2] = 0.1

        elif self._agent_type in DOMAIN['underwater']:
            self._bounds[0, 2] = -20
            self._bounds[1, 2] = -0.1
        

        return env_params, obs_params
    
class TrajectoryEnv(NavEnv):
    """Trajectory-following task for aerial, surface, and underwater agents.

    The agent must track a pre-defined 3-D path as closely as possible.
    Observations are augmented with Frenet-frame errors (cross-track /
    along-track / heading) and a lookahead window of upcoming waypoints
    expressed in body frame, mirroring the design of TrajectoryFollowerEnv.

    Reward composition
    ------------------
    - Tiered CTE zone reward  (exact / near / far bands)
    - Progress reward         (waypoint advancement delta)
    - Heading alignment       (cos of yaw error to path tangent)
    - Attitude stability      (roll / pitch penalty)
    - Angular-velocity smoothness penalty
    - End-of-trajectory bonus on success
    """

    # Reward zone thresholds (metres)
    ZONE_EXACT = 0.5
    ZONE_NEAR  = 2.0
    ZONE_FAR   = 5.0

    # Reward weights
    W_CTE    = 2.0
    W_PROG   = 3.0
    W_ALIGN  = 0.5
    W_STABLE = 2.0
    W_SMOOTH = 0.1
    BONUS_END = 10.0

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        target_trajectory: str | NDArray = 'sine',
        n_lookahead: int = 5,
        waypoint_radius: float = 0.2,
        render_mode: str = None,
        reward_version: str = "v1",
    ) -> None:
        # Must be set before super().__init__() triggers _init_spaces()
        self._n_lookahead = n_lookahead
        self._waypoint_radius = waypoint_radius

        super().__init__(seed, agent_type, control_abstraction, location, rotation,
                         batch_size, observation_type, show_viewer, timestep,
                         action_stack, target_factor, render_mode, reward_version=reward_version)

        # --- Build trajectory (needs self.rng and self._location from super) ---
        if isinstance(target_trajectory, np.ndarray):
            # Caller supplies a complete 3-D (N, 3) trajectory in world frame.
            self.trajectory = self.arc_length_parameterize(
                target_trajectory.astype(np.float32)
            )
        else:
            factory = {
                'sine':    self._sine_trajectory,
                'figure8': self._figure8_trajectory,
                'spiral':  self._spiral_trajectory,
                'random':  self._random_spline_trajectory,
            }
            assert target_trajectory in factory, (
                f"Unknown trajectory '{target_trajectory}'. "
                f"Available: {list(factory.keys())}"
            )
            # All factories return (N, 3) in local frame (z relative to 0).
            # A single offset by spawn location places the path in world frame.
            traj = self.arc_length_parameterize(factory[target_trajectory]())
            loc = np.asarray(self._location, dtype=np.float32)
            self.trajectory = (traj + loc).astype(np.float32)

        self.n_wp: int = len(self.trajectory)

        # x/y bounds: trajectory extents + margin
        margin = 5.0
        self._bounds = np.array([
            self.trajectory.min(axis=0) - margin,
            self.trajectory.max(axis=0) + margin,
        ], dtype=np.float32)

        # z bounds: apply domain-specific constraints (mirrors NavEnv rules) so
        # the OOB termination respects the vehicle's operating envelope.
        if self._agent_type in DOMAIN['aereo']:
            self._bounds[0, 2] = max(float(self._bounds[0, 2]), 0.0)
        elif self._agent_type in DOMAIN['surface']:
            self._bounds[0, 2] = max(float(self._bounds[0, 2]), -0.5)
            self._bounds[1, 2] = min(float(self._bounds[1, 2]),  0.1)
        elif self._agent_type in DOMAIN['underwater']:
            self._bounds[0, 2] = max(float(self._bounds[0, 2]), -20.0)
            self._bounds[1, 2] = min(float(self._bounds[1, 2]),  -0.1)

        self._tangents, self._curvatures = self._precompute_path_geometry()
        self._wp_idx: int = 0
        self._prev_progress: float = 0.0

    @property
    def max_episode_steps(self) -> int:
        return 600

    # ------------------------------------------------------------------
    # Trajectory factory helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sine_trajectory(n: int = 200, amplitude: float = 1.0,
                         frequency: float = 1.0, length: float = 10.0) -> NDArray:
        x = np.linspace(0, length, n)
        y = amplitude * np.sin(frequency * x * 2 * np.pi / length)
        return np.column_stack([x, y, np.zeros(n, dtype=np.float32)]).astype(np.float32)

    @staticmethod
    def _figure8_trajectory(n: int = 300, scale: float = 2.5) -> NDArray:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x = scale * np.sin(t)
        y = scale * np.sin(t) * np.cos(t)
        return np.column_stack([x, y, np.zeros(n, dtype=np.float32)]).astype(np.float32)

    def _spiral_trajectory(self, n: int = 400, max_radius: float = 4.0,
                           turns: float = 3.0, z_range: float = 2.5) -> NDArray:
        if self._agent_type in DOMAIN['surface']:
            raise ValueError(
                f"Spiral trajectory is not available for surface vehicles "
                f"({self._agent_type})."
            )
        loc_z = float(np.asarray(self._location, dtype=np.float32)[2])
        near_surface = abs(loc_z) <= 1.5

        t = np.linspace(0, turns * 2 * np.pi, n)
        r = max_radius * t / (turns * 2 * np.pi)
        progress = t / (turns * 2 * np.pi)  # 0 → 1 along the spiral

        if self._agent_type in DOMAIN['aereo']:
            if near_surface:
                direction = 1  # must ascend when close to ground
            else:
                direction = int(self.rng.choice([-1, 1]))
                # Ensure descent can't go below ground in world frame
                if direction == -1 and loc_z < z_range:
                    direction = 1
            z = direction * z_range * progress

        elif self._agent_type in DOMAIN['underwater']:
            if near_surface:
                direction = -1  # must descend when close to surface
            else:
                direction = int(self.rng.choice([-1, 1]))
                # Ensure ascent can't breach the surface in world frame
                if direction == 1 and abs(loc_z) < z_range + 0.1:
                    direction = -1
            z = direction * z_range * progress

        else:  # multi-domain — no z restriction
            direction = int(self.rng.choice([-1, 1]))
            z = direction * z_range * progress

        return np.column_stack([r * np.cos(t), r * np.sin(t), z]).astype(np.float32)

    @staticmethod
    def arc_length_parameterize(traj: NDArray) -> NDArray:
        """Re-sample any (N, D) trajectory to uniform arc-length spacing."""
        diffs = np.diff(traj, axis=0)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
        uniform = np.linspace(0, arc[-1], len(traj))
        return np.column_stack([
            np.interp(uniform, arc, traj[:, i]) for i in range(traj.shape[1])
        ]).astype(np.float32)

    def _random_spline_trajectory(self, n: int = 50, n_ctrl: int = 12,
                                  extent: float = 5.0, z_range: float = 2.5) -> NDArray:
        if self._agent_type in DOMAIN['surface']:
            raise ValueError(
                f"Random spline trajectory is not available for surface vehicles "
                f"({self._agent_type})."
            )
        ctrl_xy = self.rng.uniform(-extent, extent, size=(n_ctrl, 2))
        ctrl_xy[0] = [0.0, 0.0]  # anchor start to spawn position (local frame)
        if self._agent_type in DOMAIN['aereo']:
            ctrl_z = self.rng.uniform(0.0, z_range, size=(n_ctrl, 1))
        elif self._agent_type in DOMAIN['underwater']:
            ctrl_z = self.rng.uniform(-z_range, 0.0, size=(n_ctrl, 1))
        else:  # multi-domain
            ctrl_z = self.rng.uniform(-z_range / 2, z_range / 2, size=(n_ctrl, 1))
        ctrl_z[0] = [0.0]  # anchor start z to spawn altitude (local frame)
        ctrl = np.hstack([ctrl_xy, ctrl_z])
        ctrl = np.vstack([ctrl, ctrl[0]])  # close loop for periodic spline
        tck, _ = splprep([ctrl[:, 0], ctrl[:, 1], ctrl[:, 2]], s=0, per=True, k=3)
        x, y, z = splev(np.linspace(0, 1, n), tck)
        return np.column_stack([x, y, z]).astype(np.float32)

    # ------------------------------------------------------------------
    # Path geometry
    # ------------------------------------------------------------------

    def _precompute_path_geometry(self):
        pts = self.trajectory
        n = self.n_wp
        tangents = np.zeros((n, 3), dtype=np.float32)
        curvatures = np.zeros(n, dtype=np.float32)

        for i in range(n):
            d = pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]
            nrm = float(np.linalg.norm(d))
            tangents[i] = d / (nrm + 1e-8)

        for i in range(1, n - 1):
            v1, v2 = pts[i] - pts[i - 1], pts[i + 1] - pts[i]
            cross = np.cross(v1, v2)
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-8
            curvatures[i] = float(np.linalg.norm(cross)) / denom

        return tangents, curvatures

    def _find_nearest_wp(self, pos: NDArray, window: int = 30) -> int:
        lo = max(0, self._wp_idx - 2)
        hi = min(self.n_wp, self._wp_idx + window)
        dists = np.linalg.norm(self.trajectory[lo:hi] - pos, axis=1)
        return lo + int(np.argmin(dists))

    def _frenet_errors(self, pos: NDArray, wp_idx: int):
        wp = self.trajectory[wp_idx]
        tangent = self._tangents[wp_idx]
        delta = pos - wp
        along = float(np.dot(delta, tangent))
        cross_vec = delta - along * tangent
        cross_mag = float(np.linalg.norm(cross_vec))
        # Sign: positive when left of path in the horizontal plane
        sign = 1.0 if float(np.cross(tangent[:2], delta[:2])) >= 0.0 else -1.0
        return sign * cross_mag, along

    def _heading_error(self, rpy: NDArray, wp_idx: int) -> float:
        tangent = self._tangents[wp_idx]
        tangent_yaw = float(np.arctan2(tangent[1], tangent[0]))
        diff = float(rpy[2]) - tangent_yaw
        return float((diff + np.pi) % (2 * np.pi) - np.pi)

    def _lookahead_obs(self, pos: NDArray, rpy: NDArray, wp_idx: int) -> NDArray:
        yaw = float(rpy[2])
        cos_y, sin_y = float(np.cos(yaw)), float(np.sin(yaw))
        result = []
        for k in range(1, self._n_lookahead + 1):
            delta = self.trajectory[min(wp_idx + k * 5, self.n_wp - 1)] - pos
            result.extend([
                 cos_y * delta[0] + sin_y * delta[1],
                -sin_y * delta[0] + cos_y * delta[1],
                delta[2],
            ])
        return np.array(result, dtype=np.float32)

    def _traj_obs(self, pos: NDArray, rpy: NDArray) -> NDArray:
        cte, ate = self._frenet_errors(pos, self._wp_idx)
        h_err = self._heading_error(rpy, self._wp_idx)
        prog = self._wp_idx / max(self.n_wp - 1, 1)
        curv = float(self._curvatures[self._wp_idx])
        return np.concatenate([
            np.array([cte, ate, h_err, prog, curv], dtype=np.float32),
            self._lookahead_obs(pos, rpy, self._wp_idx),
        ])

    # ------------------------------------------------------------------
    # Tiered zone reward
    # ------------------------------------------------------------------

    @staticmethod
    def _zone_reward(abs_cte: float) -> float:
        if abs_cte <= TrajectoryEnv.ZONE_EXACT:
            return 1.0
        if abs_cte <= TrajectoryEnv.ZONE_NEAR:
            t = (abs_cte - TrajectoryEnv.ZONE_EXACT) / (TrajectoryEnv.ZONE_NEAR - TrajectoryEnv.ZONE_EXACT)
            return 1.0 - 0.8 * t
        if abs_cte <= TrajectoryEnv.ZONE_FAR:
            t = (abs_cte - TrajectoryEnv.ZONE_NEAR) / (TrajectoryEnv.ZONE_FAR - TrajectoryEnv.ZONE_NEAR)
            return 0.2 * (1.0 - t)
        return -1.0

    # ------------------------------------------------------------------
    # BiguaGymEnv contract
    # ------------------------------------------------------------------

    def _init_spaces(self) -> None:
        super()._init_spaces()
        # 5 Frenet/progress scalars + 3 body-frame deltas per lookahead waypoint
        extra_dim = 5 + 3 * self._n_lookahead
        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim + extra_dim,),
            dtype=np.float32,
        )

    def _wrap_state(self, state: dict) -> NDArray:
        base_obs = super()._wrap_state(state)
        if self._dynamics is None:
            traj_obs = np.zeros(5 + 3 * self._n_lookahead, dtype=np.float32)
        else:
            pos = np.asarray(self._dynamics[6:9])
            rpy = np.asarray(self._dynamics[15:])
            self._wp_idx = self._find_nearest_wp(pos)
            traj_obs = self._traj_obs(pos, rpy)
        return np.concatenate([base_obs, traj_obs])

    def _reward(self) -> float:
        pos = np.asarray(self._dynamics[6:9])
        ang_vel = np.asarray(self._dynamics[12:15])
        rpy = np.asarray(self._dynamics[15:])

        cte, _ = self._frenet_errors(pos, self._wp_idx)
        h_err = self._heading_error(rpy, self._wp_idx)
        prog = self._wp_idx / max(self.n_wp - 1, 1)

        r_cte = self.W_CTE * self._zone_reward(abs(cte))
        r_prog = self.W_PROG * max(prog - self._prev_progress, 0.0)
        r_align = self.W_ALIGN * float(np.cos(h_err))
        rpy_ratio = np.clip(np.linalg.norm(rpy[:2]) / np.radians(35.0), 0.0, 1.0)
        r_stable = -(rpy_ratio ** 2) * self.W_STABLE
        r_smooth = -self.W_SMOOTH * float(np.linalg.norm(ang_vel))

        return r_cte + r_prog + r_align + r_stable + r_smooth

    def _env_constraints(self) -> None:
        # Draw the full trajectory at a fixed stride so the rendered path stays
        # stationary in world space regardless of the current waypoint index.
        stride = max(1, self.n_wp // 100)
        for i in range(0, self.n_wp, stride):
            self._env.draw_point(self.trajectory[i].tolist())

    def _reset(self) -> tuple:
        state = self._unwrap_state(self._env.reset())
        self._set_dynamics(state)
        self._episode_steps = 0
        self._last_norm = None
        self._on_target = False
        self._wp_idx = 0
        self._prev_progress = 0.0
        return self._wrap_state(state), {}

    def _step(self, action: NDArray) -> tuple:
        self._env_constraints()
        self._episode_steps += self._action_repeat

        flat = np.asarray(action, dtype=np.float32).ravel()
        action_arg = (
            np.tile(flat, (self._batch_size, 1)).tolist()
            if self._batch_size > 1 else flat.tolist()
        )

        state = self._unwrap_state(self._env.step(action_arg, action_repeat=self._action_repeat))
        self._set_dynamics(state)
        obs = self._wrap_state(state)  # updates _wp_idx via _find_nearest_wp

        pos = np.asarray(self._dynamics[6:9])
        r_ang, p_ang, _ = np.asarray(self._dynamics[15:])
        cte, _ = self._frenet_errors(pos, self._wp_idx)
        prog = self._wp_idx / max(self.n_wp - 1, 1)

        out_of_bounds = bool(np.any((pos < self._bounds[0]) | (pos > self._bounds[1])))
        strayed = bool(abs(cte) > self.ZONE_FAR * 2.5)
        reached_end = bool(self._wp_idx >= self.n_wp - 3)

        self._on_target = reached_end
        if reached_end:
            self._on_target_buf += 1

        terminated = bool(
            (abs(r_ang) > np.radians(15))
            or (abs(p_ang) > np.radians(15))
            or out_of_bounds
            or reached_end
        )
        truncated = bool(self._episode_steps >= self.max_episode_steps or strayed)

        reward = self._reward()
        self._prev_progress = prog

        if reached_end:
            reward += self.BONUS_END

        info = {
            "waypoint_index": self._wp_idx,
            "progress": prog,
            "cross_track_error": abs(cte),
            "reached_end": reached_end,
            "reached_goals": int(self._on_target_buf),
        }
        return obs, float(reward), terminated, truncated, info


# ---------------------------------------------------------------------------
# _RangeObsMixin
# ---------------------------------------------------------------------------

class _RangeObsMixin:
    """Mixin that appends range/sonar sensor observations to the flat state vector.

    Sensor assignment by domain:
    - Aerial  (DjiMatrice):                  RangeFinderSensor only  → +10 dims
    - Underwater (BlueROV2, BlueROVHeavy,
                  TorpedoAUV):               ProfilingSonar only     → +100 dims
    - Surface / Multi-domain (BlueBoat,
                              Hydrone):      both sensors            → +110 dims

    Place before the task env in the MRO.  No extra ``__init__`` is needed.
    ProfilingSonar data arrives as ``{'raw': (10,10) float32}`` and is
    automatically flattened to 100 dimensions.
    """

    @property
    def _use_range_finder(self) -> bool:
        return self._agent_type not in DOMAIN['underwater']

    @property
    def _use_profiling_sonar(self) -> bool:
        return self._agent_type not in DOMAIN['aereo']

    @property
    def _range_extra_dim(self) -> int:
        return (10 if self._use_range_finder else 0) + (100 if self._use_profiling_sonar else 0)

    def _build_params(self):
        env_params, obs_params = super()._build_params()
        range_cfg = self._load_config(f"{CONFIG}/range.json")
        _id = self._agent_id(env_params, 'robot')
        range_id = self._agent_id(range_cfg, 'robot')

        wanted = set()
        if self._use_range_finder:
            wanted.add('RangeFinderSensor')
        if self._use_profiling_sonar:
            wanted.add('ProfilingSonar')

        sensors_to_add = [
            s for s in range_cfg['agents'][range_id]['sensors']
            if s.get('sensor_type') in wanted
        ]
        env_params['agents'][_id]['sensors'].extend(sensors_to_add)
        return env_params, obs_params

    def _init_spaces(self) -> None:
        super()._init_spaces()
        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(base_dim + self._range_extra_dim,),
            dtype=np.float32,
        )

    def _wrap_state(self, state: dict) -> NDArray:
        base_obs = super()._wrap_state(state)
        parts = [base_obs]

        if self._use_range_finder:
            rf = state.get('RangeFinderSensor')
            parts.append(
                np.asarray(rf, dtype=np.float32).ravel()
                if rf is not None else np.zeros(10, dtype=np.float32)
            )

        if self._use_profiling_sonar:
            sonar = state.get('ProfilingSonar')
            if sonar is not None:
                raw = sonar['raw'] if isinstance(sonar, dict) else sonar
                parts.append(np.asarray(raw, dtype=np.float32).ravel())
            else:
                parts.append(np.zeros(100, dtype=np.float32))

        return np.concatenate(parts)


# ---------------------------------------------------------------------------
# _PixelObsMixin
# ---------------------------------------------------------------------------

class _PixelObsMixin:
    """Mixin that replaces the flat-Box observation with stacked-pixel Dict observations.

    Place before the task env in the MRO so its ``_wrap_state`` and ``_reset``
    overrides take effect.  Call ``_setup_pixel`` after the parent ``__init__``
    finishes (so ``observation_space`` is already a Box).

    The resulting ``observation_space`` is a :class:`gymnasium.spaces.Dict` keyed
    by channel name (``"rgb"``, ``"depth"``, ``"segmentation"``, ``"normal"``) and
    optionally ``"state"``.  Each channel contains a ``(3*frame_stack, H, W)``
    array so the shape is compatible with convolutional encoders.

    ``self._pixel_channels`` **must** be set before ``super().__init__()`` is
    called so that ``_build_params`` can filter the sensor list to only the
    cameras that are actually needed.
    """

    # Maps pixel channel names to the corresponding biguasim sensor_name in pixels.json.
    _CHANNEL_SENSOR: dict = {
        'rgb': 'RGBCamera',
        'depth': 'DepthCamera',
        'segmentation': 'AnnotationComponent',
        # 'normal' has no backing sensor — always synthesised as zeros
    }

    def _setup_pixel(
        self,
        frame_stack: int,
        frame_size: tuple,
        include_state: bool,
        render_channel: str | None = None,
    ) -> None:
        self._frame_stack_n = frame_stack
        self._frame_size = tuple(frame_size)
        self._include_state = include_state
        self._render_channel = render_channel
        self._pixel_stack_obj = PixelStack(self._pixel_channels, list(frame_size), frame_stack)
        self._wrap_is_reset = False
        state_dim = int(self.observation_space.shape[0])
        self.observation_space = self._build_pixel_obs_space(state_dim)

    def _build_params(self):
        env_params, obs_params = super()._build_params()
        pixel_cfg = self._load_config(f"{CONFIG}/pixels.json")
        _id = self._agent_id(env_params, 'robot')
        pixel_id = self._agent_id(pixel_cfg, 'robot')

        required = {self._CHANNEL_SENSOR[ch] for ch in self._pixel_channels if ch in self._CHANNEL_SENSOR}
        # Also load the render channel sensor even when it is not an observation channel.
        rc = getattr(self, '_render_channel', None)
        if rc and rc in self._CHANNEL_SENSOR:
            required.add(self._CHANNEL_SENSOR[rc])
        visual = set(self._CHANNEL_SENSOR.values())

        all_sensors = pixel_cfg['agents'][pixel_id]['sensors']
        env_params['agents'][_id]['sensors'] = [
            s for s in all_sensors
            if s.get('sensor_name') not in visual or s.get('sensor_name') in required
        ]
        return env_params, obs_params

    def _build_pixel_obs_space(self, state_dim: int) -> spaces.Dict:
        pixel_spaces = {}
        for ch in self._pixel_channels:
            if ch in ('rgb', 'segmentation'):
                pixel_spaces[ch] = spaces.Box(
                    0, 255, (3 * self._frame_stack_n, *self._frame_size), np.uint8
                )
            else:
                pixel_spaces[ch] = spaces.Box(
                    -np.inf, np.inf, (3 * self._frame_stack_n, *self._frame_size), np.float32
                )
        if self._include_state:
            pixel_spaces['state'] = spaces.Box(
                -np.inf, np.inf, (state_dim,), np.float32
            )
        return spaces.Dict(pixel_spaces)

    def _wrap_state(self, state: dict) -> dict:
        state_obs = super()._wrap_state(state)

        h, w = self._frame_size

        # --- RGB: (H, W, 4) RGBA uint8 → drop alpha (tolerate optional batch axis) ---
        raw_rgb = np.asarray(state.get('RGBCamera'))
        if raw_rgb.ndim == 4 and raw_rgb.shape[0] == 1:
            raw_rgb = raw_rgb[0]
        rgb = raw_rgb[:, :, :3] if raw_rgb.ndim == 3 else np.zeros((h, w, 3), dtype=np.float32)

        # --- Depth: dict with 'depth_map' key → (H, W) float32 → normalize to grayscale (H, W, 3) ---
        raw_depth = state.get('DepthCamera')
        if raw_depth is not None:
            depth_map = np.asarray(
                raw_depth['depth_map'] if isinstance(raw_depth, dict) else raw_depth,
                dtype=np.float32,
            )
            d_min, d_max = float(depth_map.min()), float(depth_map.max())
            gray = ((depth_map - d_min) / (d_max - d_min) * 255.0) if d_max > d_min else np.zeros_like(depth_map)
            depth = np.stack([gray, gray, gray], axis=-1)
        else:
            depth = np.zeros((h, w, 3), dtype=np.float32)

        # --- Segmentation: (H, W, 4) RGBA uint8 → drop alpha (may be absent) ---
        raw_seg = state.get('AnnotationComponent')
        if raw_seg is not None:
            raw_seg = np.asarray(raw_seg)
            if raw_seg.ndim == 4 and raw_seg.shape[0] == 1:
                raw_seg = raw_seg[0]
            seg = raw_seg[:, :, :3] if raw_seg.ndim == 3 else np.zeros((h, w, 3), dtype=np.float32)
        else:
            seg = np.zeros((h, w, 3), dtype=np.float32)

        # Normal: no sensor available; synthesised as zeros
        normal = np.zeros((h, w, 3), dtype=np.float32)

        raw_frames = {'rgb': rgb, 'depth': depth, 'segmentation': seg, 'normal': normal}

        # Override the render frame with the chosen pixel channel (pre-stack, uint8 BGR).
        if self._render_channel and self._render_channel in raw_frames:
            # import cv2
            # frame = np.clip(raw_frames[self._render_channel], 0, 255).astype(np.uint8)
            # self._last_render_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._last_render_frame = raw_frames[self._render_channel]

        self._pixel_stack_obj.append((rgb, depth, seg, normal), self._wrap_is_reset)
        self._wrap_is_reset = False
        stacked = self._pixel_stack_obj.built_stack()
        obs = {ch: stacked[ch] for ch in self._pixel_channels}
        if self._include_state:
            obs['state'] = state_obs
        return obs

    def _reset(self):
        self._wrap_is_reset = True
        return super()._reset()


# ---------------------------------------------------------------------------
# Pixel env classes
# ---------------------------------------------------------------------------

class HoverPixelEnv(_PixelObsMixin, HoverEnv):
    """Pixel-observation variant of :class:`HoverEnv`.

    Parameters
    ----------
    pixel_channels:
        Subset of ``["rgb", "depth", "segmentation", "normal"]`` to include.
        Defaults to ``["rgb"]``.
    frame_stack:
        Number of consecutive frames to stack (observation shape axis 0 = 3 * frame_stack).
    frame_size:
        ``(H, W)`` target resolution for each frame.
    include_state:
        When ``True``, adds a ``"state"`` key with the flat state vector.
    render_channel:
        Which pixel channel to use as the render / recording source
        (``"rgb"``, ``"depth"``, or ``"segmentation"``).  ``None`` falls back
        to ``CameraView``.  The channel's sensor is loaded even when it is not
        in ``pixel_channels``.
    """

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        render_mode: str = None,
        pixel_channels: list = None,
        frame_stack: int = 3,
        frame_size: tuple = (84, 84),
        include_state: bool = False,
        render_channel: str | None = None,
        reward_version: str = "v1",
    ) -> None:
        self._pixel_channels = pixel_channels if pixel_channels is not None else ['rgb']
        self._render_channel = render_channel
        super().__init__(
            seed, agent_type, control_abstraction, location, rotation,
            batch_size, observation_type, show_viewer, timestep,
            action_stack, target_factor, render_mode, reward_version=reward_version,
        )
        self._setup_pixel(frame_stack, frame_size, include_state, render_channel)


class LandPixelEnv(_PixelObsMixin, LandEnv):
    """Pixel-observation variant of :class:`LandEnv`."""

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        render_mode: str = None,
        pixel_channels: list = None,
        frame_stack: int = 3,
        frame_size: tuple = (84, 84),
        include_state: bool = False,
        render_channel: str | None = None,
        reward_version: str = "v1",
    ) -> None:
        self._pixel_channels = pixel_channels if pixel_channels is not None else ['rgb']
        self._render_channel = render_channel
        super().__init__(
            seed, agent_type, control_abstraction, location, rotation,
            batch_size, observation_type, show_viewer, timestep,
            action_stack, target_factor, render_mode, reward_version=reward_version,
        )
        self._setup_pixel(frame_stack, frame_size, include_state, render_channel)


class DockPixelEnv(_PixelObsMixin, DockEnv):
    """Pixel-observation variant of :class:`DockEnv`."""

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        render_mode: str = None,
        pixel_channels: list = None,
        frame_stack: int = 3,
        frame_size: tuple = (84, 84),
        include_state: bool = False,
        render_channel: str | None = None,
        reward_version: str = "v1",
    ) -> None:
        self._pixel_channels = pixel_channels if pixel_channels is not None else ['rgb']
        self._render_channel = render_channel
        super().__init__(
            seed, agent_type, control_abstraction, location, rotation,
            batch_size, observation_type, show_viewer, timestep,
            action_stack, target_factor, render_mode, reward_version=reward_version,
        )
        self._setup_pixel(frame_stack, frame_size, include_state, render_channel)


class NavPixelEnv(_PixelObsMixin, NavEnv):
    """Pixel-observation variant of :class:`NavEnv`."""

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        render_mode: str = None,
        pixel_channels: list = None,
        frame_stack: int = 3,
        frame_size: tuple = (84, 84),
        include_state: bool = False,
        render_channel: str | None = None,
        reward_version: str = "v1",
    ) -> None:
        self._pixel_channels = pixel_channels if pixel_channels is not None else ['rgb']
        self._render_channel = render_channel
        super().__init__(
            seed, agent_type, control_abstraction, location, rotation,
            batch_size, observation_type, show_viewer, timestep,
            action_stack, target_factor, render_mode, reward_version=reward_version,
        )
        self._setup_pixel(frame_stack, frame_size, include_state, render_channel)


class TrajectoryPixelEnv(_PixelObsMixin, TrajectoryEnv):
    """Pixel-observation variant of :class:`TrajectoryEnv`."""

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        target_trajectory: str | NDArray = 'sine',
        n_lookahead: int = 5,
        waypoint_radius: float = 0.2,
        render_mode: str = None,
        pixel_channels: list = None,
        frame_stack: int = 3,
        frame_size: tuple = (84, 84),
        include_state: bool = False,
        render_channel: str | None = None,
        reward_version: str = "v1",
    ) -> None:
        self._pixel_channels = pixel_channels if pixel_channels is not None else ['rgb']
        self._render_channel = render_channel
        super().__init__(
            seed, agent_type, control_abstraction, location, rotation,
            batch_size, observation_type, show_viewer, timestep,
            action_stack, target_factor, target_trajectory, n_lookahead,
            waypoint_radius, render_mode, reward_version=reward_version,
        )
        self._setup_pixel(frame_stack, frame_size, include_state, render_channel)


# ---------------------------------------------------------------------------
# LandCoopPixelEnv  (mobile-target landing, -v2)
# ---------------------------------------------------------------------------

class LandCoopPixelEnv(_PixelObsMixin, LandEnv):
    """Pixel-observation landing on a mobile BlueBoat target.

    The landing pad is a BlueBoat agent (``target_robot``) that moves along a
    sinusoidal or figure-8 surface trajectory.  Each step the boat receives a
    position command ``[x, y, z, yaw]`` via ``cmd_pos_yaw``; the main agent
    must descend and land on the moving pad.

    Pixel cameras are rotated ``[0, −90, 0]`` so they face straight down,
    giving a top-down view ideal for pad detection.

    Parameters
    ----------
    target_trajectory:
        ``'sine'`` or ``'figure8'`` — path shape for the boat.
    trajectory_scale:
        Amplitude / radius of the boat's path in metres.
    trajectory_speed:
        Phase increment per simulation step (rad/step).  Controls how fast
        the boat moves along its path.
    """

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        render_mode: str = None,
        pixel_channels: list = None,
        frame_stack: int = 3,
        frame_size: tuple = (84, 84),
        include_state: bool = False,
        render_channel: str | None = None,
        target_trajectory: str = 'sine',
        trajectory_scale: float = 5.0,
        trajectory_speed: float = 0.02,
        camera_half_fov_deg: float = 45.0,
        reward_version: str = "v1",
    ) -> None:
        self._pixel_channels = pixel_channels if pixel_channels is not None else ['rgb']
        self._render_channel = render_channel
        self._target_trajectory_type = target_trajectory
        self._trajectory_scale = trajectory_scale
        self._trajectory_speed = trajectory_speed
        self._traj_phase = 0.0
        self._traj_center = np.array([location[0], location[1]], dtype=np.float32)
        self._camera_half_fov_deg = camera_half_fov_deg

        super().__init__(
            seed, agent_type, control_abstraction, location, rotation,
            batch_size, observation_type, show_viewer, timestep,
            action_stack, target_factor, render_mode, reward_version=reward_version,
        )
        self._setup_pixel(frame_stack, frame_size, include_state, render_channel)

    def _build_params(self):
        # Use cooperation.json (robot + target_robot) instead of state.json
        env_params = self._load_config(f"{CONFIG}/cooperation.json")

        _id = self._agent_id(env_params, 'robot')
        env_params['agents'][_id]['agent_type'] = self._agent_type
        env_params['agents'][_id]['control_abstraction'] = self._control_abstraction
        env_params['agents'][_id]['location'] = self._location
        env_params['agents'][_id]['rotation'] = self._rotation
        env_params['agents'][_id]['dynamics']['batch_size'] = self._batch_size

        # Place the boat at surface level near the robot's xy spawn position
        target_id = self._agent_id(env_params, 'target_robot')
        env_params['agents'][target_id]['location'] = [
            float(self._location[0]), float(self._location[1]), 0.0
        ]
        env_params['agents'][target_id]['rotation'] = [0, 0, 0]
        env_params['agents'][target_id]['dynamics']['batch_size'] = 1

        loc = np.asarray(self._location, dtype=np.float32)
        self._bounds = np.array([loc - 20.0, loc + 20.0])
        self._bounds[0, 2] = max(float(self._bounds[0, 2]), 0.1)
        self._bounds[1, 2] = max(float(loc[2]), 2.0 * self._bounds[0, 2])

        if isinstance(self._observation_type, str):
            self._observation_type = [self._observation_type]
        obs_params = {
            obs_type: np.zeros(DATA_REGISTRY[obs_type]).ravel().shape
            for obs_type in self._observation_type
        }

        # Add pixel cameras from pixels.json with downward-looking rotation
        pixel_cfg = self._load_config(f"{CONFIG}/pixels.json")
        pixel_id = self._agent_id(pixel_cfg, 'robot')
        required = {self._CHANNEL_SENSOR[ch] for ch in self._pixel_channels if ch in self._CHANNEL_SENSOR}
        rc = getattr(self, '_render_channel', None)
        if rc and rc in self._CHANNEL_SENSOR:
            required.add(self._CHANNEL_SENSOR[rc])

        for s in pixel_cfg['agents'][pixel_id]['sensors']:
            if s.get('sensor_name') in required:
                cam = dict(s)
                cam['socket'] = 'CameraSocket'
                if self._agent_type in DOMAIN['multi-domain']:
                    cam['location'] = [0, 0, -1]
                cam['rotation'] = [0, 90, 0]
                env_params['agents'][_id]['sensors'].append(cam)

        return env_params.copy(), obs_params.copy()

    def _env_constraints(self) -> None:
        pass

    def _sample_traj_center(self) -> NDArray:
        """Sample (x, y) uniformly within the intersection of env xy-bounds and the
        downward camera's FOV disk at spawn altitude."""
        h = max(float(self._location[2]), 0.1)
        fov_radius = h * np.tan(np.radians(self._camera_half_fov_deg))
        cx, cy = float(self._location[0]), float(self._location[1])

        x_lo, x_hi = float(self._bounds[0, 0]), float(self._bounds[1, 0])
        y_lo, y_hi = float(self._bounds[0, 1]), float(self._bounds[1, 1])
        bound_radius = min(x_hi - cx, cx - x_lo, y_hi - cy, cy - y_lo)

        r = self.rng.uniform(0, fov_radius - ((bound_radius / self._trajectory_scale) * 0.5))
        angle = self.rng.uniform(0.0, 2 * np.pi)

        x = np.clip(cx + r * np.cos(angle), x_lo, x_hi)
        y = np.clip(cy + r * np.sin(angle), y_lo, y_hi)
        return np.array([x, y], dtype=np.float32)

    def _compute_target_pos(self, phase: float) -> list:
        """Return ``[x, y, z, yaw]`` for the boat at the given trajectory phase."""
        tx, ty = float(self._traj_center[0]), float(self._traj_center[1])
        s = self._trajectory_scale

        if self._target_trajectory_type == 'sine':
            x = tx + s * np.sin(phase)
            y = ty + s * np.cos(phase * 0.5)
            dx = s * float(np.cos(phase))
            dy = -s * 0.5 * float(np.sin(phase * 0.5))
        else:  # figure8
            x = tx + s * np.sin(phase)
            y = ty + s * np.sin(phase) * np.cos(phase)
            dx = s * float(np.cos(phase))
            dy = s * float(np.cos(phase) ** 2 - np.sin(phase) ** 2)

        yaw = float(np.arctan2(dy, dx))
        return [float(x), float(y), 0.0, yaw]

    def _reset(self):
        # Multi-agent reset returns {agent_name: {sensor_name: data}}
        full_state = self._env.reset()

        # Resample trajectory center within FOV (gated by target_factor curriculum)
        if self._on_target_buf % self._target_factor == 0:
            self._traj_center = self._sample_traj_center()
            tx, ty = float(self._traj_center[0]), float(self._traj_center[1])
            yaw = float(np.arctan2(ty, tx))
            self._env.move_agent('target_robot', [tx, ty, 0.1], [0.0,0.0,yaw])
            

        # Boat starts at the trajectory origin, which is the center itself:
        # figure8 at phase=0 → (tx, ty);  sine at phase=π → (tx, ty)
        self._traj_phase = 0.0 if self._target_trajectory_type == 'figure8' else np.pi
        target_pos = self._compute_target_pos(self._traj_phase)
        self._target = np.array(target_pos[:3], dtype=np.float32)
        self._target_list = self._target.tolist()

        self._wrap_is_reset = True

        # Teleport target_robot to its phase-based start position.
        # env.reset() always restores config spawn (cx, cy); one step with zero
        # robot action physically moves the boat to the sampled start position.
        
        flat = np.zeros(int(self._env.action_space.shape[0]), dtype=np.float32)
        robot_action = flat.tolist() if self._batch_size == 1 else np.tile(flat, (self._batch_size, 1)).tolist()
        full_state = self._env.step(
            {'robot': robot_action, 'target_robot': target_pos},
            action_repeat=1,
        )

        state = full_state['robot']
        self._set_dynamics(state)
        self._episode_steps = 0
        self._last_norm = None
        self._on_target = False

        return self._wrap_state(state), {}

    def _step(self, action: NDArray) -> tuple:
        self._episode_steps += self._action_repeat

        # Advance the boat along its trajectory
        self._traj_phase += self._trajectory_speed
        target_pos = self._compute_target_pos(self._traj_phase)
        self._target = np.array(target_pos[:3], dtype=np.float32)
        self._target_list = self._target.tolist()

        self._env_constraints()

        flat = np.asarray(action, dtype=np.float32).ravel()
        if self._batch_size > 1:
            robot_action = np.tile(flat, (self._batch_size, 1)).tolist()
        else:
            robot_action = flat.tolist()

        # Multi-agent step: robot gets control action, target_robot gets pos command
        # Returns {agent_name: {sensor_name: data}}
        full_state = self._env.step(
            {'robot': robot_action, 'target_robot': target_pos},
            action_repeat=self._action_repeat,
        )
        state = full_state['robot']
        self._set_dynamics(state)
        obs = self._wrap_state(state)

        pos = np.asarray(self._dynamics[6:9])
        vel = np.asarray(self._dynamics[3:6])
        r, p, _ = np.asarray(self._dynamics[15:])

        truncated = self._episode_steps >= self.max_episode_steps
        out_of_bounds = bool(np.any((pos < self._bounds[0]) | (pos > self._bounds[1])))

        height_above_target = max(float(pos[2] - self._target[2]), 0.0)
        proximity_weight = float(np.exp(-height_above_target / 3.0))
        descent_speed = max(float(-vel[2]), 0.0)
        hard_landing = bool(proximity_weight > 0.5 and descent_speed > 2.0)

        terminated = bool(
            (abs(r) > np.radians(15))
            or (abs(p) > np.radians(15))
            or out_of_bounds
            or self._on_target
            or hard_landing
        )

        reward = self._reward()
        if self._on_target:
            reward = 3.0 * abs(reward)
        elif hard_landing:
            reward = -5.0

        info = {"reached_goals": int(self._on_target_buf), "hard_landing": hard_landing}
        return obs, float(reward), terminated, truncated, info


# ---------------------------------------------------------------------------
# Range env classes  (-v2)
# ---------------------------------------------------------------------------

class NavRangeEnv(_RangeObsMixin, NavEnv):
    """Range/sonar-observation variant of :class:`NavEnv`.

    Observation = flat state vector + range sensor data.
    Sensor assignment follows domain rules defined in :class:`_RangeObsMixin`.
    """

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        render_mode: str = None,
        reward_version: str = "v1",
    ) -> None:
        super().__init__(
            seed, agent_type, control_abstraction, location, rotation,
            batch_size, observation_type, show_viewer, timestep,
            action_stack, target_factor, render_mode, reward_version=reward_version,
        )


class TrajectoryRangeEnv(_RangeObsMixin, TrajectoryEnv):
    """Range/sonar-observation variant of :class:`TrajectoryEnv`.

    Observation = flat state + Frenet/lookahead augmentation + range sensor data.
    Sensor assignment follows domain rules defined in :class:`_RangeObsMixin`.
    """

    def __init__(
        self,
        seed: int,
        agent_type: str,
        control_abstraction: str,
        location: list,
        rotation: list,
        batch_size: int = 1,
        observation_type: str | List[str] = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        target_trajectory: str | NDArray = 'sine',
        n_lookahead: int = 5,
        waypoint_radius: float = 0.2,
        render_mode: str = None,
        reward_version: str = "v1",
    ) -> None:
        super().__init__(
            seed, agent_type, control_abstraction, location, rotation,
            batch_size, observation_type, show_viewer, timestep,
            action_stack, target_factor, target_trajectory, n_lookahead,
            waypoint_radius, render_mode, reward_version=reward_version,
        )
