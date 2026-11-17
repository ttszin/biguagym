"""Goto do Hydrone no dique inundado do Pier-Harbor (spec hydrone-goto-dique).

Tarefa dos trabalhos do Dranaju (Depth-CUPRL, arXiv 2206.15211) e do Ricardo Grando: ir até alvos aleatórios no ar e na
água, comandando velocidade para a frente, velocidade de guinada e velocidade vertical. Ação, recompensa e término seguem
o código do Dranaju (``dranaju/curl_navigation``, ``environment_hydrone_image_depth.py``). O cenário (área, pilares,
caixas de exclusão, faixas dos alvos) vem de ``config/goto_dique.json``.
"""
import json
import math
import warnings

import numpy as np

from gymnasium import spaces
from numpy.typing import NDArray

from .environments import CONFIG, HoverEnv


def _wrap(ang: float) -> float:
    return (ang + math.pi) % (2 * math.pi) - math.pi


def carregar_cenario(nome: str) -> dict:
    """Lê o cenário ``nome`` (``pilares`` ou ``dique``) de ``config/goto_dique.json``."""
    with open(f"{CONFIG}/goto_dique.json") as f:
        cenarios = json.load(f)['cenarios']
    if nome not in cenarios:
        raise ValueError(f"cenário desconhecido: {nome!r} (use {sorted(cenarios)})")
    return cenarios[nome]


def recompensa_dranaju(d_ant: float, d: float, z_rel: float, colisao: bool, chegou: bool) -> float:
    """Recompensa do Dranaju, na mesma ordem do ``setReward`` dele (z relativo ao nível da água)."""
    r = GotoEnv.GANHO_DIST * (d_ant - d)
    if GotoEnv.FAIXA_SUPERFICIE[0] < z_rel < GotoEnv.FAIXA_SUPERFICIE[1]:
        if r < 0.0:
            r -= GotoEnv.BONUS_SUPERFICIE
        if r > 0.0:
            r += GotoEnv.BONUS_SUPERFICIE
    if z_rel < GotoEnv.Z_PENALIDADE_FUNDO:
        r = GotoEnv.R_FUNDO
    if colisao:
        r = GotoEnv.R_COLISAO
    if chegou:
        r = GotoEnv.R_CHEGADA
    return float(r)


class GotoEnv(HoverEnv):
    """Goto do Hydrone com ``cmd_vel_yaw``; a ação é [frente, guinada, vertical] em [-1, 1].

    Observação (``-v0``): pose de 9 valores do Dranaju (heading, heading_z, distância, posição e alvo relativos à origem
    do cenário) + as 3 ações anteriores.
    """

    # Ação do Dranaju (curl_prio_image_depth.py: ACTION_V_MAX = ACTION_W_MAX = 0,3; step(): linear + 0,01)
    V_MAX = 0.3
    V_OFFSET = 0.01
    W_MAX = 0.3
    VZ_MAX = 0.3
    TICKS_POR_S = 20                     # state.json: ticks_per_sec
    YAW_DESVIO_MAX = math.radians(45.0)  # desvio máximo enviado ao cmd_vel_yaw

    # Recompensa e término do Dranaju (setReward/getState)
    GANHO_DIST = 0.1
    FAIXA_SUPERFICIE = (-0.23, -0.05)
    BONUS_SUPERFICIE = 0.01
    Z_PENALIDADE_FUNDO = -0.75
    R_FUNDO = -0.1
    R_COLISAO = -1.0
    R_CHEGADA = 1.0
    RAIO_CHEGADA = 0.4
    DIST_COLISAO = 0.625
    MAX_PASSOS = 1000                    # Depth-CUPRL: episódios de 1000 passos

    N_ACOES = 3

    def __init__(
        self,
        seed: int,
        agent_type: str = 'Hydrone',
        control_abstraction: str = 'cmd_vel_yaw',
        location: list = None,
        rotation: list = None,
        batch_size: int = 1,
        observation_type: str | list = "DynamicsSensor",
        show_viewer: bool = False,
        timestep: bool = False,
        action_stack: int = 1,
        target_factor: int = 1,
        render_mode: str = None,
        cenario: str = 'pilares',
        reward_version: str = "v1",
        env_version: str = "v2",
    ) -> None:
        self._cena = carregar_cenario(cenario)
        self._cenario = cenario
        if self._cena.get('provisorio'):
            warnings.warn(f"cenário {cenario!r} do Goto com coordenadas provisórias (config/goto_dique.json)")
        self._validar_spawn()
        if location is None:
            location = list(self._cena['spawn'])
        if rotation is None:
            rotation = [0.0, 0.0, float(self._cena.get('spawn_yaw_deg', 0.0))]
        self._yaw_ref = 0.0
        self._acao_ant = np.zeros(self.N_ACOES, dtype=np.float32)
        self._alvos_atingidos = 0
        super().__init__(seed, agent_type, control_abstraction, location, rotation, batch_size, observation_type,
                         show_viewer, timestep, action_stack, target_factor, render_mode,
                         reward_version=reward_version, env_version=env_version)

    @property
    def max_episode_steps(self) -> int:
        return self.MAX_PASSOS

    # ------------------------------------------------------------------
    # Construção
    # ------------------------------------------------------------------
    def _build_params(self):
        env_params, obs_params = super()._build_params()
        _id = self._agent_id(env_params, 'robot')
        env_params['agents'][_id]['sensors'].append(
            {'sensor_type': 'CollisionSensor', 'location': [0, 0, 0], 'rotation': [0, 0, 0]})
        agua = float(self._cena['nivel_agua'])
        (x0, y0), (x1, y1) = self._cena['area']
        z0, z1 = self._cena['z_limites']
        self._bounds = np.array([[x0, y0, agua + z0], [x1, y1, agua + z1]], dtype=np.float64)
        return env_params, obs_params

    def _obs_dim(self) -> int:
        return 9 + self.N_ACOES

    def _init_spaces(self) -> None:
        self.observation_space = spaces.Box(-np.inf, np.inf, (self._obs_dim(),), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.N_ACOES,), np.float32)

    # ------------------------------------------------------------------
    # Cenário: pilares e alvos
    # ------------------------------------------------------------------
    def _spawn_pilares(self) -> None:
        """Recria os pilares (o reset do simulador remove os objetos criados com spawn_prop)."""
        agua = float(self._cena['nivel_agua'])
        for i, p in enumerate(self._cena['pilares']):
            r, h = float(p['raio']), float(p['altura'])
            centro_z = agua + float(p['base_z']) + h / 2.0
            self._env.spawn_prop('cylinder', location=[float(p['xy'][0]), float(p['xy'][1]), centro_z],
                                 rotation=[0.0, 0.0, 0.0], scale=[2 * r, 2 * r, h],
                                 material='steel', tag='Pilar', name=f'pilar{i}')

    def _dist_pilar(self, pos) -> float:
        """Menor distância horizontal até a superfície de um pilar cuja altura cobre pos[2] (inf se nenhum)."""
        agua = float(self._cena['nivel_agua'])
        dmin = math.inf
        for p in self._cena['pilares']:
            base = agua + float(p['base_z'])
            if base <= pos[2] <= base + float(p['altura']):
                dmin = min(dmin, math.hypot(pos[0] - p['xy'][0], pos[1] - p['xy'][1]) - float(p['raio']))
        return dmin

    def _dist_exclusao(self, pos) -> float:
        """Menor distância até uma caixa de exclusão (0 dentro dela; inf se não há caixas)."""
        agua = float(self._cena['nivel_agua'])
        dmin = math.inf
        for caixa in self._cena['exclusao']:
            lo = np.array(caixa['min'], dtype=np.float64) + [0.0, 0.0, agua]
            hi = np.array(caixa['max'], dtype=np.float64) + [0.0, 0.0, agua]
            dmin = min(dmin, float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - pos, pos - hi)))))
        return dmin

    def _validar_spawn(self) -> None:
        spawn = np.asarray(self._cena['spawn'], dtype=np.float64)
        if min(self._dist_pilar(spawn), self._dist_exclusao(spawn)) < self.DIST_COLISAO:
            raise ValueError(f"spawn {spawn.tolist()} a menos de {self.DIST_COLISAO} m de um pilar ou caixa de exclusão "
                             f"no cenário {self._cenario!r} (goto_dique.json)")

    def _sortear_alvo(self, pos=None) -> NDArray:
        c = self._cena
        agua = float(c['nivel_agua'])
        folga = float(c['folga_alvo'])
        (x0, y0), (x1, y1) = c['area']
        for _ in range(10000):
            x = self.rng.uniform(x0 + folga, x1 - folga)
            y = self.rng.uniform(y0 + folga, y1 - folga)
            faixa = c['alvo_z_agua'] if self.rng.uniform() < float(c['p_alvo_agua']) else c['alvo_z_ar']
            alvo = np.array([x, y, agua + self.rng.uniform(*faixa)], dtype=np.float64)
            if self._dist_pilar(alvo) < folga or self._dist_exclusao(alvo) < folga:
                continue
            if pos is not None and np.linalg.norm(alvo - np.asarray(pos, dtype=np.float64)) < float(c['dist_min_alvo']):
                continue
            return alvo
        raise RuntimeError(f"não achei alvo livre no cenário {self._cenario!r}; revise area/folga em goto_dique.json")

    # ------------------------------------------------------------------
    # Ação: [frente, guinada, vertical] → cmd_vel_yaw [vx, vy, vz, desvio_graus]
    # ------------------------------------------------------------------
    def _comando(self, acao) -> NDArray:
        a = np.clip(np.asarray(acao, dtype=np.float64).ravel()[:self.N_ACOES], -1.0, 1.0)
        v = (a[0] + 1.0) / 2.0 * self.V_MAX + self.V_OFFSET
        w = a[1] * self.W_MAX
        vz = a[2] * self.VZ_MAX
        psi = float(self._dynamics[17])
        self._yaw_ref = _wrap(self._yaw_ref + w * self._action_repeat / self.TICKS_POR_S)
        desvio = _wrap(self._yaw_ref - psi)
        if abs(desvio) > self.YAW_DESVIO_MAX:
            desvio = math.copysign(self.YAW_DESVIO_MAX, desvio)
            self._yaw_ref = _wrap(psi + desvio)
        return np.array([v * math.cos(psi), v * math.sin(psi), vz, math.degrees(desvio)], dtype=np.float32)

    # ------------------------------------------------------------------
    # Colisão, observação, reset e passo
    # ------------------------------------------------------------------
    def _colisao(self, state: dict):
        """Motivo da colisão ou None (regra do Dranaju: obstáculo a menos de 0,625 m)."""
        pos = np.asarray(self._dynamics[6:9], dtype=np.float64)
        contato = state.get('CollisionSensor')
        if contato is not None and bool(np.any(np.asarray(contato))):
            return "contato"
        lo, hi = self._bounds
        if min(pos[0] - lo[0], hi[0] - pos[0], pos[1] - lo[1], hi[1] - pos[1]) < self.DIST_COLISAO:
            return "parede"
        if not (lo[2] <= pos[2] <= hi[2]):
            return "altura"
        if self._dist_pilar(pos) < self.DIST_COLISAO:
            return "pilar"
        if self._dist_exclusao(pos) < self.DIST_COLISAO:
            return "obstaculo"
        return None

    def _distancia(self) -> float:
        return float(np.linalg.norm(np.asarray(self._target) - np.asarray(self._dynamics[6:9], dtype=np.float64)))

    def _pose9(self) -> NDArray:
        """[heading, heading_z, distância, posição (3), alvo (3)], posições relativas à origem do cenário e à água."""
        pos = np.asarray(self._dynamics[6:9], dtype=np.float64)
        alvo = np.asarray(self._target, dtype=np.float64)
        d = alvo - pos
        heading = _wrap(math.atan2(d[1], d[0]) - float(self._dynamics[17]))
        heading_z = math.atan2(d[2], math.hypot(d[0], d[1]))
        ref = np.array([*self._cena['origem'], self._cena['nivel_agua']], dtype=np.float64)
        return np.concatenate([[heading, heading_z, np.linalg.norm(d)], pos - ref, alvo - ref]).astype(np.float32)

    def _wrap_state(self, state: dict) -> NDArray:
        raw = state.get('CameraView')
        if raw is not None:
            self._last_render_frame = np.asarray(raw, dtype=np.uint8)[:, :, 0:3]
        return np.concatenate([self._pose9(), self._acao_ant]).astype(np.float32)

    def _novo_alvo(self) -> None:
        self._target = self._sortear_alvo(self._dynamics[6:9])
        self._target_list = self._target.tolist()
        self._last_norm = self._distancia()

    def _reset(self):
        state = self._unwrap_state(self._env.reset())
        self._spawn_pilares()
        self._set_dynamics(state)
        self._episode_steps = 0
        self._alvos_atingidos = 0
        self._yaw_ref = float(self._dynamics[17])
        self._acao_ant = np.zeros(self.N_ACOES, dtype=np.float32)
        self._novo_alvo()
        return self._wrap_state(state), {}

    def _step(self, action):
        self._env_constraints()
        self._episode_steps += self._action_repeat
        cmd = self._comando(action)
        action_arg = np.tile(cmd, (self._batch_size, 1)).tolist() if self._batch_size > 1 else cmd.tolist()
        state = self._unwrap_state(self._env.step(action_arg, action_repeat=self._action_repeat))
        self._set_dynamics(state)
        self._acao_ant = np.clip(np.asarray(action, dtype=np.float32).ravel()[:self.N_ACOES], -1.0, 1.0)

        motivo = self._colisao(state)
        d = self._distancia()
        chegou = d < self.RAIO_CHEGADA
        z_rel = float(self._dynamics[8]) - float(self._cena['nivel_agua'])
        d_ant = self._last_norm
        reward = recompensa_dranaju(d_ant, d, z_rel, motivo is not None, chegou)
        self._last_norm = d
        if chegou:
            self._alvos_atingidos += 1
            self._on_target_buf += 1
            self._novo_alvo()

        terminated = motivo is not None
        truncated = self._episode_steps >= self.max_episode_steps
        if motivo is None and truncated:
            motivo = "timeout"
        info = {"reached_goals": int(self._alvos_atingidos),
                "termination_reason": motivo,
                "reward_terms": {"distancia": float(self.GANHO_DIST * (d_ant - d)), "chegou": float(chegou),
                                 "z_rel": z_rel, "total": reward}}
        return self._wrap_state(state), reward, terminated, truncated, info
