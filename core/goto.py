"""Goto do Hydrone no dique inundado do Pier-Harbor (spec hydrone-goto-dique).

Tarefa dos trabalhos do Dranaju (Depth-CUPRL, arXiv 2206.15211) e do Ricardo Grando: ir até alvos aleatórios no ar e na
água, comandando velocidade para a frente, velocidade de guinada e velocidade vertical. Ação, recompensa e término seguem
o código do Dranaju (``dranaju/curl_navigation``, ``environment_hydrone_image_depth.py``). O cenário (área, obstáculos,
caixas de exclusão, faixas dos alvos) vem de ``config/goto_dique.json``.

Obstáculos: o mundo SkyDive não cria objetos (``SpawnProp`` recusado pelo motor), então pilares e vigas são feitos de
agentes (``agente_obstaculo``, TorpedoAUV) em lote, parados em ``accel`` com aceleração zero e com a posição reimposta a cada
tick (sem isso derivam e, encostados, são arremessados).
"""
import json
import math
import resource
import warnings

import numpy as np

from gymnasium import spaces
from numpy.typing import NDArray

from .environments import CONFIG, HoverEnv, _PixelObsMixin


def _wrap(ang: float) -> float:
    return (ang + math.pi) % (2 * math.pi) - math.pi


def carregar_cenario(nome: str) -> dict:
    """Lê o cenário ``nome`` (``pilares`` ou ``dique``) de ``config/goto_dique.json``."""
    with open(f"{CONFIG}/goto_dique.json") as f:
        cenarios = json.load(f)['cenarios']
    if nome not in cenarios:
        raise ValueError(f"cenário desconhecido: {nome!r} (use {sorted(cenarios)})")
    return cenarios[nome]


def carregar_agente_obstaculo() -> dict:
    """Tipo e medidas do agente de que os obstáculos são feitos (``agente_obstaculo`` de ``config/goto_dique.json``)."""
    with open(f"{CONFIG}/goto_dique.json") as f:
        return json.load(f)['agente_obstaculo']


def geometria_obstaculos(cena: dict, agente: dict) -> dict:
    """Torpedos dos obstáculos do cenário: centro geométrico (z absoluto), eixo (unitário, do nariz), rotação do ator
    [roll, pitch, yaw] em graus, ponto imposto ao ator (centro + origem_eixo·eixo), tipo e índice do obstáculo.

    pilar: feixe n × n em pé (pitch 90°), ``niveis`` empilhados a partir de ``base_z`` com ``folga`` entre eles;
    viga: torpedos deitados em fila de ``de`` até ``ate`` (mesmo z), quantos couberem, centrados.
    """
    agua = float(cena['nivel_agua'])
    comp, diam = float(agente['comprimento']), float(agente['diametro'])
    folga, folga_feixe = float(agente['folga']), float(agente['folga_feixe'])
    centros, eixos, rots, tipos, idx = [], [], [], [], []
    for i, ob in enumerate(cena.get('obstaculos', [])):
        if ob['tipo'] == 'pilar':
            n = int(ob.get('feixe', 1))
            passo = diam + folga_feixe
            desloc = [(k - (n - 1) / 2.0) * passo for k in range(n)]
            for nivel in range(int(ob['niveis'])):
                z = agua + float(ob['base_z']) + nivel * (comp + folga) + comp / 2.0
                for dx in desloc:
                    for dy in desloc:
                        centros.append([float(ob['xy'][0]) + dx, float(ob['xy'][1]) + dy, z])
                        eixos.append([0.0, 0.0, 1.0])
                        rots.append([0.0, 90.0, 0.0])
                        tipos.append('pilar')
                        idx.append(i)
        elif ob['tipo'] == 'viga':
            de, ate = np.asarray(ob['de'], dtype=np.float64), np.asarray(ob['ate'], dtype=np.float64)
            if not math.isclose(de[2], ate[2]):
                raise ValueError(f"viga {i} inclinada (de {de.tolist()}, até {ate.tolist()}); use o mesmo z")
            d = ate - de
            tam = float(np.linalg.norm(d))
            u = d / tam
            n = max(1, math.ceil((tam + folga) / (comp + folga)))
            meio = (de + ate) / 2.0 + [0.0, 0.0, agua]
            yaw = math.degrees(math.atan2(u[1], u[0]))
            for k in range(n):
                centros.append((meio + (k - (n - 1) / 2.0) * (comp + folga) * u).tolist())
                eixos.append(u.tolist())
                rots.append([0.0, 0.0, yaw])
                tipos.append('viga')
                idx.append(i)
        else:
            raise ValueError(f"obstáculo {i} de tipo desconhecido: {ob['tipo']!r} (use 'pilar' ou 'viga')")
    centros = np.asarray(centros, dtype=np.float64).reshape(-1, 3)
    eixos = np.asarray(eixos, dtype=np.float64).reshape(-1, 3)
    return {'centro': centros, 'eixo': eixos, 'rotacao': np.asarray(rots, dtype=np.float64).reshape(-1, 3),
            'imposto': centros + float(agente.get('origem_eixo', 0.0)) * eixos,
            'tipo': tipos, 'obstaculo': np.asarray(idx, dtype=int),
            'meio_comprimento': max(comp / 2.0 - diam / 2.0, 0.0), 'raio': diam / 2.0}


def dist_capsulas(pos, geo: dict):
    """Menor distância de ``pos`` à superfície dos torpedos (cápsulas) e o índice do torpedo (inf e -1 sem torpedos)."""
    c = geo['centro']
    if len(c) == 0:
        return math.inf, -1
    rel = np.asarray(pos, dtype=np.float64)[None, :] - c
    t = np.clip(np.einsum('ij,ij->i', rel, geo['eixo']), -geo['meio_comprimento'], geo['meio_comprimento'])
    d = np.linalg.norm(rel - t[:, None] * geo['eixo'], axis=1) - geo['raio']
    j = int(np.argmin(d))
    return float(d[j]), j


def carregar_sensores(versao: str) -> list:
    """Sensores do Goto (``v1`` ou ``v2``) de ``config/goto_sensores.json`` (spec sensores-goto)."""
    with open(f"{CONFIG}/goto_sensores.json") as f:
        return json.load(f)[versao]


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
    # O motor do BiguaSim 1.1.0 não aplica a aceleração angular ao Hydrone e descarta acelerações lineares pequenas (spec
    # rotacao-hydrone); a integração das acelerações da dinâmica é feita aqui e a velocidade (linear e angular) é imposta a
    # cada tick. O motor amortece v e ω impostas a ~1 s⁻¹ (0,3 → 0,285 m/s e 0,524 → 0,498 rad/s em um tick de 0,05 s,
    # tools/diag_rotacao_hydrone.py).
    AMORTECIMENTO_LINEAR_MOTOR = 1.0
    AMORTECIMENTO_ANGULAR_MOTOR = 1.0
    # Parado, o corpo "dorme" no motor: velocidades impostas pequenas não o acordam e ele mede v = ω = 0 (0,05 m/s em um
    # tick acorda, 0,01 não). Abaixo destes limiares, com o motor medindo zero, o integrador segue com a própria v e ω.
    VEL_DORMINDO = 0.1       # m/s
    VEL_ANG_DORMINDO = 0.1   # rad/s

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

    # Obstáculos feitos de agentes: esperam no alto acima do lugar (níveis em alturas diferentes) e descem na vertical, um
    # nível por tick (o teleporte do motor varre o caminho e para no primeiro obstáculo)
    NOME_OBSTACULO = 'obstaculo'
    ALTURA_ESPERA = 30.0                 # m acima da água
    # os clones nascem sobrepostos e o motor os separa com violência: no lab, com 48 torpedos nascendo 28 m acima do spawn,
    # o Hydrone foi empurrado 2,5 m (para fora da área) em todo reset. Nascem longe, ao lado e bem acima
    NASCIMENTO_DESLOC = (30.0, 0.0, 50.0)   # m em relação ao spawn (z: acima da água)
    PASSO_ESPERA = 2.5                   # m entre os níveis à espera (torpedo de 2,1 m em pé)
    TOL_POSICAO = 0.3                    # m: diferença para o ponto imposto além da comum aos outros torpedos
    TENTATIVAS_POSICAO = 3
    # o set_physics_state varre o caminho e para no primeiro obstáculo, e os clones nascem sobrepostos no mesmo ponto: no
    # lab, 35 de 48 ficaram presos (2026-11-18). No posicionamento os torpedos andam com teleport (comando de outro tipo);
    # depois são segurados com set_physics_state (zera a velocidade)
    POSICIONAR_COM_TELEPORTE = True
    # cada sensor e buffer de cada agente é um arquivo em /dev/shm: com 66 torpedos o limite padrão de 1024 arquivos abertos
    # estourou no lab (OSError 24, 2026-11-18)
    LIMITE_ARQUIVOS = 65536

    N_ACOES = 3
    _COM_ACOES = True                    # ações anteriores na observação (o -v1 tem só a pose de 9, como o CUPRL espera)

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
        integrar_dinamica: bool = True,
    ) -> None:
        if integrar_dinamica and batch_size != 1:
            raise ValueError("integrar_dinamica só funciona com batch_size = 1")
        self._integrar_dinamica = integrar_dinamica
        self._acc = np.zeros(6)
        self._v, self._w = np.zeros(3), np.zeros(3)
        self._preparar_cenario(cenario)
        if self._n_obstaculos():
            self._subir_limite_arquivos()
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

    def _preparar_cenario(self, cenario: str) -> None:
        """Lê o cenário e monta a geometria dos obstáculos (torpedos)."""
        self._cena = carregar_cenario(cenario)
        self._cenario = cenario
        self._agente_obst = carregar_agente_obstaculo()
        self._geo = geometria_obstaculos(self._cena, self._agente_obst)
        self._obst_ponto = self._geo['imposto'].copy()   # onde cada torpedo é mantido (ponto de espera ou lugar final)
        self._estado_obst = None

    # ------------------------------------------------------------------
    # Construção
    # ------------------------------------------------------------------
    def _build_params(self):
        env_params, obs_params = super()._build_params()
        _id = self._agent_id(env_params, 'robot')
        env_params['agents'][_id]['sensors'].append(
            {'sensor_type': 'CollisionSensor', 'location': [0, 0, 0], 'rotation': [0, 0, 0]})
        agua = float(self._cena['nivel_agua'])
        # o mundo não exporta a superfície da água; a dinâmica do BiguaSim 1.1.0 aceita o nível em dynamics.water_level
        env_params['agents'][_id]['dynamics']['water_level'] = agua
        (x0, y0), (x1, y1) = self._cena['area']
        z0, z1 = self._cena['z_limites']
        self._bounds = np.array([[x0, y0, agua + z0], [x1, y1, agua + z1]], dtype=np.float64)
        n = self._n_obstaculos()
        if n:
            # os clones nascem todos no mesmo ponto, alto, e são levados ao lugar no reset (_posicionar_obstaculos). Só o
            # DynamicsSensor que a dinâmica do BiguaSim exige, no centro de massa (o IMUSocket fica no nariz, a ~0,86 m do
            # centro): cada sensor a mais pesa no passo (48 torpedos com dois sensores: 5,4 passos/s no lab)
            dinamica = {'sensor_type': 'DynamicsSensor', 'socket': 'IMUSocket',
                        'configuration': {'UseCOM': True, 'UseRPY': False}, 'location': [0, 0, 0], 'rotation': [0, 0, 0]}
            spawn = self._cena['spawn']
            env_params['agents'].append({
                'agent_name': self.NOME_OBSTACULO, 'agent_type': self._agente_obst['tipo'],
                'control_abstraction': 'accel', 'dynamics': {'batch_size': n, 'water_level': agua},
                'location': [float(spawn[0]) + self.NASCIMENTO_DESLOC[0], float(spawn[1]) + self.NASCIMENTO_DESLOC[1],
                             agua + self.NASCIMENTO_DESLOC[2]],
                'rotation': [0.0, 90.0, 0.0], 'sensors': [dinamica]})
        return env_params, obs_params

    @classmethod
    def _subir_limite_arquivos(cls) -> None:
        """Sobe o limite flexível de arquivos abertos do processo até LIMITE_ARQUIVOS (sem passar do rígido)."""
        mole, duro = resource.getrlimit(resource.RLIMIT_NOFILE)
        alvo = cls.LIMITE_ARQUIVOS if duro == resource.RLIM_INFINITY else min(cls.LIMITE_ARQUIVOS, duro)
        if mole != resource.RLIM_INFINITY and mole < alvo:
            resource.setrlimit(resource.RLIMIT_NOFILE, (alvo, duro))

    def _n_obstaculos(self) -> int:
        return len(self._geo['centro'])

    def _obs_dim(self) -> int:
        return 9 + (self.N_ACOES if self._COM_ACOES else 0)

    def _init_spaces(self) -> None:
        self.observation_space = spaces.Box(-np.inf, np.inf, (self._obs_dim(),), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.N_ACOES,), np.float32)

    # ------------------------------------------------------------------
    # Cenário: pilares e alvos
    # ------------------------------------------------------------------
    def _dist_obstaculo(self, pos):
        """Menor distância até a superfície de um torpedo dos obstáculos e o tipo ('pilar'/'viga'; inf e None se não há)."""
        d, j = dist_capsulas(pos, self._geo)
        return d, (self._geo['tipo'][j] if j >= 0 else None)

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
        if min(self._dist_obstaculo(spawn)[0], self._dist_exclusao(spawn)) < self.DIST_COLISAO:
            raise ValueError(f"spawn {spawn.tolist()} a menos de {self.DIST_COLISAO} m de um obstáculo ou caixa de exclusão "
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
            if self._dist_obstaculo(alvo)[0] < folga or self._dist_exclusao(alvo) < folga:
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
    # Integração da dinâmica (spec rotacao-hydrone)
    # ------------------------------------------------------------------
    def _instalar_integrador(self) -> None:
        """Envolve o step da dinâmica do BiguaSim: entrega a ela o v/ω do integrador, guarda [v̇, ω̇] (mundo) e manda
        aceleração zero ao motor."""
        modelo = self._env._dynamics_dict['robot']
        if getattr(modelo, '_step_original', None) is None:
            modelo._step_original = modelo.step
        original = modelo._step_original

        def step(estados, *a, **k):
            din = np.array(estados[0], copy=True).ravel()   # cópia: o estado do motor (e a observação) não muda
            din[3:6], din[12:15] = self._conciliar(din[3:6], din[12:15])
            acc = np.asarray(original([din], *a, **k), dtype=np.float64).reshape(-1, 6)
            self._acc = acc[0].copy()
            return np.zeros_like(acc).tolist()

        modelo.step = step

    def _conciliar(self, v_med, w_med):
        """v e ω (mundo) deste tick: os medidos pelo motor, salvo com o corpo dormindo (o motor mede zero e o integrador
        ainda não passou do limiar em que ele acorda); aí seguem os integrados, para a velocidade acumular a partir do
        repouso."""
        dormindo = (np.allclose(v_med, 0.0, atol=1e-4) and np.allclose(w_med, 0.0, atol=1e-4)
                    and np.linalg.norm(self._v) < self.VEL_DORMINDO and np.linalg.norm(self._w) < self.VEL_ANG_DORMINDO)
        if not dormindo:
            self._v, self._w = np.array(v_med, dtype=np.float64), np.array(w_med, dtype=np.float64)
        return self._v, self._w

    def _impor_estado(self, state: dict) -> None:
        """Impõe v + v̇·dt e ω + ω̇·dt (mundo) para o próximo tick, mantendo a posição e a atitude do motor."""
        dt = 1.0 / self.TICKS_POR_S
        rpy = np.asarray(state['RPYDynamicsSensor'], dtype=np.float64).ravel()   # posição do ator e atitude em graus
        self._v = self._v + self._acc[:3] * dt                                   # leitura esperada no próximo tick
        self._w = self._w + self._acc[3:] * dt
        v_novo = self._v * math.exp(self.AMORTECIMENTO_LINEAR_MOTOR * dt)
        w_novo = self._w * math.exp(self.AMORTECIMENTO_ANGULAR_MOTOR * dt)
        self._env.agents['robot-id0'].set_physics_state(rpy[6:9], rpy[15:18], v_novo, np.degrees(w_novo))

    # ------------------------------------------------------------------
    # Obstáculos feitos de agentes
    # ------------------------------------------------------------------
    def _cmds(self, action_arg):
        """Comando do step: só o do robô, ou o dicionário robô + aceleração zero a cada torpedo dos obstáculos."""
        n = self._n_obstaculos()
        if not n:
            return action_arg
        return {'robot': action_arg, self.NOME_OBSTACULO: [[0.0] * 6 for _ in range(n)]}

    def _unwrap_state(self, state: dict) -> dict:
        """Estado do robô; com os obstáculos, o estado vem por agente ({'robot': ..., 'obstaculo': ...}) e cada sensor do
        robô numa lista de um elemento (``_get_full_state`` do BiguaSim): tira da lista, como no estado de um agente só
        (a DepthCamera vinha como [dict] e o -v1 quebrava no lab, 2026-11-18)."""
        if isinstance(state, dict) and self.NOME_OBSTACULO in state:
            self._estado_obst = state[self.NOME_OBSTACULO]
            state = {k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in state['robot'].items()}
        return HoverEnv._unwrap_state(state)

    def _segurar_obstaculos(self, teleporte: bool = False) -> None:
        """Reimpõe posição e atitude de cada torpedo (parados em accel ainda derivam ~0,7 m em 5 s); com ``teleporte``,
        usa o teleport do agente (no tick vale só o último comando dado a cada agente)."""
        zero = [0.0, 0.0, 0.0]
        for j in range(self._n_obstaculos()):
            agente = self._env.agents[f'{self.NOME_OBSTACULO}-id{j}']
            if teleporte:
                agente.teleport(self._obst_ponto[j], self._geo['rotacao'][j])
            else:
                agente.set_physics_state(self._obst_ponto[j], self._geo['rotacao'][j], zero, zero)

    def _tick_obstaculos(self, teleporte: bool = False) -> dict:
        """Um tick com o robô parado (cmd_vel_yaw zero), segurando os obstáculos onde estão marcados."""
        self._segurar_obstaculos(teleporte)
        return self._env.step(self._cmds([0.0] * 4), action_repeat=1)

    def _pontos_espera(self) -> NDArray:
        """Ponto de espera de cada torpedo: acima do lugar, um nível por altura (torpedos na mesma vertical não se tocam)."""
        agua = float(self._cena['nivel_agua'])
        nivel = self._niveis_descida()
        espera = self._geo['imposto'].copy()
        espera[:, 2] = agua + self.ALTURA_ESPERA + self.PASSO_ESPERA * nivel
        return espera

    def _niveis_descida(self) -> NDArray:
        """Ordem de descida de cada torpedo: pelo z final, de baixo para cima (quem desce não passa por quem já desceu)."""
        zs = np.round(self._geo['centro'][:, 2], 3)
        return np.searchsorted(np.unique(zs), zs)

    def _posicoes_obstaculos(self):
        """Posição medida (centro de massa, DynamicsSensor[6:9]) de cada torpedo, ou None sem estado."""
        din = self._estado_obst.get('DynamicsSensor') if isinstance(self._estado_obst, dict) else None
        if din is None:
            return None
        return np.asarray(din, dtype=np.float64).reshape(self._n_obstaculos(), -1)[:, 6:9]

    def _dinamica_parada(self) -> None:
        """Troca a dinâmica do lote de obstáculos por uma que devolve aceleração zero (sem a conta do modelo a cada tick)."""
        modelo = getattr(self._env, '_dynamics_dict', {}).get(self.NOME_OBSTACULO)
        if modelo is not None:
            n = self._n_obstaculos()
            modelo.step = lambda estados, cmds, *a, **k: [[0.0] * 6 for _ in range(n)]

    def _fora_do_lugar(self) -> NDArray:
        """Índices dos torpedos fora do lugar: a diferença medida − imposto difere da mediana (o deslocamento constante
        entre a origem do ator e o ponto imposto) em mais de TOL_POSICAO."""
        medido = self._posicoes_obstaculos()
        if medido is None:
            return np.zeros(0, dtype=int)
        dif = medido - self._geo['imposto']
        erro = np.zeros(len(dif))
        # o deslocamento vai ao longo do eixo do torpedo: comum só entre os de mesma orientação
        _, grupo = np.unique(np.round(self._geo['eixo'], 3), axis=0, return_inverse=True)
        for g in np.unique(grupo):
            sel = grupo.ravel() == g
            erro[sel] = np.linalg.norm(dif[sel] - np.median(dif[sel], axis=0), axis=1)
        return np.flatnonzero(erro > self.TOL_POSICAO)

    def _posicionar_obstaculos(self):
        """Leva os torpedos do ponto de nascimento ao lugar: sobem/vão ao ponto de espera, descem um nível por tick e os
        que não chegaram repetem subida e descida. Devolve o estado do último tick (None sem obstáculos)."""
        n = self._n_obstaculos()
        if not n:
            return None
        espera = self._pontos_espera()
        nivel = self._niveis_descida()
        raw = None
        todos = np.arange(n)
        for tentativa in range(self.TENTATIVAS_POSICAO + 1):
            tele = self.POSICIONAR_COM_TELEPORTE
            self._obst_ponto[todos] = espera[todos]
            raw = self._tick_obstaculos(tele)
            for k in np.unique(nivel[todos]):
                sel = todos[nivel[todos] == k]
                self._obst_ponto[sel] = self._geo['imposto'][sel]
                raw = self._tick_obstaculos(tele)
            raw = self._tick_obstaculos()                    # zera a velocidade e mede com todos no lugar
            self._unwrap_state(raw)
            todos = self._fora_do_lugar()
            if not len(todos):
                break
        else:
            warnings.warn(f"{len(todos)} torpedo(s) dos obstáculos fora do lugar depois de {self.TENTATIVAS_POSICAO} "
                          f"tentativas: {todos.tolist()}")
        self._obst_ponto = self._geo['imposto'].copy()
        return self._robo_no_spawn()

    def _robo_no_spawn(self) -> dict:
        """Põe o Hydrone de volta no spawn, parado (teleport e depois set_physics_state), segurando os obstáculos: o que
        acontece durante o posicionamento não pode mudar o início do episódio."""
        robo = self._env.agents['robot-id0']
        zero = [0.0, 0.0, 0.0]
        robo.teleport(self._location, self._rotation)
        self._tick_obstaculos()
        robo.set_physics_state(self._location, self._rotation, zero, zero)
        return self._tick_obstaculos()

    def _passo_sim(self, action_arg) -> dict:
        """Avança ``action_repeat`` ticks; com a integração ligada ou com obstáculos, um tick por vez (impondo v e ω ao
        robô depois de cada um e reimpondo os obstáculos antes)."""
        cmds = self._cmds(action_arg)
        if not self._integrar_dinamica and not self._n_obstaculos():
            return self._env.step(cmds, action_repeat=self._action_repeat)
        for _ in range(self._action_repeat):
            self._segurar_obstaculos()
            raw = self._env.step(cmds, action_repeat=1)
            if self._integrar_dinamica:
                self._impor_estado(self._unwrap_state(raw))
        return raw

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
        d, tipo = self._dist_obstaculo(pos)
        if d < self.DIST_COLISAO:
            return tipo
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
        if not self._COM_ACOES:
            return self._pose9()
        return np.concatenate([self._pose9(), self._acao_ant]).astype(np.float32)

    def _novo_alvo(self) -> None:
        self._target = self._sortear_alvo(self._dynamics[6:9])
        self._target_list = self._target.tolist()
        self._last_norm = self._distancia()

    def _reset(self):
        state = self._unwrap_state(self._env.reset())
        if self._integrar_dinamica:
            self._instalar_integrador()
        if self._n_obstaculos():
            self._dinamica_parada()
        raw = self._posicionar_obstaculos()
        if raw is not None:
            state = self._unwrap_state(raw)
        self._acc = np.zeros(6)
        self._v, self._w = np.zeros(3), np.zeros(3)
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
        state = self._unwrap_state(self._passo_sim(action_arg))
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


# ---------------------------------------------------------------------------
# Sensores (spec sensores-goto)
# ---------------------------------------------------------------------------

class GotoPixelEnv(_PixelObsMixin, GotoEnv):
    """``-v1`` (Dranaju): RGB e profundidade embarcadas, 3 quadros de 100×100, e ``state`` = pose de 9 valores."""

    _COM_ACOES = False
    _pixel_channels = ['rgb', 'depth']

    def __init__(self, seed: int, frame_stack: int = 3, frame_size: tuple = (100, 100),
                 render_channel: str | None = None, pixel_channels: list | None = None,
                 include_state: bool = True, **kwargs) -> None:
        # o run.py manda pixel_channels/include_state ao CUPRL; aqui os canais são fixos (RGB + profundidade + pose)
        if pixel_channels is not None and list(pixel_channels) != self._pixel_channels:
            raise ValueError(f"GotoPixelEnv só entrega {self._pixel_channels}, não {list(pixel_channels)}")
        if not include_state:
            raise ValueError("GotoPixelEnv sempre inclui a pose ('state'); include_state=False não é suportado")
        self._render_channel = render_channel
        super().__init__(seed, **kwargs)
        self._setup_pixel(frame_stack, frame_size, True, render_channel)

    def _build_params(self):
        # pula o _build_params do mixin (que usaria as câmeras do pixels.json, em terceira pessoa)
        env_params, obs_params = GotoEnv._build_params(self)
        _id = self._agent_id(env_params, 'robot')
        env_params['agents'][_id]['sensors'].extend(carregar_sensores('v1'))
        return env_params, obs_params


class GotoRangeEnv(GotoEnv):
    """``-v2`` (Ricardo): 20 distâncias do LiDAR no ar ou do sonar na água + 3 ações anteriores + [d, heading, heading_z]."""

    N_LEITURAS = 20
    LIDAR_MAX = 10.0
    LIDAR_N = 160                                                   # 2,25° entre lasers
    LIDAR_IDX = [(3 + 6 * j) % 160 for j in range(-10, 10)]         # ±6,75° … ±128,25°, a cada 13,5°
    SONAR_MIN, SONAR_MAX = 0.5, 20.0
    SONAR_LIMIAR = 1e-3                                             # intensidade abaixo disso = sem retorno

    def _build_params(self):
        env_params, obs_params = super()._build_params()
        _id = self._agent_id(env_params, 'robot')
        env_params['agents'][_id]['sensors'].extend(carregar_sensores('v2'))
        return env_params, obs_params

    def _obs_dim(self) -> int:
        return self.N_LEITURAS + self.N_ACOES + 3

    @staticmethod
    def _validas(d, maximo) -> NDArray:
        d = np.asarray(d, dtype=np.float32).ravel()
        return np.where(np.isfinite(d) & (d > 0.0), np.minimum(d, maximo), maximo).astype(np.float32)

    def _lidar20(self, raw) -> NDArray:
        if raw is None:
            return np.full(self.N_LEITURAS, self.LIDAR_MAX, dtype=np.float32)
        d = self._validas(raw, self.LIDAR_MAX)
        if d.size != self.LIDAR_N:
            raise ValueError(f"LiDAR com {d.size} lasers; esperado {self.LIDAR_N} (goto_sensores.json)")
        return d[self.LIDAR_IDX]

    def _sonar20(self, raw) -> NDArray:
        """Por feixe, a distância do bin de maior intensidade; feixe sem retorno → alcance máximo."""
        if isinstance(raw, dict):
            raw = raw.get('raw')
        if raw is None:
            return np.full(self.N_LEITURAS, self.SONAR_MAX, dtype=np.float32)
        img = np.nan_to_num(np.asarray(raw, dtype=np.float32), nan=0.0)   # (RangeBins, AzimuthBins)
        n_bins = img.shape[0]
        idx = np.argmax(img, axis=0)
        d = self.SONAR_MIN + (idx + 0.5) * (self.SONAR_MAX - self.SONAR_MIN) / n_bins
        d = np.where(img.max(axis=0) > self.SONAR_LIMIAR, d, self.SONAR_MAX)
        return self._validas(d, self.SONAR_MAX)

    def _wrap_state(self, state: dict) -> NDArray:
        pose = self._pose9()
        no_ar = float(self._dynamics[8]) >= float(self._cena['nivel_agua'])
        dist = self._lidar20(state.get('Lidar')) if no_ar else self._sonar20(state.get('Sonar'))
        raw = state.get('CameraView')
        if raw is not None:
            self._last_render_frame = np.asarray(raw, dtype=np.uint8)[:, :, 0:3]
        return np.concatenate([dist, self._acao_ant, [pose[2], pose[0], pose[1]]]).astype(np.float32)
