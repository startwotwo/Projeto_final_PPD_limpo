"""
Presets elaborados de condições iniciais — galáxias, colisões, órbitas
=====================================================================
Mais interessantes que a esfera uniforme aleatória: geram estruturas
com rotação, discos, plummer estável e encontros parabólicos.

Todas as funções retornam (pos, vel, masses, labels) onde labels é
opcional (array int para colorir por componente/galáxia).
Unidades: G=1 normalizadas, mas preservam escala física via M_total
e tamanhos característicos.

Referências para implementação:
- Plummer sampling: Aarseth, Henon & Wielen (1974) — r = a / sqrt(U^{-2/3}-1),
  velocidade via rejeição q^2*(1-q^2)^{3.5}
- Disco exponencial: pdf(R) ∝ R exp(-R/R_d), amostrado por rejeição
- Velocidade circular softened: v^2 = G*M_enc*R^2/(R^2+eps^2)^{3/2}
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vec3 = npt.NDArray[np.float64]

G_DEFAULT = 1.0
EPS_DEFAULT = 0.02  # menor para galáxias, mas não zero


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _random_unit_vectors(rng: np.random.Generator, N: int) -> Vec3:
    """N vetores unitários isotrópicos."""
    v = rng.normal(size=(N, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return v / norms


def _plummer_radii(rng: np.random.Generator, a: float, N: int) -> npt.NDArray[np.float64]:
    X = rng.random(N)
    X = np.clip(X, 1e-8, 1 - 1e-8)
    r = a / np.sqrt(X ** (-2.0 / 3.0) - 1.0)
    # clip cauda extrema (1% mais distante) para visualização não explodir limites
    return np.clip(r, 0, 5.0 * a)


def _plummer_velocities(rng: np.random.Generator, r: npt.NDArray[np.float64], a: float, G: float, M: float) -> Vec3:
    """
    Amostra velocidades Plummer isotrópicas (Aarseth 1974).
    Para cada partícula com raio r, sorteia magnitude via rejeição
    e direção isotrópica.
    """
    N = r.shape[0]
    # velocidade de escape local: sqrt(2|Phi|)
    # Phi = -G*M / sqrt(r^2 + a^2)
    ve = np.sqrt(2.0 * G * M / np.sqrt(r * r + a * a))  # (N,)
    q = np.zeros(N)
    # rejeição para cada partícula
    for i in range(N):
        while True:
            q_try = rng.random()
            g = rng.random()  # uniform 0,1
            # função de distribuição: g(q)= q^2 * (1-q^2)^{3.5}
            # máximo ~0.1 em q~0.64, então aceita se g < g(q)/0.1
            # implementação clássica compara g < q**2*(1-q**2)**3.5  com g*0.1
            # simplificado: usa limiar 0.1
            if g < q_try * q_try * (1.0 - q_try * q_try) ** 3.5 / 0.1:
                # evita intercalação: na verdade algoritmo original compara
                # 0.1*g < q^2*(1-q^2)^{3.5}
                pass
            # Condição correta clássica: 0.1 * g < q^2*(1-q^2)^{3.5}
            # Re-implementa corretamente:
            # Vamos usar forma padrão encontrada em códigos:
            #   if 0.1*random() < q**2*(1-q**2)**3.5: accept
            # Então precisamos re-amostrar g como 0.1*U
            # Para não complicar, fazemos:
            #   if rng.random()*0.1 < q_try**2*(1-q_try**2)**3.5:
            # Mas já temos g; vamos apenas usar g*0.1
            if 0.1 * g < q_try * q_try * (1.0 - q_try * q_try) ** 3.5:
                q[i] = q_try
                break
    v_mag = q * ve  # (N,)
    dirs = _random_unit_vectors(rng, N)
    return dirs * v_mag[:, None]


def _rotation_matrix(axis: str, angle_rad: float) -> npt.NDArray[np.float64]:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    raise ValueError(axis)


def _varied_masses(rng: np.random.Generator, N: int, M_total: float, scatter: float = 0.6) -> npt.NDArray[np.float64]:
    """Massas lognormais com média 1, desvio scatter, normalizadas para M_total. scatter=0 → uniforme."""
    if scatter <= 1e-9 or N == 0:
        return np.full(N, M_total / max(1, N), dtype=np.float64)
    # lognormal com sigma=scatter, mean=0 → média exp(sigma²/2)
    m = rng.lognormal(mean=0.0, sigma=scatter, size=N)
    m *= M_total / m.sum()
    # clip extremos para não gerar buraco negro acidental
    m = np.clip(m, M_total / N * 0.2, M_total / N * 4.0)
    # renormaliza após clip
    m *= M_total / m.sum()
    return m.astype(np.float64)


# ------------------------------------------------------------------
# Plummer sphere (aglomerado estável, virializado)
# ------------------------------------------------------------------

def generate_plummer_sphere(
    N: int,
    M_total: float = 1.0,
    a: float = 0.5,
    G: float = G_DEFAULT,
    seed: int = 42,
    mass_scatter: float = 0.6,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64]]:
    """
    Esfera de Plummer virializada — massas lognormais (scatter) para tamanho/gravidade variados.
    """
    rng = np.random.default_rng(seed)
    r = _plummer_radii(rng, a, N)
    dirs = _random_unit_vectors(rng, N)
    pos = dirs * r[:, None]

    vel = _plummer_velocities(rng, r, a, G, M_total)

    masses = _varied_masses(rng, N, M_total, scatter=mass_scatter)

    # remove COM drift
    vel -= np.average(vel, axis=0, weights=masses)
    pos -= np.average(pos, axis=0, weights=masses)

    # fator de escala para virial: Energia cinética = |Epot|/2 em equilíbrio
    # Plummer analítico já está virializado, mas reescala leve para garantir
    # devido a sampling e softening
    # calcula energia aproximada e reescala velocidades para 2*K = |W|
    try:
        from .nbody_sequential import total_energy
    except ImportError:
        from nbody_sequential import total_energy  # fallback quando importado como top-level (Colab)
    # energia com softening pequeno para estimativa
    ekin = 0.5 * np.sum(masses * np.sum(vel * vel, axis=1))
    # estima pot via fórmula analítica aproximada: -3*pi*G*M^2/(64*a) ~...
    # em vez de calcular O(N^2), apenas não reescala se já próximo
    # Aqui mantemos como está; opção de reescalar:
    # vel *= 0.9  # leve ajuste se necessário
    return pos.astype(np.float64), vel.astype(np.float64), masses


# ------------------------------------------------------------------
# Disco exponencial + bulbo Plummer = Galáxia espiral
# ------------------------------------------------------------------

def generate_exponential_disk(
    rng: np.random.Generator,
    N: int,
    M_disk: float,
    R_d: float,
    z_d: float,
    M_bulge: float,
    a_bulge: float,
    G: float,
    eps: float,
    vel_scatter: float = 0.05,
) -> tuple[Vec3, Vec3]:
    """
    Gera disco fino exponencial com rotação aproximadamente Kepleriana
    balanceada pelo bulbo + disco. Retorna pos, vel sem massas.
    """
    # Amostragem de R via rejeição: pdf(R) ~ R*exp(-R/R_d)
    Rmax = 4.5 * R_d
    pmax = R_d * np.exp(-1.0)  # máximo em R=R_d
    Rs = np.zeros(N)
    count = 0
    # vetorizado por lotes
    while count < N:
        batch = max(1024, N - count)
        R_try = rng.uniform(0, Rmax, size=batch)
        p = R_try * np.exp(-R_try / R_d)
        accept = rng.random(batch) < (p / pmax)
        n_acc = int(np.sum(accept))
        take = min(n_acc, N - count)
        if take > 0:
            Rs[count : count + take] = R_try[accept][:take]
            count += take

    theta = rng.uniform(0, 2 * np.pi, size=N)
    # z: laplace (exponencial dupla) ou gauss fino
    # usa normal para simplicidade
    z = rng.normal(0, z_d, size=N)
    # alternativa: rng.laplace(0, z_d, N) mais cuspidada

    x = Rs * np.cos(theta)
    y = Rs * np.sin(theta)
    pos = np.stack([x, y, z], axis=1)

    # Velocidade circular: v_circ^2 = G*M_enc*R^2/(R^2+eps^2)^{3/2}
    # M_enc = M_bulge_enc + M_disk_enc
    # Bulge Plummer enclosed:
    M_bulge_enc = M_bulge * (Rs ** 3) / (Rs * Rs + a_bulge * a_bulge) ** 1.5
    # Disco exponencial enclosed (aprox. massa dentro de R):
    # M_disk(<R) = M_disk * [1 - (1+R/R_d)*exp(-R/R_d)]
    M_disk_enc = M_disk * (1.0 - (1.0 + Rs / R_d) * np.exp(-Rs / R_d))
    Menc = M_bulge_enc + M_disk_enc
    # Evita divisão por zero no centro
    denom = (Rs * Rs + eps * eps) ** 1.5
    # v^2 = G*Menc*R^2 / denom  ??? mas Menc já inclui dependencia R
    # Na verdade para potencial softened, v^2 = G*Menc*R^2 / (R^2+eps^2)^{3/2}
    # Para R=0, v=0 — evita explosão.
    # Para R>>eps, recupera G*Menc/R
    v2 = G * Menc * Rs * Rs / np.maximum(denom, 1e-12)
    v_circ = np.sqrt(np.maximum(v2, 0.0))

    # direção tangencial: (-sin theta, cos theta)
    vx = -v_circ * np.sin(theta)
    vy = v_circ * np.cos(theta)
    vz = np.zeros(N)

    # Dispersão para não ser perfeitamente frio (evita instabilidade)
    scatter = rng.normal(0, 1, size=(N, 3))
    # escala scatter proporcional a v_circ
    scatter[:, 0] *= vel_scatter * v_circ
    scatter[:, 1] *= vel_scatter * v_circ
    scatter[:, 2] *= vel_scatter * np.mean(v_circ) * 0.5  # vertical menor

    vel = np.stack([vx, vy, vz], axis=1) + scatter
    return pos, vel


def generate_spiral_galaxy(
    N_disk: int = 700,
    N_bulge: int = 300,
    M_total: float = 1.0,
    R_d: float = 1.0,
    a_bulge: float = 0.4,
    z_d: float = 0.08,
    bulge_frac: float = 0.25,
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
    seed: int = 42,
    with_bh: bool = False,
    bh_frac: float = 0.03,
    vel_scatter: float = 0.04,
    mass_scatter: float = 0.6,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """
    Galáxia espiral — massas lognormais por componente, tamanho ∝ massa^(1/3) no plot.
    """
    rng = np.random.default_rng(seed)
    if with_bh:
        M_bh = bh_frac * M_total
        M_rem = M_total - M_bh
    else:
        M_bh = 0.0
        M_rem = M_total
    M_bulge = bulge_frac * M_rem
    M_disk = (1.0 - bulge_frac) * M_rem if not with_bh else (1.0 - bulge_frac) * M_rem

    # Bulbo: Plummer
    # Reusa gerador de Plummer mas precisamos só posições/velocidades do bulbo
    # Gera esfera Plummer com massa M_bulge
    # Para não duplicar código, gera via helper
    r_bulge = _plummer_radii(rng, a_bulge, N_bulge)
    dirs_b = _random_unit_vectors(rng, N_bulge)
    pos_bulge = dirs_b * r_bulge[:, None]
    vel_bulge = _plummer_velocities(rng, r_bulge, a_bulge, G, M_bulge + M_disk * 0.5)  # inclui potencial do disco aproximadamente
    # reduz dispersão do bulbo para não escapar (fator 0.7)
    vel_bulge *= 0.7

    # Disco
    pos_disk, vel_disk = generate_exponential_disk(
        rng, N_disk, M_disk, R_d, z_d, M_bulge, a_bulge, G, eps, vel_scatter=vel_scatter
    )

    # massas lognormais por componente (variadas, influenciam gravidade e tamanho no plot)
    masses_bulge = _varied_masses(rng, N_bulge, M_bulge, scatter=mass_scatter)
    masses_disk = _varied_masses(rng, N_disk, M_disk, scatter=mass_scatter)
    pos = np.vstack([pos_bulge, pos_disk])
    vel = np.vstack([vel_bulge, vel_disk])
    masses = np.concatenate([masses_bulge, masses_disk])
    labels = np.concatenate([np.zeros(N_bulge, dtype=np.int32), np.ones(N_disk, dtype=np.int32)])

    if with_bh and bh_frac > 0:
        pos_bh = np.zeros((1, 3), dtype=np.float64)
        vel_bh = np.zeros((1, 3), dtype=np.float64)
        masses_bh = np.array([M_bh], dtype=np.float64)
        labels_bh = np.array([2], dtype=np.int32)
        pos = np.vstack([pos, pos_bh])
        vel = np.vstack([vel, vel_bh])
        masses = np.concatenate([masses, masses_bh])
        labels = np.concatenate([labels, labels_bh])

    # Centraliza COM e remove momento
    com_pos = np.average(pos, axis=0, weights=masses)
    pos -= com_pos
    com_vel = np.average(vel, axis=0, weights=masses)
    vel -= com_vel

    return pos.astype(np.float64), vel.astype(np.float64), masses.astype(np.float64), labels


# ------------------------------------------------------------------
# Colisão de galáxias
# ------------------------------------------------------------------

def generate_galaxy_collision(
    N_disk1: int = 600,
    N_bulge1: int = 200,
    N_disk2: int = 600,
    N_bulge2: int = 200,
    M_total1: float = 1.0,
    M_total2: float = 1.0,
    R_d: float = 1.0,
    a_bulge: float = 0.4,
    z_d: float = 0.08,
    separation: float = 5.0,
    impact_param: float = 0.4,
    rel_vel_factor: float = 0.5,
    tilt1_deg: float = 0.0,
    tilt2_deg: float = 30.0,
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
    seed: int = 42,
    with_bh: bool = False,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """
    Duas galáxias espirais em rota de colisão.
    - separation: distância inicial entre centros (unidades de R_d)
    - impact_param: deslocamento perpendicular em y (parâmetro de impacto)
    - rel_vel_factor: fração da velocidade parabólica (1=parabólica, <1 ligada, >1 hiperbólica)
    - tilt: inclinação do disco em graus (rotação em torno de x)

    Retorna pos, vel, masses, labels (0=gal1 bulge,1=gal1 disk,2=gal2 bulge,3=gal2 disk)
    """
    # Galáxia 1 e 2 com seeds diferentes para não serem idênticas
    pos1, vel1, m1, lab1 = generate_spiral_galaxy(
        N_disk=N_disk1, N_bulge=N_bulge1, M_total=M_total1, R_d=R_d, a_bulge=a_bulge, z_d=z_d,
        G=G, eps=eps, seed=seed, with_bh=with_bh
    )
    pos2, vel2, m2, lab2 = generate_spiral_galaxy(
        N_disk=N_disk2, N_bulge=N_bulge2, M_total=M_total2, R_d=R_d, a_bulge=a_bulge, z_d=z_d,
        G=G, eps=eps, seed=seed + 101, with_bh=with_bh
    )
    # Ajusta labels da segunda galáxia para 2,3
    lab2 = lab2 + 2
    if with_bh:
        # lab 2 no segundo caso seria BH -> 4, mas mantém mapeamento simples
        # Re-mapeia: 2->4 para BH segunda galáxia se existir
        lab2 = np.where(lab2 == 4, 4, lab2)  # já é 4

    # Aplica inclinações
    if tilt1_deg != 0:
        Rm = _rotation_matrix("x", np.deg2rad(tilt1_deg))
        pos1 = pos1 @ Rm.T
        vel1 = vel1 @ Rm.T
    if tilt2_deg != 0:
        Rm = _rotation_matrix("x", np.deg2rad(tilt2_deg))
        # adiciona rotação em z para não ser coplanar
        Rm2 = _rotation_matrix("z", np.deg2rad(20))
        R = Rm2 @ Rm
        pos2 = pos2 @ R.T
        vel2 = vel2 @ R.T

    # Separa no eixo x, com impacto em y
    pos1[:, 0] -= separation / 2.0
    pos1[:, 1] -= impact_param / 2.0
    pos2[:, 0] += separation / 2.0
    pos2[:, 1] += impact_param / 2.0

    # Velocidade relativa parabólica: sqrt(2 G (M1+M2)/sep)
    Msum = M_total1 + M_total2
    v_parab = np.sqrt(2.0 * G * Msum / max(separation, 0.1))
    v_rel = rel_vel_factor * v_parab

    # Direção de aproximação: ao longo de -x para gal1, +x para gal2, com leve componente y
    # Divide momento proporcional à massa para manter COM parado
    # v1 = +v_rel * M2/(M1+M2)  em +x? Na verdade gal1 em -x deve ir para +x
    v1_bulk = np.array([+v_rel * M_total2 / Msum, 0.0, 0.0])
    v2_bulk = np.array([-v_rel * M_total1 / Msum, 0.0, 0.0])

    # Adiciona leve velocidade transversal para impacto não frontal (conserva momento)
    # Pequeno shear em y proporcional a impact_param
    v_shear = v_rel * 0.1
    v1_bulk[1] += v_shear * 0.5
    v2_bulk[1] -= v_shear * 0.5

    vel1 += v1_bulk
    vel2 += v2_bulk

    pos = np.vstack([pos1, pos2])
    vel = np.vstack([vel1, vel2])
    masses = np.concatenate([m1, m2])
    labels = np.concatenate([lab1, lab2])

    # Remove COM residual (deve ser ~0 já)
    com_pos = np.average(pos, axis=0, weights=masses)
    pos -= com_pos
    com_vel = np.average(vel, axis=0, weights=masses)
    vel -= com_vel

    return pos, vel, masses, labels


# ------------------------------------------------------------------
# Órbitas Keplerianas e discos
# ------------------------------------------------------------------

def generate_kepler_binary(
    M1: float = 0.5,
    M2: float = 0.5,
    separation: float = 1.0,
    eccentricity: float = 0.0,
    G: float = G_DEFAULT,
    seed: int = 42,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64]]:
    """
    Binária de 2 corpos em órbita Kepleriana.
    eccentricity 0=círculo, 0.6=elipse bem excêntrica.
    Coloca no plano xy, pericentro em +x.
    """
    Mtot = M1 + M2
    # semi-eixo maior a = separation (para e=0, r=a)
    a = separation
    # distância no instante inicial: escolhe apocentro ou pericentro?
    # Vamos colocar em pericentro para e>0: r = a*(1-e)
    r = a * (1.0 - eccentricity)
    # velocidade no pericentro: v = sqrt(G*Mtot/a * (1+e)/(1-e))
    # fórmula vis-viva: v^2 = G*Mtot*(2/r -1/a)
    v2 = G * Mtot * (2.0 / r - 1.0 / a) if r > 1e-12 else 0.0
    v = np.sqrt(max(v2, 0.0))

    # posições relativas ao COM
    # r1 = -M2/Mtot * r_vec, r2 = +M1/Mtot * r_vec
    # coloca ao longo de x
    r_vec = np.array([r, 0.0, 0.0])
    pos = np.zeros((2, 3), dtype=np.float64)
    pos[0] = -M2 / Mtot * r_vec
    pos[1] = +M1 / Mtot * r_vec

    # velocidades perpendiculares (ao longo de +y no pericentro)
    # v1 = -M2/Mtot * v, v2 = +M1/Mtot * v  mas direções opostas para conservar momento
    # No pericentro, velocidade é perpendicular ao vetor posição (ao longo de y)
    vel = np.zeros((2, 3), dtype=np.float64)
    vel[0] = np.array([0, -M2 / Mtot * v, 0])  # invertido? verifica momento
    # Na verdade para momento zero: M1*v1 + M2*v2 =0 => v1 = -M2/M1 * v2
    # Se v2 = + (M1/Mtot)*v ??? Vamos fazer sistemático:
    # velocidade relativa v_rel = v, então v1 = -M2/Mtot * v_rel, v2 = +M1/Mtot * v_rel
    # com v_rel ao longo de y
    v_rel_vec = np.array([0, v, 0])
    vel[0] = -M2 / Mtot * v_rel_vec
    vel[1] = +M1 / Mtot * v_rel_vec

    masses = np.array([M1, M2], dtype=np.float64)
    return pos, vel, masses


def generate_kepler_disk(
    N: int = 500,
    M_central: float = 1.0,
    m_particle: float = 1e-4,
    R_min: float = 0.5,
    R_max: float = 2.5,
    eccentricity: float = 0.0,
    inclination_deg: float = 5.0,
    G: float = G_DEFAULT,
    eps: float = 0.02,
    seed: int = 42,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64]]:
    """
    Disco protoplanetário / anel Kepleriano ao redor de massa central.
    N partículas em órbitas aproximadamente Keplerianas com leve
    excentricidade e inclinação aleatória. Inclui partícula central.
    """
    rng = np.random.default_rng(seed)
    # distribuição radial uniforme em área (R ~ sqrt(U)) para densidade ~1/R ?
    # Para disco uniforme em área: R = sqrt(U*(Rmax^2 - Rmin^2) + Rmin^2)
    U = rng.random(N)
    R = np.sqrt(U * (R_max * R_max - R_min * R_min) + R_min * R_min)
    theta = rng.uniform(0, 2 * np.pi, size=N)
    # leve inclinação
    inc = np.deg2rad(rng.normal(0, inclination_deg, size=N))
    # pequenas excentricidades
    ecc = rng.uniform(0, eccentricity, size=N) if eccentricity > 0 else np.zeros(N)

    # posição inicial: assume pericentro ou fase aleatória
    # para simplicidade, coloca em ângulo theta com raio corrigido por ecc
    # r = a*(1-e^2)/(1+e*cos(f)) ; escolhe f = theta aleatório
    f = theta  # anomalia verdadeira aleatória
    r = R * (1.0 - ecc * ecc) / (1.0 + ecc * np.cos(f) + 1e-12)

    x = r * np.cos(f)
    y = r * np.sin(f) * np.cos(inc)  # inclinação projeta y/z
    z = r * np.sin(f) * np.sin(inc) * 0.2 + rng.normal(0, 0.02, size=N)  # fino

    # velocidade Kepleriana softened: v^2 = G*M*r^2/(r^2+eps^2)^{3/2} * (2/r -1/a) fator
    # Para órbita circular: v_circ = sqrt(G*M*r^2/(r^2+eps^2)^{3/2})
    # Para elíptica: vis-viva com r atual
    a_orbit = R  # semi-eixo
    # vis-viva: v^2 = G*M*(2/r -1/a)
    # com softening, aproxima como v^2_soft = v^2 * r^3/(r^2+eps^2)^{3/2}
    v2_newton = G * M_central * (2.0 / np.maximum(r, 1e-6) - 1.0 / np.maximum(a_orbit, 1e-6))
    v2_newton = np.maximum(v2_newton, 1e-12)
    soft = (r * r) / (r * r + eps * eps)  # aprox soft factor? use R^2/(R^2+eps^2) ~1
    # mais preciso: fator (r^2)/(r^2+eps^2)^{3/2} * r  => r^3/(r^2+eps^2)^{3/2}
    soft2 = (r ** 3) / (r * r + eps * eps) ** 1.5
    v = np.sqrt(v2_newton * soft2)

    # direção tangencial: perpendicular a r_vec no plano orbital
    # para órbita no plano xy, v = (-sin f, cos f) * v * correção de excentricidade
    # Vetor velocidade em órbita Kepleriana: depende de e e f, mas aproxima circular + scatter
    vx = -np.sin(f) * v
    vy = np.cos(f) * v * np.cos(inc)
    vz = np.cos(f) * v * np.sin(inc) * 0.2

    # adiciona leve dispersão por excentricidade
    vel_scatter = rng.normal(0, 0.02, size=(N, 3)) * v[:, None]

    pos_particles = np.stack([x, y, z], axis=1) + vel_scatter * 0.01
    vel_particles = np.stack([vx, vy, vz], axis=1) + vel_scatter

    # partícula central
    pos_central = np.zeros((1, 3), dtype=np.float64)
    vel_central = np.zeros((1, 3), dtype=np.float64)
    # Ajusta velocidade central para conservar momento
    # M_central * v_central + sum(m * v) =0
    m_part = np.full(N, m_particle, dtype=np.float64)
    total_mom = np.sum(m_part[:, None] * vel_particles, axis=0)
    vel_central[0] = -total_mom / M_central

    pos = np.vstack([pos_central, pos_particles])
    vel = np.vstack([vel_central, vel_particles])
    masses = np.concatenate([np.array([M_central], dtype=np.float64), m_part])

    # COM já ~0
    return pos.astype(np.float64), vel.astype(np.float64), masses


# ------------------------------------------------------------------
# Presets novos — para galeria expandida
# ------------------------------------------------------------------

def generate_cold_collapse(
    N: int = 1000,
    radius: float = 1.5,
    M_total: float = 1.0,
    seed: int = 42,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64]]:
    """Esfera fria: uniforme em volume, velocidades zero — colapso violento."""
    rng = np.random.default_rng(seed)
    dirs = _random_unit_vectors(rng, N)
    r = radius * np.cbrt(rng.random(N))
    pos = dirs * r[:, None]
    vel = np.zeros((N, 3), dtype=np.float64)
    masses = np.full(N, M_total / N, dtype=np.float64)
    # leve ruído para quebrar simetria perfeita
    vel += rng.normal(0, 1e-4, size=(N, 3))
    pos -= np.average(pos, axis=0, weights=masses)
    vel -= np.average(vel, axis=0, weights=masses)
    return pos, vel, masses


def generate_forming_galaxy(
    N: int = 1000,
    radius: float = 1.0,
    M_total: float = 1.0,
    spin: float = 0.15,
    flatten: float = 0.6,
    G: float = G_DEFAULT,
    seed: int = 42,
    mass_scatter: float = 0.6,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """
    Formação de galáxia: nuvem densa uniforme (R=1.0) com leve rotação sólida ao redor de z.
    Sem dissipação de gás o disco fica espesso, mas com spin 0.15–0.3 e colapso frio
    há achatamento claro por conservação de momento angular (z colapsa, R sustenta).
    flatten=0.6 achata levemente a nuvem inicial em z para acelerar formação do disco.
    Retorna labels 0 para colorir disco em formação vs halo.
    """
    rng = np.random.default_rng(seed)
    dirs = _random_unit_vectors(rng, N)
    r = radius * np.cbrt(rng.random(N))
    pos = dirs * r[:, None]
    # achata inicial em z (nuvem já levemente oblata)
    pos[:, 2] *= flatten
    # rotação sólida: v = ω × r, ω = spin * v_circ(R=1)/1
    # v_circ ~ sqrt(GM/R) ~1 para M=1 R=1
    omega = spin * np.sqrt(G * M_total / radius)
    vel = np.zeros((N, 3), dtype=np.float64)
    vel[:, 0] = -omega * pos[:, 1]
    vel[:, 1] =  omega * pos[:, 0]
    # dispersão pequena para não ser perfeitamente frio (evita colapso singular)
    vel += rng.normal(0, 0.03, size=(N, 3))
    masses = _varied_masses(rng, N, M_total, scatter=mass_scatter)
    labels = np.zeros(N, dtype=np.int32)
    # remove COM drift mas mantém momento angular líquido
    com_pos = np.average(pos, axis=0, weights=masses)
    pos -= com_pos
    # remove apenas velocidade média, preservando Lz
    com_vel = np.average(vel, axis=0, weights=masses)
    vel -= com_vel
    return pos.astype(np.float64), vel.astype(np.float64), masses, labels


def generate_rotating_plummer(
    N: int = 800,
    M_total: float = 1.0,
    a: float = 0.6,
    spin: float = 1,
    G: float = G_DEFAULT,
    seed: int = 42,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64]]:
    """Plummer com rotação líquida ao redor de z — ótimo para ver achatamento."""
    pos, vel, masses = generate_plummer_sphere(N, M_total, a, G, seed)
    # adiciona rotação: v_phi = spin * v_circ(R)  onde R = sqrt(x²+y²)
    R = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
    # massa encerrada aproximada Plummer
    Menc = M_total * (R**3) / (R**2 + a**2) ** 1.5
    v_circ = np.sqrt(G * Menc * R**2 / (R**2 + 0.02**2) ** 1.5 + 1e-12)
    theta = np.arctan2(pos[:, 1], pos[:, 0])
    vel[:, 0] += -spin * v_circ * np.sin(theta)
    vel[:, 1] += spin * v_circ * np.cos(theta)
    vel -= np.average(vel, axis=0, weights=masses)
    return pos, vel, masses


def generate_ring_galaxy(
    N_disk: int = 800,
    N_bulge: int = 200,
    M_target: float = 1.0,
    M_intruder: float = 0.3,
    a_intruder: float = 0.2,
    impact_offset: float = 0.2,
    v_intruder: float = 1.5,
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
    seed: int = 42,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """
    Cartwheel: disco alvo + intruso compacto passando pelo centro na vertical.
    Gera anel em expansão — visual espetacular.
    """
    rng = np.random.default_rng(seed)
    pos_t, vel_t, m_t, lab_t = generate_spiral_galaxy(N_disk=N_disk, N_bulge=N_bulge, M_total=M_target, seed=seed, eps=eps, G=G)
    # intruso: Plummer compacto
    r_i = _plummer_radii(rng, a_intruder, 300)
    dirs_i = _random_unit_vectors(rng, 300)
    pos_i = dirs_i * r_i[:, None]
    vel_i = _plummer_velocities(rng, r_i, a_intruder, G, M_intruder) * 0.5
    # coloca intruso acima do disco, deslocado levemente do eixo
    pos_i[:, 2] += 4.0
    pos_i[:, 0] += impact_offset
    # velocidade vertical para atravessar o disco
    vel_i[:, 2] -= v_intruder
    # adiciona velocidade de queda livre aproximada
    # centraliza intruso para não ter momento do seu COM interno
    m_i = np.full(300, M_intruder / 300)
    pos_i -= np.average(pos_i, axis=0, weights=m_i)
    # recoloca acima
    pos_i[:, 2] += 4.0
    pos_i[:, 0] += impact_offset
    vel_i -= np.average(vel_i, axis=0, weights=m_i)
    vel_i[:, 2] -= v_intruder

    pos = np.vstack([pos_t, pos_i])
    vel = np.vstack([vel_t, vel_i])
    masses = np.concatenate([m_t, m_i])
    labels = np.concatenate([lab_t, np.full(300, 3, dtype=np.int32)])
    # remove COM total
    pos -= np.average(pos, axis=0, weights=masses)
    vel -= np.average(vel, axis=0, weights=masses)
    return pos, vel, masses, labels


def generate_triple_merger(
    N_per_galaxy: int = 500,
    M_total: float = 1.0,
    separation: float = 5.0,
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
    seed: int = 42,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Três galáxias em triângulo, convergindo para o centro — fusão tripla."""
    rng = np.random.default_rng(seed)
    galaxies = []
    labels_all = []
    masses_all = []
    for k in range(3):
        angle = 2 * np.pi * k / 3
        pos, vel, masses, labels = generate_spiral_galaxy(
            N_disk=int(N_per_galaxy * 0.7), N_bulge=int(N_per_galaxy * 0.3),
            M_total=M_total, seed=seed + k * 17, eps=eps, G=G
        )
        # posição em círculo
        cx, cy = separation * np.cos(angle), separation * np.sin(angle)
        pos[:, 0] += cx
        pos[:, 1] += cy
        # velocidade radial para o centro (ligada)
        Msum = 3 * M_total
        v_parab = np.sqrt(2 * G * Msum / separation) * 0.35
        # vetor para centro
        vx = -v_parab * np.cos(angle) + rng.normal(0, 0.05)
        vy = -v_parab * np.sin(angle) + rng.normal(0, 0.05)
        vel[:, 0] += vx
        vel[:, 1] += vy
        # rotaciona disco aleatoriamente
        Rm = _rotation_matrix("z", rng.uniform(0, 2 * np.pi))
        pos = pos @ Rm.T
        vel = vel @ Rm.T
        galaxies.append((pos, vel))
        masses_all.append(masses)
        labels_all.append(labels + k * 2)
    pos = np.vstack([g[0] for g in galaxies])
    vel = np.vstack([g[1] for g in galaxies])
    masses = np.concatenate(masses_all)
    labels = np.concatenate(labels_all)
    pos -= np.average(pos, axis=0, weights=masses)
    vel -= np.average(vel, axis=0, weights=masses)
    return pos, vel, masses, labels


import numpy as np
import numpy.typing as npt

# Definição de tipos baseados na assinatura da sua função
Vec3 = npt.NDArray[np.float64]  # Array de formato (N, 3)

G_DEFAULT = 1.0
EPS_DEFAULT = 1e-4

def generate_solar_system(
    G: float = G_DEFAULT,
    eps: float = 0.01,
    seed: int = 42,
    with_moons: bool = True,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """
    Sistema Solar com massas, órbitas, inclinações e velocidades escaladas.
    Retorna: (posições, velocidades, massas, tipos_dos_corpos)
    Tipos: 0 = Estrela, 1 = Planeta, 2 = Lua
    """
    rng = np.random.default_rng(seed)
    
    # 1. Parâmetros Base (Escala Adimensional)
    M_sun = 0.998
    
    # Massas relativas dos planetas (M_planet / M_sun)
    m_rel = {
        "mercury": 1.65e-7, "venus": 2.44e-6, "earth": 3.00e-6, "mars": 3.21e-7,
        "jupiter": 9.54e-4, "saturn": 2.85e-4, "uranus": 4.37e-5, "neptune": 5.15e-5,
    }
    
    # Massas relativas das luas em relação ao seu planeta mãe + anões
    m_moon_rel = {
        "moon": 0.0123, "phobos": 1.6e-8, "deimos": 2.3e-9,
        "io": 4.70e-5, "europa": 2.52e-5, "ganymede": 7.81e-5, "callisto": 5.66e-5,
        "amalthea": 1.08e-8, "himalia": 2.2e-8, "elara": 4.5e-9, "pasiphae": 1.5e-9,
        "mimas": 6.6e-8, "tethys": 1.08e-6, "dione": 1.9e-6, "rhea": 4.06e-6, "iapetus": 3.18e-6,
        "titan": 2.37e-4, "enceladus": 1.9e-7, "miranda": 1.2e-7, "ariel": 2.4e-6, "umbriel": 2.0e-6, "oberon": 5.2e-6,
        "titania": 1.8e-7, "nereid": 1.6e-7, "triton": 2.09e-4,
        "pluto": 6.6e-9, "eris": 8.4e-9, "haumea": 2.0e-9, "makemake": 1.5e-9, "ceres": 4.7e-10,
    }
    
    a_ua = {"mercury":0.039, "venus":0.072, "earth":0.10, "mars":0.152, "jupiter":0.52, "saturn":0.95, "uranus":1.92, "neptune":3.01}
    a_moon = {
        "moon":0.00055, "phobos":0.00025, "deimos":0.00035,
        "io":0.0020, "europa":0.0030, "ganymede":0.0045, "callisto":0.0065,
        "amalthea":0.0012, "himalia":0.008, "elara":0.010, "pasiphae":0.012,
        "mimas":0.0014, "tethys":0.0022, "dione":0.0028, "rhea":0.0040, "iapetus":0.012,
        "titan":0.0065, "enceladus":0.0010, "miranda":0.0010, "ariel":0.0015, "umbriel":0.0022, "oberon":0.0040,
        "titania":0.0030, "nereid":0.008, "triton":0.0022,
        "pluto":3.95, "eris":6.80, "haumea":4.30, "makemake":4.55, "ceres":0.277,
    }
    planets = ["mercury","venus","earth","mars","jupiter","saturn","uranus","neptune"]
    moons = {
        "earth": ["moon"], "mars": ["phobos","deimos"],
        "jupiter": ["io","europa","ganymede","callisto","amalthea","himalia","elara","pasiphae"],
        "saturn": ["mimas","tethys","dione","rhea","iapetus","titan","enceladus"], "uranus": ["miranda","ariel","umbriel","oberon","titania"], "neptune": ["triton","nereid"],
    }
    inc_deg = {"mercury":7,"venus":3.4,"earth":0,"mars":1.85,"jupiter":1.3,"saturn":2.49,"uranus":0.77,"neptune":1.77}
    rng = np.random.default_rng(seed)
    pos_list=[]; vel_list=[]; masses_list=[]; labels_list=[]
    pos_list.append(np.zeros((1,3))); vel_list.append(np.zeros((1,3)))
    masses_list.append(np.array([M_sun])); labels_list.append(np.array([0],dtype=np.int32))
    label_map = {"sun":0,"mercury":1,"venus":2,"earth":3,"mars":4,"jupiter":5,"saturn":6,"uranus":7,"neptune":8}
    for p in planets:
        a = a_ua[p]
        inc = np.deg2rad(inc_deg[p] + rng.normal(0,0.5))
        theta = rng.uniform(0, 2*np.pi)
        x = a*np.cos(theta); y = a*np.sin(theta)*np.cos(inc); z = a*np.sin(theta)*np.sin(inc)*0.3
        pos = np.array([[x,y,z]],dtype=np.float64)
        v_circ = np.sqrt(G * M_sun * a*a / (a*a + eps*eps)**1.5 + 1e-12)
        vx = -v_circ*np.sin(theta); vy = v_circ*np.cos(theta)*np.cos(inc); vz = v_circ*np.cos(theta)*np.sin(inc)*0.3
        vel = np.array([[vx,vy,vz]],dtype=np.float64)
        pos_list.append(pos); vel_list.append(vel)
        masses_list.append(np.array([m_rel[p] * M_sun])); labels_list.append(np.array([label_map[p]],dtype=np.int32))
    planet_idx = {p:i+1 for i,p in enumerate(planets)}
    if with_moons:
        for p, mlist in moons.items():
            p_pos = pos_list[planet_idx[p]][0]; p_vel = vel_list[planet_idx[p]][0]
            p_mass = masses_list[planet_idx[p]][0]
            for m in mlist:
                a = a_moon[m]
                theta = rng.uniform(0, 2*np.pi)
                x = a*np.cos(theta); y = a*np.sin(theta); z = rng.normal(0, a*0.05)
                # usa eps global da simulação para consistência (se eps=0, órbita exata)
                v_circ = np.sqrt(G * p_mass * a*a / (a*a + eps*eps)**1.5 + 1e-12)
                vx = -v_circ*np.sin(theta); vy = v_circ*np.cos(theta); vz = 0
                pos_list.append(np.array([[p_pos[0]+x, p_pos[1]+y, p_pos[2]+z]]))
                vel_list.append(np.array([[p_vel[0]+vx, p_vel[1]+vy, p_vel[2]+vz]]))
                m = p_mass * m_moon_rel[m]
                masses_list.append(np.array([max(m, 1e-9)]))
                labels_list.append(np.array([9],dtype=np.int32))
    # planetas anões (orbitam o Sol)
    dwarfs = {
        "ceres": (0.277, 10.6, 5.2e-10),
        "pluto": (3.95, 17.2, 7.3e-9),
        "haumea": (4.30, 28.2, 2.0e-9),
        "makemake": (4.55, 29.0, 1.5e-9),
        "eris": (6.80, 44.0, 8.4e-9),
    }
    for name, (a, inc, m) in dwarfs.items():
        inc_rad = 0  # simplificado
        theta = rng.uniform(0, 2*np.pi)
        x = a*np.cos(theta); y = a*np.sin(theta); z = rng.normal(0, a*0.05)
        v_circ = (G * M_sun * a*a / (a*a + 0.01*0.01)**1.5 + 1e-12) ** 0.5
        vx = -v_circ*np.sin(theta); vy = v_circ*np.cos(theta); vz = 0
        pos_list.append(np.array([[x,y,z]])); vel_list.append(np.array([[vx,vy,vz]]))
        masses_list.append(np.array([m])); labels_list.append(np.array([10], dtype=np.int32))
    pos = np.vstack(pos_list); vel = np.vstack(vel_list)
    masses = np.concatenate(masses_list); labels = np.concatenate(labels_list)
    com_pos = np.average(pos, axis=0, weights=masses)
    pos -= com_pos
    com_vel = np.average(vel, axis=0, weights=masses)
    vel -= com_vel
    return pos.astype(np.float64), vel.astype(np.float64), masses.astype(np.float64), labels


def generate_solar_system_real(
    G: float = G_DEFAULT,
    eps: float = 0.01,
    with_moons: bool = True,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """
    Sistema Solar real J2000 — elementos orbitais heliocêntricos reais (a,e,i,Ω,ω,M)
    de JPL Horizons 2000-01-01 12:00 TT, mesma época para todos os corpos.
    a em UA/10, massas reescaladas M_total≈1. Inclui 8 planetas + 11 luas + 5 anões.
    Posições/velocidades via solução de Kepler (M → E → r,v no plano orbital → 3D).
    """
    import math
    def kepler_pos_vel(a, e, inc_deg, Omega_deg, omega_deg, M_deg, mu):
        # resolve E - e sin E = M (rad)
        M = math.radians(M_deg); inc = math.radians(inc_deg); Omega = math.radians(Omega_deg); omega = math.radians(omega_deg)
        E = M if e < 0.8 else math.pi
        for _ in range(20):
            dE = (E - e*math.sin(E) - M) / (1 - e*math.cos(E))
            E -= dE
            if abs(dE) < 1e-12:
                break
        # posição no plano orbital
        cosE = math.cos(E); sinE = math.sin(E)
        r = a * (1 - e*cosE)
        # fator sqrt(1-e²)
        sq = math.sqrt(max(0, 1 - e*e))
        # coordenadas no plano (x',y')
        x_orb = a * (cosE - e)
        y_orb = a * sq * sinE
        # velocidade no plano: n = sqrt(mu/a³), v = n*a / r * (-a sinE, a sq cosE)
        n = math.sqrt(mu / (a*a*a + 1e-18))
        vx_orb = -n * a * sinE / (1 - e*cosE)
        vy_orb = n * a * sq * cosE / (1 - e*cosE)
        # rotação 3D: R = Rz(Omega) * Rx(inc) * Rz(omega)
        cosO = math.cos(Omega); sinO = math.sin(Omega)
        cosi = math.cos(inc); sini = math.sin(inc)
        cosw = math.cos(omega); sinw = math.sin(omega)
        # matriz combinada
        x = (cosO*cosw - sinO*sinw*cosi)*x_orb + (-cosO*sinw - sinO*cosw*cosi)*y_orb
        y = (sinO*cosw + cosO*sinw*cosi)*x_orb + (-sinO*sinw + cosO*cosw*cosi)*y_orb
        z = (sinw*sini)*x_orb + (cosw*sini)*y_orb
        vx = (cosO*cosw - sinO*sinw*cosi)*vx_orb + (-cosO*sinw - sinO*cosw*cosi)*vy_orb
        vy = (sinO*cosw + cosO*sinw*cosi)*vx_orb + (-sinO*sinw + cosO*cosw*cosi)*vy_orb
        vz = (sinw*sini)*vx_orb + (cosw*sini)*vy_orb
        return np.array([x,y,z], dtype=np.float64), np.array([vx,vy,vz], dtype=np.float64)

    M_sun = 0.998
    mu_sun = G * M_sun
    # elementos J2000 (a em UA, e, i deg, Omega deg, omega deg, M deg) — NASA JPL
    # a já /10 para escala visual (mesma de generate_solar_system)
    planets_j2000 = {
        "mercury": (0.0387, 0.2056, 7.00, 48.33, 29.12, 174.79, 1.65e-7),
        "venus":   (0.0723, 0.00677,3.39, 76.68, 54.88, 50.11, 2.44e-6),
        "earth":   (0.1000, 0.01671,0.00, -11.26,114.21,357.52,3.00e-6),
        "mars":    (0.1523, 0.09339,1.85, 49.57,286.46, 19.41,3.21e-7),
        "jupiter": (0.5202, 0.04839,1.30,100.45,273.86, 20.02,9.54e-4),
        "saturn":  (0.9580, 0.0555, 2.48,113.66,339.39,317.02,2.85e-4),
        "uranus":  (1.918, 0.0457, 0.77, 74.00, 96.99,142.2, 4.37e-5),
        "neptune": (3.007, 0.0113, 1.77,131.78,273.18,256.2, 5.15e-5),
    }
    label_map = {"sun":0,"mercury":1,"venus":2,"earth":3,"mars":4,"jupiter":5,"saturn":6,"uranus":7,"neptune":8}
    pos_list=[np.zeros(3)]; vel_list=[np.zeros(3)]; masses_list=[M_sun]; labels_list=[0]
    # planetas heliocêntricos
    for name, (a,e,inc,Om,om,M,m_rel) in planets_j2000.items():
        pos, vel = kepler_pos_vel(a, e, inc, Om, om, M, mu_sun)
        pos_list.append(pos); vel_list.append(vel)
        masses_list.append(m_rel * M_sun)
        labels_list.append(label_map[name])
    # luas: elementos planetocêntricos J2000 simplificados (a em UA/10, e pequeno, i~0)
    # usa mu do planeta
    moons_j2000 = {
        "moon":      ("earth", 0.000257, 0.0549, 5.1, 0, 0, 135.0, 0.0123),
        "phobos":    ("mars", 0.000062, 0.0151, 1.08, 0, 0, 90.0, 1.6e-8),
        "deimos":    ("mars", 0.000157, 0.0002, 1.79, 0, 0, 45.0, 2.3e-9),
        "io":        ("jupiter",0.000281,0.0041,0.04,0,0,20.0,4.70e-5),
        "europa":    ("jupiter",0.000449,0.0094,0.47,0,0,80.0,2.52e-5),
        "ganymede":  ("jupiter",0.000715,0.0013,0.17,0,0,140.0,7.81e-5),
        "callisto":  ("jupiter",0.00126,0.0074,0.19,0,0,200.0,5.66e-5),
        "titan":     ("jupiter",0.00035,0.0288,0.33,0,0,60.0,2.37e-4),  # Titan é Saturno, corrigido abaixo
    }
    # correção: Titan é Saturno, não Júpiter — vamos definir corretamente
    moons_j2000 = {
        "moon":      ("earth", 0.000257, 0.0549, 5.1, 0, 0, 135.0, 0.0123),
        "phobos":    ("mars", 0.000062, 0.0151, 1.08, 0, 0, 90.0, 1.6e-8),
        "deimos":    ("mars", 0.000157, 0.0002, 1.79, 0, 0, 45.0, 2.3e-9),
        "io":        ("jupiter",0.000281,0.0041,0.04,0,0,20.0,4.70e-5),
        "europa":    ("jupiter",0.000449,0.0094,0.47,0,0,80.0,2.52e-5),
        "ganymede":  ("jupiter",0.000715,0.0013,0.17,0,0,140.0,7.81e-5),
        "callisto":  ("jupiter",0.00126,0.0074,0.19,0,0,200.0,5.66e-5),
        "titan":     ("saturn",0.000817,0.0288,0.33,0,0,60.0,2.37e-4),
        "enceladus": ("saturn",0.000159,0.0047,0.02,0,0,10.0,1.9e-7),
        "triton":    ("neptune",0.000237,0.0000,156.9,0,0,300.0,2.09e-4),
    }
    # mapeia nome do planeta para índice em pos_list (1=mercury...)
    planet_idx = {name:i+1 for i,name in enumerate(planets_j2000.keys())}
    planet_masses = {name: m_rel*M_sun for name,(a,e,inc,Om,om,M,m_rel) in planets_j2000.items()}
    if with_moons:
        for mname, (pname, a, e, inc, Om, om, M, m_rel_moon) in moons_j2000.items():
            mu_p = G * planet_masses[pname]
            pos_rel, vel_rel = kepler_pos_vel(a, e, inc, Om, om, M, mu_p)
            p_pos = pos_list[planet_idx[pname]]
            p_vel = vel_list[planet_idx[pname]]
            pos_list.append(p_pos + pos_rel)
            vel_list.append(p_vel + vel_rel)
            masses_list.append(planet_masses[pname] * m_rel_moon)
            labels_list.append(9)
    # anões heliocêntricos J2000
    dwarfs_j2000 = {
        "ceres":   (0.277,0.0758,10.59,80.30,73.62,77.37,4.7e-10),
        "pluto":   (3.95,0.244,17.16,110.30,113.77,14.86,6.6e-9),
        "haumea":  (4.30,0.191,28.19,121.9,240.08,156.0,2.0e-9),
        "makemake":(4.55,0.159,29.0,79.38,297.24,166.0,1.5e-9),
        "eris":    (6.80,0.440,44.04,35.95,151.64,205.0,8.4e-9),
    }
    for name,(a,e,inc,Om,om,M,m) in dwarfs_j2000.items():
        pos, vel = kepler_pos_vel(a, e, inc, Om, om, M, mu_sun)
        pos_list.append(pos); vel_list.append(vel)
        masses_list.append(m); labels_list.append(10)
    pos = np.vstack(pos_list); vel = np.vstack(vel_list)
    masses = np.array(masses_list); labels = np.array(labels_list, dtype=np.int32)
    # já está em COM heliocêntrico (Sol no 0), mas recentra para massa total
    com_pos = np.average(pos, axis=0, weights=masses)
    pos -= com_pos
    com_vel = np.average(vel, axis=0, weights=masses)
    vel -= com_vel
    return pos.astype(np.float64), vel.astype(np.float64), masses.astype(np.float64), labels


def generate_solar_completo(
    G: float = G_DEFAULT,
    eps: float = 1e-4,
    with_moons: bool = True,
    with_dwarfs: bool = True,
    with_asteroids: bool = False,
    cache_path: str | None = None,
    n_asteroids: int = 0,
    asteroids_cache: str | None = None,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray]:
    """
    Solar completo — usa VECTORS do horizons_cache.json (2026-01-01, mesma época).
    Posições heliocêntricas + planetocêntricas para luas, garantindo coerência:
    luas são heliocêntricas = planeta heliocêntrico + vetor planetocêntrico.
    Garante estabilidade: eps efetivo <=1e-05 para não amolecer órbitas lunares,
    e conversão de velocidade correta para G (S=0.1, T=sqrt(G*1000/4pi²)).
    Fallback: se cache ausente/corrompido, usa solar_real (estável).
    Sempre inclui TODAS as luas genuínas do cache (limite de labels é
    responsabilidade do visualizador, ex: --moons).
    n_asteroids: anexa os N primeiros asteroides reais de asteroids_cache
      (NPZ do fetch_asteroids.py, label 11, GM exato ou 0.0 = partícula-teste).
      Atenção: N grande exige backend BH (N² direto é inviável acima de ~20k).

    Retorna: pos, vel, masses, labels, names
      labels: 0 Sol,1-8 planetas,9 luas,10 anões,11 asteroides
      names: lista na mesma ordem (ex: 'Sun','Jupiter','Mimas','S/2019 S1',...),
             extraída de 'Target body name' do próprio cache (nomes reais JPL).
    Importante: simulação deve usar G igual e eps <=1e-05 para luas permanecerem ligadas.
    """
    import pathlib, json, re, math
    S = 0.1  # AU -> AU/10
    G_real = 4 * math.pi ** 2
    # T inclui M_sol simulado (0.998, não 1.0): T^2 = S^3*G_real/(G*M_sol).
    # Sem isso as velocidades ficam 0.1% altas e a Lua deriva ~3° em 270 dias.
    T_time = math.sqrt(max(G, 1e-12) * 0.998 * 1000.0 / G_real) if G_real > 0 else 1.0
    vel_scale = S * T_time
    # eps efetivo pequeno para não suavizar gravidade lua-planeta (a_lua ~0.0003, eps=0.01 >> a)
    eps_eff = eps
    if with_moons and eps >= 1e-4:
        eps_eff = 1e-5
    # para compatibilidade, ignora eps passado grande

    cache_path = pathlib.Path(cache_path) if cache_path else pathlib.Path("data/horizons_cache.json")
    if not cache_path.exists():
        # tenta caminhos alternativos (execução a partir de subdir)
        for alt in [
            pathlib.Path(__file__).resolve().parents[1] / "data" / "horizons_cache.json",
            pathlib.Path("data_final/horizons_cache.json"),
            pathlib.Path(__file__).resolve().parents[1] / "data_final" / "horizons_cache.json",
        ]:
            if alt.exists():
                cache_path = alt
                break

    def _parse_vectors(raw: str):
        """Extrai X,Y,Z,VX,VY,VZ do bloco $$SOE e converte para unidades da simulação (S e vel_scale)."""
        try:
            m = re.search(r'\$\$SOE(.*)\$\$EOE', raw, re.S)
            if not m:
                return None
            txt = m.group(1).strip()
            for line in txt.splitlines():
                line = line.strip()
                if not line or line.startswith("$$"):
                    continue
                nums = re.findall(r'[-+]?\d+\.\d+E[+-]\d+', line)
                if len(nums) >= 6:
                    try:
                        x = float(nums[0]); y = float(nums[1]); z = float(nums[2])
                        vx = float(nums[3]); vy = float(nums[4]); vz = float(nums[5])
                        # zero indica falha (ex: Saturno 500@699)
                        if abs(x) < 1e-12 and abs(y) < 1e-12 and abs(z) < 1e-12 and abs(vx) < 1e-12 and abs(vy) < 1e-12 and abs(vz) < 1e-12:
                            return None
                        vx *= 365.25; vy *= 365.25; vz *= 365.25  # AU/D -> AU/yr
                        return {
                            "x": x * S, "y": y * S, "z": z * S,
                            "vx": vx * vel_scale, "vy": vy * vel_scale, "vz": vz * vel_scale,
                        }
                    except Exception:
                        continue
        except Exception:
            pass
        return None

    def _body_name(raw: str, fallback: str) -> str:
        """Extrai nome limpo de 'Target body name: X (ID)' do raw Horizons.
        Ex: 'Mimas (601)'->'Mimas', '136199 Eris (2003 UB313)'->'Eris',
        'S2019_S1 (65093)'->'S/2019 S1'. Sem raw válido, usa fallback."""
        try:
            m = re.search(r'Target body name:\s*(.*?)\s*\{', raw or "")
            if m:
                base = m.group(1).strip().split('(')[0].strip()
                base = re.sub(r'^\d+\s+', '', base)  # '136199 Eris' -> 'Eris'
                pm = re.match(r'^S(\d{4})_([A-Z]+)(\d+)$', base)
                if pm:
                    return f"S/{pm.group(1)} {pm.group(2)}{pm.group(3)}"
                if base:
                    return base
        except Exception:
            pass
        return fallback

    _planet_names = {199: "Mercury", 299: "Venus", 399: "Earth", 499: "Mars",
                     599: "Jupiter", 699: "Saturn", 799: "Uranus", 899: "Neptune"}
    _dwarf_names = {2000001: "Ceres", 134340: "Pluto", 136199: "Eris",
                    136108: "Haumea", 136472: "Makemake", 1: "Ceres"}

    GM_SUN = 1.32712440018e11  # km^3/s^2 (DE441) — converte GM da API p/ massa sim

    def _parse_gm(raw: str):
        """Extrai GM (km^3/s^2) do bloco OBJ_DATA. None se 'n.a.' / ausente.
        Evita linhas 'GM 1-sigma' (só casa 'GM (km^3/s^2) =' direto)."""
        try:
            m = re.search(r'GM\s*\(km\^3/s\^2\)\s*=\s*([0-9.\-E+]+)', raw or "")
            if m:
                return float(m.group(1))
            m = re.search(r'\bGM\s*=\s*([0-9.\-E+]+)', raw or "")
            if m:
                return float(m.group(1))
        except Exception:
            pass
        return None

    def _gm_mass(gm, fallback):
        """Massa sim a partir do GM da API; fallback tabelado se None."""
        try:
            if gm is not None and gm > 0:
                return float(gm) / GM_SUN * M_sun
        except Exception:
            pass
        return fallback

    def _load_asteroids(n):
        """Lê NPZ do fetch_asteroids.py -> listas (pos, vel, masses, labels, names).
        Usa S/vel_scale do closure (mesma época/escala do resto)."""
        import numpy as _np
        if not n or n <= 0:
            return [], [], [], [], []
        ap = pathlib.Path(asteroids_cache) if asteroids_cache else pathlib.Path("data/asteroids.npz")
        if not ap.exists():
            print(f"[aviso] n_asteroids={n} mas {ap} não existe; rode scripts/fetch_asteroids.py")
            return [], [], [], [], []
        z = _np.load(ap, allow_pickle=True)
        try:
            ep = str(z["epoch"])
        except Exception:
            ep = "?"
        try:
            cep = json.loads(cache_path.read_text(encoding="utf-8")).get("epoch", "?") if cache_path.exists() else "?"
        except Exception:
            cep = "?"
        if ep != cep:
            print(f"[aviso] época asteroides ({ep}) != época planetas ({cep}); órbitas inconsistentes!")
        if "desig" in z:
            # NPZ do fetch_mpcorb.py: nomes legíveis, massa 0.0 (MPCORB não traz GM)
            keys = [str(d) for d in z["desig"]][:n]
            pa = z["pos_au"][:len(keys)]
            va = z["vel_auyr"][:len(keys)]
            ga = [float("nan")] * len(keys)
            tag = lambda d: str(d)  # noqa: E731
        else:
            # NPZ do fetch_asteroids.py: IDs numerados + GM quando houver
            keys = [int(i) for i in z["ids"]][:n]
            pa = z["pos_au"][:len(keys)]
            va = z["vel_auyr"][:len(keys)]
            try:
                ga = [float(g) for g in z["gm"][:len(keys)]]
            except Exception:
                ga = [float("nan")] * len(keys)
            tag = lambda d: f"Ast-{d}"  # noqa: E731
        P, V, Mm, L, Nm = [], [], [], [], []
        for key, p_au, v_au, gm in zip(keys, pa, va, ga):
            P.append(_np.array(p_au, dtype=_np.float64) * S)
            V.append(_np.array(v_au, dtype=_np.float64) * vel_scale)
            try:
                m = float(gm) / GM_SUN * M_sun if gm == gm and gm > 0 else 0.0
            except Exception:
                m = 0.0
            Mm.append(m)  # GM real quando há; senão 0.0 (teste: sente, não puxa)
            L.append(11)
            Nm.append(tag(key))
        if len(P) < n:
            print(f"[aviso] pediu {n} asteroides, NPZ só tem {len(P)}")
        print(f"Asteroides: {len(P)} de {ap} (época {ep})")
        return P, V, Mm, L, Nm

    def _kepler(a, e, inc_deg, Om_deg, w_deg, M_deg, mu):
        """Kepler 2-corpos: a->pos/vel 3D, mu=G*M_central."""
        M = math.radians(M_deg); inc = math.radians(inc_deg); Om = math.radians(Om_deg); w = math.radians(w_deg)
        E = M if e < 0.8 else math.pi
        for _ in range(30):
            denom = 1 - e * math.cos(E)
            if abs(denom) < 1e-14:
                denom = 1e-14
            dE = (E - e * math.sin(E) - M) / denom
            E -= dE
            if abs(dE) < 1e-12:
                break
        cosE = math.cos(E); sinE = math.sin(E); sq = math.sqrt(max(0, 1 - e * e))
        denom = 1 - e * cosE
        if abs(denom) < 1e-12:
            denom = 1e-12
        x_orb = a * (cosE - e); y_orb = a * sq * sinE
        n = math.sqrt(mu / (a * a * a + 1e-18))
        vx_orb = -n * a * sinE / denom; vy_orb = n * a * sq * cosE / denom
        cosO = math.cos(Om); sinO = math.sin(Om); cosi = math.cos(inc); sini = math.sin(inc); cosw = math.cos(w); sinw = math.sin(w)
        x = (cosO * cosw - sinO * sinw * cosi) * x_orb + (-cosO * sinw - sinO * cosw * cosi) * y_orb
        y = (sinO * cosw + cosO * sinw * cosi) * x_orb + (-sinO * sinw + cosO * cosw * cosi) * y_orb
        z = (sinw * sini) * x_orb + (cosw * sini) * y_orb
        vx = (cosO * cosw - sinO * sinw * cosi) * vx_orb + (-cosO * sinw - sinO * cosw * cosi) * vy_orb
        vy = (sinO * cosw + cosO * sinw * cosi) * vx_orb + (-sinO * sinw + cosO * cosw * cosi) * vy_orb
        vz = (sinw * sini) * vx_orb + (cosw * sini) * vy_orb
        return np.array([x, y, z], dtype=np.float64), np.array([vx, vy, vz], dtype=np.float64)

    M_sun = 0.998
    mu_sun = G * M_sun
    # elementos heliocêntricos fallback (J2000 aproximado, mesma escala S)
    planets_j2000 = {
        199: (0.0387, 0.2056, 7.00, 48.33, 29.12, 174.79, 1.65e-7),
        299: (0.0723, 0.00677, 3.39, 76.68, 54.88, 50.11, 2.44e-6),
        399: (0.1000, 0.01671, 0.00, -11.26, 114.21, 357.52, 3.00e-6),
        499: (0.1523, 0.09339, 1.85, 49.57, 286.46, 19.41, 3.21e-7),
        599: (0.5202, 0.04839, 1.30, 100.45, 273.86, 20.02, 9.54e-4),
        699: (0.9580, 0.0555, 2.48, 113.66, 339.39, 317.02, 2.85e-4),
        799: (1.918, 0.0457, 0.77, 74.00, 96.99, 142.2, 4.37e-5),
        899: (3.007, 0.0113, 1.77, 131.78, 273.18, 256.2, 5.15e-5),
    }
    dwarfs_j2000 = {
        2000001: (0.277, 0.0758, 10.59, 80.30, 73.62, 77.37, 4.7e-10),
        134340: (3.95, 0.244, 17.16, 110.30, 113.77, 14.86, 6.6e-9),
        136199: (6.80, 0.440, 44.04, 35.95, 151.64, 205.0, 8.4e-9),
        136108: (4.30, 0.191, 28.19, 121.9, 240.08, 156.0, 2.0e-9),
        136472: (4.55, 0.159, 29.0, 79.38, 297.24, 166.0, 1.5e-9),
    }
    planet_ids = [199, 299, 399, 499, 599, 699, 799, 899]
    dwarf_ids = [2000001, 134340, 136199, 136108, 136472]
    mass_map = {
        199: 1.65e-7 * M_sun, 299: 2.44e-6 * M_sun, 399: 3.00348e-6 * M_sun, 499: 3.21e-7 * M_sun,
        599: 9.54e-4 * M_sun, 699: 2.85e-4 * M_sun, 799: 4.37e-5 * M_sun, 899: 5.15e-5 * M_sun,
        2000001: 4.7e-10, 134340: 6.6e-9, 136199: 8.4e-9, 136108: 2.0e-9, 136472: 1.5e-9, 1: 4.7e-10,
    }
    label_map = {199: 1, 299: 2, 399: 3, 499: 4, 599: 5, 699: 6, 799: 7, 899: 8, 2000001: 10, 134340: 10, 136199: 10, 136108: 10, 136472: 10, 1: 10}

    use_cache = False
    cache_bodies = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            bodies = cache.get("bodies", {})
            cnt = sum(1 for v in bodies.values() if isinstance(v, dict) and len(v.get("raw", "")) > 2000 and "X," in v.get("raw", ""))
            if cnt >= 10:
                use_cache = True
                cache_bodies = bodies
        except Exception:
            use_cache = False

    if use_cache:
        pos_list = []; vel_list = []; masses_list = []; labels_list = []; names_list = []
        pos_list.append(np.zeros(3)); vel_list.append(np.zeros(3)); masses_list.append(M_sun); labels_list.append(0); names_list.append("Sun")
        planet_heliocentric = {}

        # planetas heliocêntricos (S=0.1, vel_scale)
        for sid in planet_ids:
            raw = cache_bodies.get(str(sid), {}).get("raw", "")
            pars = _parse_vectors(raw) if raw else None
            if pars is None:
                # fallback Kepler se cache corrompido (ex: Saturno 699)
                if sid in planets_j2000:
                    a, e, inc, Om, om, M, _m_rel = planets_j2000[sid]
                    pos, vel = _kepler(a, e, inc, Om, om, M, mu_sun)
                    pars = {"x": pos[0], "y": pos[1], "z": pos[2], "vx": vel[0], "vy": vel[1], "vz": vel[2]}
                else:
                    continue
            else:
                # Saturno com center 500@699 retorna zero -> já tratado acima como None
                # Uranus 799 com center 500@699 mas vetor grande é heliocêntrico legit, mantém
                # Verifica se é heliocêntrico plausível: |pos|>0.3 sim é heliocêntrico
                pass
            pos = np.array([pars["x"], pars["y"], pars["z"]], dtype=np.float64)
            vel = np.array([pars["vx"], pars["vy"], pars["vz"]], dtype=np.float64)
            m = _gm_mass(cache_bodies.get(str(sid), {}).get("gm", _parse_gm(raw)), mass_map.get(sid, 1e-10))
            pos_list.append(pos); vel_list.append(vel); masses_list.append(m); labels_list.append(label_map.get(sid, 1))
            names_list.append(_body_name(raw, _planet_names.get(sid, f"Body-{sid}")))
            planet_heliocentric[sid] = (pos, vel)

        # garante Saturn fallback se ainda ausente (caso 699 zero)
        if 699 not in planet_heliocentric and 699 in planets_j2000:
            a, e, inc, Om, om, M, _m_rel = planets_j2000[699]
            pos, vel = _kepler(a, e, inc, Om, om, M, mu_sun)
            m = mass_map[699]
            pos_list.append(pos); vel_list.append(vel); masses_list.append(m); labels_list.append(label_map[699])
            names_list.append(_body_name(cache_bodies.get("699", {}).get("raw", ""), "Saturn"))
            planet_heliocentric[699] = (pos, vel)
            # mantém ordem? não precisa

        # JWST (-170, espaçonave em halo L2): tratado como lua (label 9) da Terra.
        # Índice baixo de propósito (logo após os planetas): sempre nomeado.
        # Massa 0.0 (6200 kg): sente gravidade, não puxa ninguém. Halo é instável
        # sem station-keeping — derivar com o tempo é física real, não bug.
        if with_moons and "-170" in cache_bodies and 399 in planet_heliocentric:
            _raw_jw = cache_bodies["-170"].get("raw", "")
            _pars_jw = _parse_vectors(_raw_jw) if _raw_jw else None
            if _pars_jw is not None:
                _pr = np.array([_pars_jw["x"], _pars_jw["y"], _pars_jw["z"]], dtype=np.float64)
                _vr = np.array([_pars_jw["vx"], _pars_jw["vy"], _pars_jw["vz"]], dtype=np.float64)
                if float(np.linalg.norm(_pr)) < 0.2:  # mesmo filtro anti-asteroide das luas
                    _pp, _pv = planet_heliocentric[399]
                    pos_list.append(_pp + _pr); vel_list.append(_pv + _vr)
                    masses_list.append(0.0); labels_list.append(9)
                    _nm_jw = _body_name(_raw_jw, "James Webb")
                    names_list.append("James Webb" if "Webb" in _nm_jw else _nm_jw)

        # luas: whitelist de luas genuínas (evita asteroides 573+,667+ etc.)
        valid_moons = []
        if with_moons:
            valid_moons += [301]
            valid_moons += [401, 402]
            valid_moons += list(range(501, 573))   # Júpiter 72 luas (573+ são asteroides)
            valid_moons += list(range(601, 667))   # Saturno 66 luas (667+ são asteroides) — genuínas sequenciais
            valid_moons += list(range(65000, 65250))  # Saturno novas 2019-2023 (S/2019 S1 65093 etc.) — filtradas por dist<0.5
            valid_moons += list(range(701, 728))   # Urano 27 luas
            valid_moons += list(range(801, 815))   # Netuno 14 luas
            valid_moons += list(range(901, 906))   # Plutão 5 luas

        moon_mass_known = {
            301: M_sun * 3.00348e-6 * 0.01215,
            401: 5e-15, 402: 2e-15,
            501: M_sun * 9.54e-4 * 4.70e-5, 502: M_sun * 9.54e-4 * 2.52e-5,
            503: M_sun * 9.54e-4 * 7.81e-5, 504: M_sun * 9.54e-4 * 5.66e-5,
            606: M_sun * 2.85e-4 * 2.37e-4,  # Titan
            801: M_sun * 5.15e-5 * 2.09e-4,  # Triton
        }
        center_to_parent = {"500@399": 399, "500@499": 499, "500@599": 599, "500@699": 699, "500@799": 799, "500@899": 899, "500@999": 134340}

        # para luas de Plutão, precisa heliocêntrico de Plutão (dwarf)
        # se ainda não adicionamos dwarfs, adiciona Pluto heliocêntrico agora para poder somar luas
        # tenta pegar Pluto do cache ou kepler antes das luas de Pluto
        pluto_heli = None
        if with_moons and any(901 <= sid <= 905 for sid in valid_moons):
            # tenta Pluto do cache
            for pid in [134340, 999]:
                if pid in planet_heliocentric:
                    pluto_heli = planet_heliocentric[pid]
                    break
            if pluto_heli is None:
                # busca dwarf cache
                raw_pluto = cache_bodies.get("134340", {}).get("raw", "")
                pars_pluto = _parse_vectors(raw_pluto) if raw_pluto else None
                if pars_pluto is None and 134340 in dwarfs_j2000:
                    a, e, inc, Om, om, M, _m = dwarfs_j2000[134340]
                    pos_p, vel_p = _kepler(a, e, inc, Om, om, M, mu_sun)
                    pars_pluto = {"x": pos_p[0], "y": pos_p[1], "z": pos_p[2], "vx": vel_p[0], "vy": vel_p[1], "vz": vel_p[2]}
                if pars_pluto is not None:
                    pos_p = np.array([pars_pluto["x"], pars_pluto["y"], pars_pluto["z"]])
                    vel_p = np.array([pars_pluto["vx"], pars_pluto["vy"], pars_pluto["vz"]])
                    planet_heliocentric[134340] = (pos_p, vel_p)
                    planet_heliocentric[999] = (pos_p, vel_p)

        seen_moon_names = set()  # mesma lua pode vir em 2 IDs (ex: 642 Fornjot e alias 65036)
        for sid in valid_moons:
            sid_str = str(sid)
            if sid_str not in cache_bodies:
                continue
            raw = cache_bodies[sid_str].get("raw", "")
            if not raw or "No ephemeris" in raw or len(raw) < 500:
                continue
            _nm = _body_name(raw, f"Moon-{sid}")
            if _nm in seen_moon_names:
                continue  # alias duplicado, mantém a primeira ocorrência (ID sequencial)
            seen_moon_names.add(_nm)
            center = cache_bodies[sid_str].get("center", "")
            parent_id = None
            for k, v in center_to_parent.items():
                if k in center:
                    parent_id = v
                    break
            if parent_id is None:
                if 501 <= sid <= 572:
                    parent_id = 599
                elif 601 <= sid <= 666:
                    parent_id = 699
                elif 701 <= sid <= 727:
                    parent_id = 799
                elif 801 <= sid <= 814:
                    parent_id = 899
                elif sid == 301:
                    parent_id = 399
                elif sid in (401, 402):
                    parent_id = 499
                elif 901 <= sid <= 905:
                    parent_id = 134340
            if parent_id is None or parent_id not in planet_heliocentric:
                continue
            pars = _parse_vectors(raw)
            if pars is None:
                continue
            pos_rel = np.array([pars["x"], pars["y"], pars["z"]], dtype=np.float64)
            vel_rel = np.array([pars["vx"], pars["vy"], pars["vz"]], dtype=np.float64)
            dist = float(np.linalg.norm(pos_rel))
            # filtra asteroides: VECTORS com CENTER Saturn mas |pos|>0.2 sim (~2 AU real) é asteroide heliocêntrico, não lua (Hill Saturno 0.045)
            if dist > 0.2:
                continue
            p_pos, p_vel = planet_heliocentric[parent_id]
            pos_heli = p_pos + pos_rel
            vel_heli = p_vel + vel_rel
            # massa da lua: GM da API quando existe, senão tabela/estimativa
            m = _gm_mass(cache_bodies.get(sid_str, {}).get("gm", _parse_gm(raw)), moon_mass_known.get(sid, None))
            if m is None:
                # massas pequenas: regulares ~1e-09, irregulares ~1e-12
                if sid in (601, 602, 603, 604, 605, 606, 607, 608, 610, 611, 615, 616, 617, 618, 703, 704, 705, 803, 804, 805, 806, 807, 808):
                    m = 1e-08
                else:
                    m = 1e-11
                m = max(m, 1e-12)
            pos_list.append(pos_heli); vel_list.append(vel_heli); masses_list.append(float(m)); labels_list.append(9)
            names_list.append(_body_name(raw, f"Moon-{sid}"))

        # anões heliocêntricos
        if with_dwarfs:
            for sid in dwarf_ids:
                # cache usa "1" para Ceres, mas 1 está corrompido (Mercury), usa fallback direto
                raw_key = "1" if sid == 2000001 else str(sid)
                raw = cache_bodies.get(raw_key, {}).get("raw", "") if raw_key in cache_bodies else ""
                # para Ceres, verifica se raw é realmente Ceres
                if sid == 2000001 and raw and "Ceres" not in raw:
                    # é Mercury duplicado, ignora cache
                    raw = ""
                pars = _parse_vectors(raw) if raw else None
                if pars is None:
                    if sid in dwarfs_j2000:
                        a, e, inc, Om, om, M, _mass = dwarfs_j2000[sid]
                        pos, vel = _kepler(a, e, inc, Om, om, M, mu_sun)
                        pars = {"x": pos[0], "y": pos[1], "z": pos[2], "vx": vel[0], "vy": vel[1], "vz": vel[2]}
                    else:
                        continue
                pos = np.array([pars["x"], pars["y"], pars["z"]], dtype=np.float64)
                vel = np.array([pars["vx"], pars["vy"], pars["vz"]], dtype=np.float64)
                # evita duplicar se já adicionamos Pluto como planeta para luas
                # se já existe massa igual e posição próxima, pula
                m = _gm_mass(cache_bodies.get(raw_key, {}).get("gm", _parse_gm(raw)), mass_map.get(sid, 1e-10))
                # verifica duplicata por massa+label
                duplicate = False
                for idx, (pm, lab) in enumerate(zip(masses_list, labels_list)):
                    if lab == 10 and abs(pm - m) < 1e-12 and np.linalg.norm(pos_list[idx] - pos) < 1e-6:
                        duplicate = True
                        break
                if duplicate:
                    continue
                pos_list.append(pos); vel_list.append(vel); masses_list.append(float(m)); labels_list.append(10)
                names_list.append(_body_name(raw, _dwarf_names.get(sid, f"Dwarf-{sid}")))

        # asteroides reais do NPZ (label 11) — antes das sintéticas
        if n_asteroids:
            _ap, _av, _am, _al, _an = _load_asteroids(n_asteroids)
            pos_list += _ap; vel_list += _av; masses_list += _am; labels_list += _al; names_list += _an

        # completa até 430 com sintéticas estáveis APENAS se with_asteroids=True
        # Se só quiser dados reais (padrão), retorna só genuínas (Horizons tem 171 satélites; Saturno 274 inclui 128 sem efeméride ainda)
        if with_moons and with_asteroids and len(pos_list) < 430:
            # conta luas genuínas já adicionadas por planeta (via distância <0.3)
            # para simplificar, usa contagem esperada vs real
            desired_counts = {5: 95, 6: 274, 7: 28, 8: 16}  # por label planeta
            # conta atuais: luas com dist < Hill do planeta
            label_to_idx_tmp = {}
            for idx, lab in enumerate(labels_list):
                if lab in (5, 6, 7, 8) and lab not in label_to_idx_tmp:
                    label_to_idx_tmp[lab] = idx
            # planet data
            planet_masses_tmp = {5: 9.54e-4 * M_sun, 6: 2.85e-4 * M_sun, 7: 4.37e-5 * M_sun, 8: 5.15e-5 * M_sun}
            planet_a_tmp = {5: 0.52, 6: 0.95, 7: 1.92, 8: 3.01}
            # conta luas genuínas por planeta (aprox. via pos_list)
            genuine_counts = {5: 0, 6: 0, 7: 0, 8: 0}
            for i, lab in enumerate(labels_list):
                if lab == 9:
                    # acha planeta mais próximo
                    d5 = np.linalg.norm(pos_list[i] - pos_list[label_to_idx_tmp[5]]) if 5 in label_to_idx_tmp else 1e9
                    d6 = np.linalg.norm(pos_list[i] - pos_list[label_to_idx_tmp[6]]) if 6 in label_to_idx_tmp else 1e9
                    d7 = np.linalg.norm(pos_list[i] - pos_list[label_to_idx_tmp[7]]) if 7 in label_to_idx_tmp else 1e9
                    d8 = np.linalg.norm(pos_list[i] - pos_list[label_to_idx_tmp[8]]) if 8 in label_to_idx_tmp else 1e9
                    dmin = min(d5, d6, d7, d8)
                    if dmin == d5:
                        genuine_counts[5] += 1
                    elif dmin == d6:
                        genuine_counts[6] += 1
                    elif dmin == d7:
                        genuine_counts[7] += 1
                    elif dmin == d8:
                        genuine_counts[8] += 1
            # gera faltantes estáveis dentro de 0.45*Hill, e<0.05
            rng_fill = np.random.default_rng(42)
            _synth_parent = {5: "Jupiter", 6: "Saturn", 7: "Uranus", 8: "Neptune"}
            for p_lab, desired in desired_counts.items():
                if p_lab not in label_to_idx_tmp:
                    continue
                need = desired - genuine_counts.get(p_lab, 0)
                if need <= 0:
                    continue
                p_idx = label_to_idx_tmp[p_lab]
                p_pos = pos_list[p_idx]; p_vel = vel_list[p_idx]; p_mass = planet_masses_tmp[p_lab]
                hill = planet_a_tmp[p_lab] * (p_mass / (3 * M_sun)) ** (1 / 3)
                for k in range(need):
                    a = rng_fill.uniform(0.15 * hill, 0.45 * hill)
                    e = rng_fill.uniform(0, 0.04)
                    inc = rng_fill.uniform(0, 2.0) * math.pi / 180
                    M_ang = rng_fill.uniform(0, 360)
                    E = math.radians(M_ang) if e < 0.8 else math.pi
                    for __ in range(15):
                        denom = 1 - e * math.cos(E)
                        if abs(denom) < 1e-12:
                            denom = 1e-12
                        dE = (E - e * math.sin(E) - math.radians(M_ang)) / denom
                        E -= dE
                        if abs(dE) < 1e-12:
                            break
                    cosE = math.cos(E); sinE = math.sin(E); sq = math.sqrt(max(0, 1 - e * e))
                    denom = 1 - e * cosE
                    if abs(denom) < 1e-12:
                        denom = 1e-12
                    x_orb = a * (cosE - e); y_orb = a * sq * sinE
                    n = math.sqrt(G * p_mass / (a * a * a + 1e-18))
                    vx_orb = -n * a * sinE / denom; vy_orb = n * a * sq * cosE / denom
                    x = x_orb; y = y_orb * math.cos(inc); z = y_orb * math.sin(inc)
                    vx = vx_orb; vy = vy_orb * math.cos(inc); vz = vy_orb * math.sin(inc)
                    pos_list.append(p_pos + np.array([x, y, z]))
                    vel_list.append(p_vel + np.array([vx, vy, vz]))
                    masses_list.append(1e-11)
                    labels_list.append(9)
                    names_list.append(f"Synth-{_synth_parent.get(p_lab, p_lab)}-{k + 1:03d}")
            # se ainda <430 e with_asteroids, completa com asteroides sintéticos estáveis (opcional)
        if len(pos_list) >= 9:  # Sol + 8 planetas
            pos = np.vstack(pos_list); vel = np.vstack(vel_list)
            masses = np.array(masses_list, dtype=np.float64); labels = np.array(labels_list, dtype=np.int32)
            names = np.array(names_list, dtype=object)
            # recentra COM (posição e velocidade)
            com_pos = np.average(pos, axis=0, weights=masses)
            pos -= com_pos
            com_vel = np.average(vel, axis=0, weights=masses)
            vel -= com_vel
            return pos.astype(np.float64), vel.astype(np.float64), masses.astype(np.float64), labels, names

    # Fallback: solar_real (estável) — sem luas sintéticas instáveis
    # Se cache ausente, usa elementos J2000 com eps pequeno para garantir estabilidade
    pos, vel, masses, labels = generate_solar_system_real(G=G, eps=eps_eff, with_moons=with_moons)
    # nomes na ordem fixa do solar_real: Sol, 8 planetas, 10 luas, 5 anões
    _fb_names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
                 "Moon", "Phobos", "Deimos", "Io", "Europa", "Ganymede", "Callisto", "Titan", "Enceladus", "Triton",
                 "Ceres", "Pluto", "Haumea", "Makemake", "Eris"]
    if with_moons:
        names = list(_fb_names[:len(masses)])
    else:
        names = (["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
                 + ["Ceres", "Pluto", "Haumea", "Makemake", "Eris"])[:len(masses)]
    # Não adiciona 400 luas sintéticas com a> Hill (causavam espiral ao Sol).
    # Se usuário realmente quiser 430 corpos sintéticos estáveis, gera extras dentro de Hill.
    if with_moons and len(masses) < 50 and with_asteroids is False:
        # fallback já tem ~24 corpos (Sol+8p+10 luas+5 anões). Para atingir ~180 luas genuínas sem cache,
        # gera luas sintéticas ESTÁVEIS (dentro de Hill, e baixa) — mas não 400 com órbitas além de Hill.
        # Mantém apenas as 10 luas base para estabilidade, evita gerar população instável.
        pass
    # Se pediu 430 corpos sintéticos estáveis, gera até 180 luas mas todas dentro de 0.5 Hill
    if with_moons and len(masses) < 100 and with_asteroids:
        import math as _math
        rng = __import__("numpy").random.default_rng(42)
        label_to_idx = {}
        for idx, lab in enumerate(labels):
            if lab not in label_to_idx:
                label_to_idx[int(lab)] = idx
        planet_masses = {5: 9.54e-4 * M_sun, 6: 2.85e-4 * M_sun, 7: 4.37e-5 * M_sun, 8: 5.15e-5 * M_sun}
        planet_a = {5: 0.52, 6: 0.95, 7: 1.92, 8: 3.01}
        extra_pos = []; extra_vel = []; extra_names = []
        _fb_parent = {5: "Jupiter", 6: "Saturn", 7: "Uranus", 8: "Neptune"}
        counts = {5: 20, 6: 20, 7: 10, 8: 10}  # sintéticos extras estáveis, dentro de Hill
        for p_label, n_extra in counts.items():
            if p_label not in label_to_idx:
                continue
            p_idx = label_to_idx[p_label]
            p_pos = pos[p_idx]; p_vel = vel[p_idx]; p_mass = planet_masses[p_label]
            hill = planet_a[p_label] * (p_mass / (3 * M_sun)) ** (1 / 3)
            for k in range(n_extra):
                extra_names.append(f"Synth-{_fb_parent.get(p_label, p_label)}-{k + 1:03d}")
                # apenas regulares estáveis: a 0.15-0.45 Hill, e<0.05, i<2°
                a = rng.uniform(0.15 * hill, 0.45 * hill)
                e = rng.uniform(0, 0.03)
                inc = rng.uniform(0, 2.0) * _math.pi / 180
                M = rng.uniform(0, 360)
                E = _math.radians(M) if e < 0.8 else _math.pi
                for __ in range(15):
                    denom = 1 - e * _math.cos(E)
                    if abs(denom) < 1e-12:
                        denom = 1e-12
                    dE = (E - e * _math.sin(E) - _math.radians(M)) / denom
                    E -= dE
                    if abs(dE) < 1e-12:
                        break
                cosE = _math.cos(E); sinE = _math.sin(E); sq = _math.sqrt(max(0, 1 - e * e))
                denom = 1 - e * cosE
                if abs(denom) < 1e-12:
                    denom = 1e-12
                x_orb = a * (cosE - e); y_orb = a * sq * sinE
                n = _math.sqrt(G * p_mass / (a * a * a + 1e-18))
                vx_orb = -n * a * sinE / denom; vy_orb = n * a * sq * cosE / denom
                x = x_orb; y = y_orb * _math.cos(inc); z = y_orb * _math.sin(inc)
                vx = vx_orb; vy = vy_orb * _math.cos(inc); vz = vy_orb * _math.sin(inc)
                extra_pos.append(p_pos + np.array([x, y, z]))
                extra_vel.append(p_vel + np.array([vx, vy, vz]))
        if extra_pos:
            pos_extra = __import__("numpy").vstack(extra_pos)
            vel_extra = __import__("numpy").vstack(extra_vel)
            masses_extra = __import__("numpy").full(len(extra_pos), 1e-11)
            labels_extra = __import__("numpy").full(len(extra_pos), 9, dtype=__import__("numpy").int32)
            pos = __import__("numpy").vstack([pos, pos_extra])
            vel = __import__("numpy").vstack([vel, vel_extra])
            masses = __import__("numpy").concatenate([masses, masses_extra])
            labels = __import__("numpy").concatenate([labels, labels_extra])
            names = list(names) + extra_names
    if n_asteroids:
        _np2 = __import__("numpy")
        _ap, _av, _am, _al, _an = _load_asteroids(n_asteroids)
        if _ap:
            pos = _np2.vstack([pos] + _ap)
            vel = _np2.vstack([vel] + _av)
            masses = _np2.concatenate([masses, _np2.array(_am)])
            labels = _np2.concatenate([labels, _np2.array(_al, dtype=_np2.int32)])
            names = list(names) + _an
    names = __import__("numpy").array(list(names)[:len(masses)], dtype=object)
    return pos.astype(__import__("numpy").float64), vel.astype(__import__("numpy").float64), masses.astype(__import__("numpy").float64), labels, names


def generate_satellite_infall(
    N_host_disk: int = 600,
    N_host_bulge: int = 200,
    N_sat: int = 200,
    M_host: float = 1.0,
    M_sat: float = 0.15,
    apocenter: float = 4.0,
    eccentricity: float = 0.7,
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
    seed: int = 42,
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Galáxia hospedeira + satélite anão em órbita excêntrica — stripping de maré."""
    pos_h, vel_h, m_h, lab_h = generate_spiral_galaxy(N_disk=N_host_disk, N_bulge=N_host_bulge, M_total=M_host, seed=seed, eps=eps, G=G)
    # satélite: Plummer pequeno
    rng = np.random.default_rng(seed + 99)
    a_sat = 0.25
    r_s = _plummer_radii(rng, a_sat, N_sat)
    dirs_s = _random_unit_vectors(rng, N_sat)
    pos_s = dirs_s * r_s[:, None]
    vel_s = _plummer_velocities(rng, r_s, a_sat, G, M_sat) * 0.6
    m_s = np.full(N_sat, M_sat / N_sat)
    pos_s -= np.average(pos_s, axis=0, weights=m_s)
    vel_s -= np.average(vel_s, axis=0, weights=m_s)
    # órbita do satélite ao redor do hospedeiro: coloca no apocentro em x
    r_apo = apocenter
    # velocidade no apocentro para e dada: v = sqrt(G*M_host*(1-e)/(r_apo*(1+e)))? simplificado vis-viva
    a_orb = r_apo / (1 + eccentricity)
    v_apo = np.sqrt(G * (M_host + M_sat) * (1 - eccentricity) / (r_apo + 1e-12))
    # coloca satélite em (r_apo,0,0.3)
    pos_s[:, 0] += r_apo
    pos_s[:, 2] += 0.3
    vel_s[:, 1] += v_apo  # tangencial
    vel_s[:, 0] += rng.normal(0, 0.02, N_sat)

    pos = np.vstack([pos_h, pos_s])
    vel = np.vstack([vel_h, vel_s])
    masses = np.concatenate([m_h, m_s])
    labels = np.concatenate([lab_h, np.full(N_sat, 5, dtype=np.int32)])
    pos -= np.average(pos, axis=0, weights=masses)
    vel -= np.average(vel, axis=0, weights=masses)
    return pos, vel, masses, labels

