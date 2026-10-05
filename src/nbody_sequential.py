"""
N-Body sequencial — Baseline O(N²)
====================================
Física: F_ij = G * m_i * m_j * r_vec / (r² + eps²)^(3/2)
Softening eps evita singularidade quando r->0 (Plummer softening).
Integradores: euler (simples) e leapfrog/Velocity Verlet (simplético, conserva energia).
Complexidade: O(N²) por timestep. tiled mantém O(N·tile) em memória.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

Float = np.float64
Vec3 = npt.NDArray[np.float64]

G_DEFAULT = 1.0
EPS_DEFAULT = 0.01
DT_DEFAULT = 1e-4


def generate_initial_conditions(
    N: int,
    seed: int = 42,
    radius: float = 1.0,
    mass_min: float = 0.5,
    mass_max: float = 1.5,
    vel_scale: float = 0.5,
    distribution: Literal["sphere", "cube", "plummer"] = "sphere",
) -> tuple[Vec3, Vec3, npt.NDArray[np.float64]]:
    rng = np.random.default_rng(seed)

    if distribution == "cube":
        pos = rng.uniform(-radius, radius, size=(N, 3))
    elif distribution == "sphere":
        pos = rng.normal(size=(N, 3))
        norms = np.linalg.norm(pos, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        pos = pos / norms
        r = radius * np.cbrt(rng.uniform(0, 1, size=(N, 1)))
        pos = pos * r
    elif distribution == "plummer":
        pos = rng.normal(scale=radius * 0.5, size=(N, 3))
        pos = np.clip(pos, -radius * 2, radius * 2)
    else:
        raise ValueError(f"distribution desconhecida: {distribution}")

    vel = rng.normal(scale=vel_scale, size=(N, 3))
    masses = rng.uniform(mass_min, mass_max, size=N)

    total_mass = masses.sum()
    com_vel = (masses[:, None] * vel).sum(axis=0) / total_mass
    vel -= com_vel

    return pos.astype(np.float64), vel.astype(np.float64), masses.astype(np.float64)


# ---------------------------------------------------------------------------
# Forças / acelerações  —  O(N²) vetorizado
# ---------------------------------------------------------------------------


def compute_forces_reference_loop(
    pos: Vec3,
    masses: npt.NDArray[np.float64],
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
) -> Vec3:
    """Referência O(N²) com loops Python — só para validar vetorização.\n
    Essa é a mais bosta, só pra ilustrar o pensamento de como fazer 100% sequencial"""
    N = pos.shape[0] # pega o númeoro de estrelas 
    accel = np.zeros_like(pos) # aceleração instantânea para cada estrela
    eps2 = eps * eps # pré-calcula o epsilon pra n ter que fazer 1 trilhão de vezes (pra cada loop)
    for i in range(N):
        for j in range(N):
            if i == j: # pular pra não calcular a força do corpo consigo mesmo, nem faria sentido isso
                continue
            r_vec = pos[j] - pos[i] # direção da atração. tipo, imagina i na pos (1, 2, 1) atraindo j (1, 1, 1), o vetor resultante vai ser na direção (0, -1, 0), que é o corpo i descendo em y
            r2 = float(np.dot(r_vec, r_vec) + eps2) # distância quadrada entre os dois. Isso vale |r|² = r²
            inv_r3 = r2 ** (-1.5) # r³ = r² * r
            accel[i] += G * masses[j] * inv_r3 * r_vec
    return accel


def compute_forces(
    pos: Vec3,
    masses: npt.NDArray[np.float64],
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
) -> Vec3:
    """Vetorizado: calcula aceleração de cada partícula devido a todas as outras.
    r_vec[i,j] = pos[j]-pos[i], preenche (N,N,3)."""
    N = pos.shape[0] 
    # minha matriz de positions tem: (N, 3) - N linhas e 3 colunas (uma para cada x, y, z)
    # "a mas pq você tem que criar várias matrizes pra vetorizar?" porque o numpy só consegue vetorizar assim: com a matriz inteira. Ele não consegue fazer o trampo do numba que eu expliquei no outro arquivo

    r_vec = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]   # (N,N,3), porque np.newaxis cria um novo axis vazio (nova dimensão com tamanho 1) e o : só copia os valore da primeira dim da original. então:
                                                            # pos[np.newaxis, :, :] = (1, N, 3) 
                                                            # pos[:, np.newaxis, :] = (N, 1, 3)
                                                            # pronto, daí eu faço o cálculo tudo de uma vez vetorizado com numpy ao invés de sequencial com python
                                                            # Isso só funciona porque o numpy percebe que as matrizes são de tamanhos diferentes e clona a matriz na direção da dimensão inexistente
                                                            # https://imgur.com/a/360Nkko
                                                            # e também algo tem que ser dito sobre o numpy: ele é relativamente foda, porque vetorização é um trampo de corno, e ele faz isso automaticamente
                                                            # em C, você tem que criar os loops e fazer as multiplicações manualmente E AINDA passar o O3 pro compiler tentar se virar nos 30 pra autovetorizar
                                                            # o numpy já faz isso automaticamente SE OS TIPOS FOREM IGUAIS (que é o nosso caso, porque são todas mult de floats)
                                                            # então é: C é sempre mais foda (óbvio), mas o numpy não deixa o python ficar pra tão trás. 
                                                            # Vou falar que o C tem IRA 20k, o numpy deixa o python com um IRA de engenharia espertinho
                                                            # python puro tem IRA de estudante da área sul  
                                                            # abraços

    r2 = np.sum(r_vec * r_vec, axis=2) + eps * eps      # (N,N), então fica todos os resultados de cada par i-j ao quadrado somados apenas na direção z (axis = 2 AXIS COMEÇA NO 0, É 0, 1, 2, Z = 2)
                                                        # (2, 1) é a direção que o corpo 2 atrai o 1, bem fofinho
    np.fill_diagonal(r2, np.inf) # pra quando fizer a conta, dar zero quando for tratar do próprio corpo se atraindo
    inv_r3 = r2 ** (-1.5)  # (N,N)
    coeff = masses[np.newaxis, :] * inv_r3  # (N,N) array pra massa/r³ para cada corpo. Mesma coisa de antes: ele vai ver que masses e o r³ têm dimensões diferentes e vai clonar as massas horizontalmente, porque as massas já foram copiadas na vertical
    accel = G * np.sum(coeff[:, :, np.newaxis] * r_vec, axis=1)     # (N,3) # aplicação da fórmula pra cada corpo (somando) no sentido do y, fica certinho: m1*x/r³ + m2*x/r³ ... e o mesmo para cada axis. Dá a impressão de que ele vai chumbar a matriz e ela vai ficar "X, Z"
                                                                    # mas não fica porque ele já transforma de volta de Z pra Y. No final ela fica igual a da implementação sequencial: X = estrela, Y = aceleração (x, y, z) da estrela
    return accel


def compute_forces_tiled(
    pos: Vec3,
    masses: npt.NDArray[np.float64],
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
    tile: int = 512,
) -> Vec3:
    """Versão em blocos: processa j em tiles para manter memória O(N·tile)"""
    # vou precisar fazer umas contas pra demonstrar porque é melhor usar isso ao invés de usar direto o vetorizado:
    # pensa que eu faço pra 1000 estrelas, o máximo vai ser a matriz 1000x1000x3
    # são 3000000 floats = 3000000 * 8 bytes = 24 milhões de bytes = 22.9MB, legal né belezinha
    # agora faz com 10k: 2,3GB
    # então pra não dar tela azul no pc se você fizer bosta e colocar um número muito grande, tem essa implementação com tile
    # ele não paraleliza nada, mas não deixa estourar a memória do pc
    # a gente faz exatamente a mesma coisa que o de cima, mas dando passo de 512 em 512 ao invés de pegar do 0 até o N (com o :)
    N = pos.shape[0]
    accel = np.zeros_like(pos)
    eps2 = eps * eps
    for j0 in range(0, N, tile):
        j1 = min(j0 + tile, N)
        r_vec = pos[np.newaxis, j0:j1, :] - pos[:, np.newaxis, :]  # (N, tile, 3)
        r2 = np.sum(r_vec * r_vec, axis=2) + eps2  # (N, tile)
        if j0 < N:
            i_start = max(0, j0)
            i_end = min(N, j1)
            for i in range(i_start, i_end):
                t = i - j0
                r2[i, t] = np.inf
        inv_r3 = r2 ** (-1.5)  # (N, tile)
        coeff = masses[j0:j1][np.newaxis, :] * inv_r3  # (N, tile)
        accel += G * np.sum(coeff[:, :, np.newaxis] * r_vec, axis=1)
    return accel




# ---------------------------------------------------------------------------
# Energia (para validação)
# ---------------------------------------------------------------------------

def total_energy(
    pos: Vec3,
    vel: Vec3,
    masses: npt.NDArray[np.float64],
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
) -> tuple[float, float, float]:
    """Retorna (E_kin, E_pot, E_total)."""
    # como o nosso sistema é fechado e (infelizmente) não tem como tirar energia do nada, 
    # vamos usar isso pra garantir que o sistema não vai ganhar eneriga
    # é igual ensino méido: energia potencial se transforma em outros tipos de energia (calor, elástica, CINÉTICA). 
    # como não tem colisão no sistema, a única transformação é de potencial gravitacional pra cinética
    # se a soma das energias começar a mudar (muito), tem alguma coisa errada 
    e_kin = 0.5 * np.sum(masses * np.sum(vel * vel, axis=1))    # como energia é escalar, tem que usar a velocidade escalar. POR SORTE, a fórmula usa 
                                                                # V², então dá pra pegar o quad de cada coord (x, y, z) e somar na direção y
                                                                # depois só multiplica essas velocidades somadas por cada massa respectiva: 
                                                                # receba conhecimento: você tem a energia cinética pra cada corpo
                                                                # soma tudo e vc consegue a total do sistema
    # cuidado que fórmula que tem que usar é a da gravitação de newton, não a Epg = m*g*h, isso só vale pra corpos perto da terra
    # tem que usar: Epg = -G*m1*m2/r

    N = pos.shape[0]
    r_vec = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]   # (N,N,3) aqui é o mesmo truque que pra calcular as forças vetorizado: ao invés de fazer o trampo de cada par de estrela com loop
                                                            # eu vou fazer com multiplicação de matriz do numpy pra encontrar todas as distâncias entre os pares de estrelas
                                                            # ironicamente só aqui eu percebi que as matrizes são todas espelhadas, porque a força é simétrica entre os corpos
                                                            # mas sabia que é pior eu fazer esse bagulho de separar as matrizes? tipo, o numpy já tem esse trampo de pegar a matriz inteira e ele é rapido 
                                                            # porque opera sobre matrizes quadradas. Se eu quisesse manualmente calcular só metade da matriz, eu ia quebrar a estrutura do numpy e 
                                                            # no final ia acabar demorando mais do que deixar ele se virar
                                                            # talvez quando eu tiver fazendo o processamento paralelo seja mais legal de fazer isso
                                                            
    r = np.sqrt(np.sum(r_vec * r_vec, axis=2) + eps * eps)  # (N,N) como precisa pegar só o r puro pra conta do newton, tem que fazer a raiz das soma dos termos ao quadrado (multiplicando termo por termo mesmo)
    m_mat = masses[:, np.newaxis] * masses[np.newaxis, :]   # pré-calculo das multiplicações dos pares de massas 
    pot_mat = G * m_mat / r
    np.fill_diagonal(pot_mat, 0) # potencial do corpo consigo mesmo = 0
    e_pot = -0.5 * np.sum(pot_mat) # soma de todas as energias potenciais gravitacionais. Obviamente dividir por 2 porque elas são simétricas
    return float(e_kin), float(e_pot), float(e_kin + e_pot)


def total_energy_tiled(
    pos: Vec3,
    vel: Vec3,
    masses: npt.NDArray[np.float64],
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
    tile: int = 1024,
) -> tuple[float, float, float]:
    """Versão em blocos para N grande."""
    e_kin = 0.5 * np.sum(masses * np.sum(vel * vel, axis=1))
    N = pos.shape[0]
    eps2 = eps * eps
    e_pot = 0.0
    for j0 in range(0, N, tile):
        j1 = min(j0 + tile, N)
        r_vec = pos[np.newaxis, j0:j1, :] - pos[:, np.newaxis, :]  # (N,tile,3)
        r = np.sqrt(np.sum(r_vec * r_vec, axis=2) + eps2)  # (N,tile)
        m_mat = masses[:, np.newaxis] * masses[np.newaxis, j0:j1]  # (N,tile)
        pot_tile = G * m_mat / r
        for i in range(max(0, j0), min(N, j1)):
            pot_tile[i, i - j0] = 0
        e_pot += np.sum(pot_tile)
    e_pot = -0.5 * e_pot
    return float(e_kin), float(e_pot), float(e_kin + e_pot)


# ---------------------------------------------------------------------------
# Integradores
# ---------------------------------------------------------------------------


# mano eu to puto até agora com essa merda, eu simplesmente NÃO CONSIGO ENTENDER PORQUE tem tanto fetiche em torno dessa porra desse leapfrog vsf
# o do euler é mais bosta, eu entendo, mas não consigo achar ninguém que explique de verdade por que tem que usar a porra do leapfrog ao invés do euler 
# tipo, PORQUE ele resolve a merda que acontece na do euler

# tem GRÁFICO COMPARATIVO mostrando as diferenças de cada um, mas mesmo assim olhando só pra conta eu nãso consigo falar porque caralhos um dá certo e o outro não
# fisicamente eu entendo que o do euler tem as falhas de aproximar a órbita pra várias retas tangentes à órbita real, mas ainda não entendi porque o leapfrog corrige/ameniza essa falha

# edit: finalmente eu entendi o leapfrog 
# se algum dia eu falei mal, NÃO ERA EU
# mas pra resumir bastante: ele minimiza (NÃO ZERA) o erro em relação ao euler
# a sacada é muito boa
# ao invés de eu pegar e calcular posição e velocidade final como se toda a força entre aquele instante de tempo e o próximo
# viessem o mesmo instante anterior, eu divido por 2 e considero que metade da velocidade atual deriva do efeito
# das forças no passado e a outra metade do instante futuro

# não sei como eu eu posso explicar melhor isso
# eu recomendo fazer o que eu fiz pra entender: faz o passo a passo de uma iteração
# pra mim só fez sentido assim
# essa é a parte mais fácil
# a parte difícil é pensar porque o leapfrog não deixa acontecer o problema de "criação" de energia igual o euler deixa
# porque pensa: se tanto o leapfrog quanto o euler erram um pouco pos e vel, COMO o leapfrog conserva energia e o euler não?

# edit2: I SEE! I FINALLY SEE! ELE CONSERVA PORQUE AO MESMO TEMPO QUE PERDE VELOCIDADE, ELE GANHA ACELERAÇÃO AAAAAAAAAAAAA
# a órbita que era pra ser perfeitamente circular fica meio elíptica
# faz todo o sentido do mundo
# o que ele injeta de energai no começo, ele começa a perder com o tempo porque aumenta a aceleração e perde velocidade
# por isso que conserva a energia

def step_euler(
    pos: Vec3,
    vel: Vec3,
    masses: npt.NDArray[np.float64],
    dt: float,
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
    tiled: bool = False,
) -> tuple[Vec3, Vec3]:
    """Euler semi-implícito — simples, mas dissipativo."""
    accel = compute_forces_tiled(pos, masses, G, eps) if tiled else compute_forces(pos, masses, G, eps)
    vel_new = vel + accel * dt
    pos_new = pos + vel_new * dt
    return pos_new, vel_new


def step_leapfrog(
    pos: Vec3,
    vel: Vec3,
    accel: Vec3,
    masses: npt.NDArray[np.float64],
    dt: float,
    G: float = G_DEFAULT,
    eps: float = EPS_DEFAULT,
    tiled: bool = False,
) -> tuple[Vec3, Vec3, Vec3]:
    """Velocity Verlet / Leapfrog — simplético, conserva energia."""
    vel_half = vel + accel * (dt * 0.5)
    pos_new = pos + vel_half * dt
    accel_new = compute_forces_tiled(pos_new, masses, G, eps) if tiled else compute_forces(pos_new, masses, G, eps)
    vel_new = vel_half + accel_new * (dt * 0.5)
    return pos_new, vel_new, accel_new


# ---------------------------------------------------------------------------
# Simulação completa
# ---------------------------------------------------------------------------


# créditos pro delano das aulas de boas práticas de programação 🙏

@dataclass
class SimConfig:
    N: int = 1000
    steps: int = 100
    dt: float = DT_DEFAULT
    G: float = G_DEFAULT
    eps: float = EPS_DEFAULT
    integrator: Literal["euler", "leapfrog"] = "leapfrog"
    tiled: bool = False
    seed: int = 42
    distribution: Literal["sphere", "cube", "plummer"] = "sphere"


@dataclass
class SimResult:
    pos: Vec3
    vel: Vec3
    masses: npt.NDArray[np.float64]
    config: SimConfig
    elapsed: float
    energy_history: list[tuple[float, float, float]]
    trajectory: Vec3 | None = None


def run_simulation(
    config: SimConfig,
    save_trajectory: bool = False,
    energy_every: int = 10,
    verbose: bool = True,
) -> SimResult:
    """Roda simulação completa com medição de tempo e energia."""
    pos, vel, masses = generate_initial_conditions(
        config.N, seed=config.seed, distribution=config.distribution
    )

    energy_history: list[tuple[float, float, float]] = []   # en cinética, potencial grav e total do sistema
    trajectory = None
    if save_trajectory:
        trajectory = np.zeros((config.steps + 1, config.N, 3), dtype=np.float64)
        trajectory[0] = pos

    if config.N <= 2000:
        e0 = total_energy(pos, vel, masses, config.G, config.eps)
    else:
        e0 = total_energy_tiled(pos, vel, masses, config.G, config.eps)
    energy_history.append(e0)
    if verbose:
        print(f"[N={config.N} steps={config.steps} dt={config.dt} {config.integrator}{' tiled' if config.tiled else ''}] E0 kin={e0[0]:.6f} pot={e0[1]:.6f} tot={e0[2]:.6f}")

    if config.integrator == "leapfrog":
        if config.tiled:
            accel = compute_forces_tiled(pos, masses, config.G, config.eps)
        else:
            accel = compute_forces(pos, masses, config.G, config.eps)

    t0 = time.perf_counter()
    for step in range(1, config.steps + 1):     # isso é um bagulho que sempre vai ser sequencial: tem que fazer um step por vez porque o próximo é sempre dependente do anterior
        if config.integrator == "euler":
            pos, vel = step_euler(pos, vel, masses, config.dt, config.G, config.eps, config.tiled)
        elif config.integrator == "leapfrog":
            pos, vel, accel = step_leapfrog(pos, vel, accel, masses, config.dt, config.G, config.eps, config.tiled)
        else:
            raise ValueError(config.integrator)

        if save_trajectory:
            trajectory[step] = pos  # pra fazer a animação depois

        if step % energy_every == 0 or step == config.steps:    # calcular a energia do sistema a cada x steps e também se acabar a simulação (último step)
            if config.N <= 2000:
                e = total_energy(pos, vel, masses, config.G, config.eps)
            else:
                e = total_energy_tiled(pos, vel, masses, config.G, config.eps)
            energy_history.append(e)
            if verbose and (step % max(1, config.steps // 10) == 0 or step == config.steps):
                drift = abs(e[2] - e0[2]) / max(1e-12, abs(e0[2]))
                print(f"  step {step:5d}/{config.steps}  Etot={e[2]:.6f}  drift={drift:.2e}")

    elapsed = time.perf_counter() - t0 # tempo total de simulação pros benchmark
    if verbose:
        print(f"  -> tempo total: {elapsed:.3f}s  ({elapsed/config.steps*1000:.2f} ms/step)  N={config.N}")

    return SimResult(pos=pos, vel=vel, masses=masses, config=config, elapsed=elapsed, energy_history=energy_history, trajectory=trajectory)


if __name__ == "__main__":
    import argparse

    default_cfg = SimConfig()

    parser = argparse.ArgumentParser(description="N-Body sequencial")
    parser.add_argument("--N", type=int, default=default_cfg.N)
    parser.add_argument("--steps", type=int, default=default_cfg.steps)
    parser.add_argument("--dt", type=float, default=default_cfg.dt)
    parser.add_argument("--eps", type=float, default=default_cfg.eps)
    parser.add_argument("--integrator", choices=["euler", "leapfrog"], default=default_cfg.integrator)
    parser.add_argument("--tiled", action="store_true") # False por padrão se não passar
    parser.add_argument("--seed", type=int, default=default_cfg.seed)
    args = parser.parse_args()

    cfg = SimConfig(N=args.N, steps=args.steps, dt=args.dt, eps=args.eps, integrator=args.integrator, tiled=args.tiled, seed=args.seed)
    result = run_simulation(cfg, verbose=True)
    e0 = result.energy_history[0]
    e1 = result.energy_history[-1]
    drift = abs(e1[2] - e0[2]) / max(1e-12, abs(e0[2]))
    