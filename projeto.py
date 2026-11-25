import heapq
import math
import os
import re
import time
from collections import defaultdict

import folium
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from matplotlib.collections import LineCollection


LUGAR = "Curitiba, Paraná, Brazil"
ARQUIVO_GRAFO = "curitiba_drive.graphml"
FATOR_VELOCIDADE = 1.0
LIMITE_CURITIBA = (-25.65, -25.30, -49.45, -49.15)

ORIGEM = dict(
    nome="Hospital de Clínicas (UFPR)",
    busca="Hospital de Clínicas UFPR, Rua General Carneiro 181, Curitiba, Brasil",
    reserva=(-25.4413, -49.2613),
)

HOSPITAIS = [
    dict(nome="Hospital Universitário Cajuru",
         busca="Hospital Universitário Cajuru, Avenida São José 300, Curitiba, Brasil",
         reserva=(-25.4470, -49.2300)),
    dict(nome="Hospital do Trabalhador",
         busca="Hospital do Trabalhador, Avenida República Argentina 4406, Curitiba, Brasil",
         reserva=(-25.4790, -49.2680)),
    dict(nome="Hospital Pequeno Príncipe",
         busca="Hospital Pequeno Príncipe, Rua Desembargador Motta 1070, Curitiba, Brasil",
         reserva=(-25.4430, -49.2745)),
    dict(nome="Hospital Evangélico Mackenzie",
         busca="Hospital Evangélico Mackenzie, Alameda Augusto Stellfeld 1908, Curitiba, Brasil",
         reserva=(-25.4378, -49.2653)),
    dict(nome="Hospital Erasto Gaertner",
         busca="Hospital Erasto Gaertner, Rua Doutor Ovande do Amaral 201, Curitiba, Brasil",
         reserva=(-25.4505, -49.2341)),
]
CORES = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

VELOCIDADE_PADRAO_KMH = {
    "motorway": 80, "motorway_link": 60, "trunk": 70, "trunk_link": 50,
    "primary": 60, "primary_link": 40, "secondary": 50, "secondary_link": 40,
    "tertiary": 40, "tertiary_link": 30, "unclassified": 40,
    "residential": 30, "living_street": 20, "service": 20,
}
VELOCIDADE_RESERVA_KMH = 30


def interpretar_maxspeed(valor):
    if valor is None:
        return None
    texto = str(valor).lower()
    numeros = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", texto)]
    if not numeros:
        return None
    v = min(numeros)
    if "mph" in texto:
        v *= 1.609
    return v if 5 <= v <= 130 else None


def velocidade_padrao(highway):
    for tipo in re.findall(r"[a-z_]+", str(highway).lower()):
        if tipo in VELOCIDADE_PADRAO_KMH:
            return VELOCIDADE_PADRAO_KMH[tipo]
    return VELOCIDADE_RESERVA_KMH


def carregar_grafo():
    if os.path.exists(ARQUIVO_GRAFO):
        G = ox.load_graphml(ARQUIVO_GRAFO)
    else:
        G = ox.graph_from_place(LUGAR, network_type="drive")
        ox.save_graphml(G, ARQUIVO_GRAFO)
    maior = max(nx.strongly_connected_components(G), key=len)
    return G.subgraph(maior).copy()


def preparar_pesos(G):
    com_limite = 0
    for _, _, _, d in G.edges(keys=True, data=True):
        d["length"] = float(d["length"])
        vel = interpretar_maxspeed(d.get("maxspeed"))
        if vel is None:
            vel = velocidade_padrao(d.get("highway"))
        else:
            com_limite += 1
        vel *= FATOR_VELOCIDADE
        d["velocidade_kmh"] = vel
        d["travel_time"] = d["length"] / (vel / 3.6)
    return com_limite / G.number_of_edges()


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def obter_coordenada(local):
    try:
        lat, lon = ox.geocode(local["busca"])
        if LIMITE_CURITIBA[0] <= lat <= LIMITE_CURITIBA[1] and LIMITE_CURITIBA[2] <= lon <= LIMITE_CURITIBA[3]:
            return (lat, lon), "geocodificada"
    except Exception:
        pass
    return local["reserva"], "reserva (aproximada)"


def no_mais_proximo(coord, lat, lon):
    return min(coord, key=lambda n: haversine_m(lat, lon, coord[n][0], coord[n][1]))


def construir_adjacencia(G, peso):
    melhor = {}
    for u, v, _, d in G.edges(keys=True, data=True):
        if (u, v) not in melhor or d[peso] < melhor[(u, v)][0]:
            melhor[(u, v)] = (d[peso], d)
    adj = defaultdict(list)
    for (u, v), (w, d) in melhor.items():
        adj[u].append((v, w, d))
    return adj


print("Carregando a malha viária de Curitiba (na 1ª vez pode levar alguns minutos)...")
G = carregar_grafo()
cobertura = preparar_pesos(G)
COORD = {n: (d["y"], d["x"]) for n, d in G.nodes(data=True)}
VMAX_MS = max(d["velocidade_kmh"] for _, _, d in G.edges(data=True)) / 3.6

print(f"Grafo: {G.number_of_nodes():,} vértices | {G.number_of_edges():,} arestas")
print(f"Arestas com limite de velocidade cadastrado no OSM: {cobertura:.1%} "
      f"(o restante usa velocidade típica por tipo de via)")


INF = float("inf")


def nome_via(d):
    nome = d.get("name")
    if not nome or isinstance(nome, float):
        return "via sem nome"
    return " / ".join(nome) if isinstance(nome, list) else str(nome)


def dijkstra(adj, origem, destinos=None, trace=0, unidade=""):
    dist = {origem: 0.0}
    pred = {}
    fila = [(0.0, origem)]
    fechados = set()
    faltam = set(destinos) if destinos else None

    while fila:
        d_u, u = heapq.heappop(fila)
        if u in fechados:
            continue
        fechados.add(u)
        mostrar = len(fechados) <= trace
        if mostrar:
            print(f"\nIteração {len(fechados)}: retira o nó {u} (menor custo da fila) -> d[{u}] = {d_u:,.1f} {unidade}")

        if faltam is not None:
            faltam.discard(u)
            if not faltam:
                break

        for v, w, d in adj.get(u, ()):
            if v in fechados:
                continue
            novo = d_u + w
            antigo = dist.get(v, INF)
            melhora = novo < antigo
            if mostrar:
                antigo_txt = "∞" if antigo == INF else f"{antigo:,.1f}"
                print(f"   {u} -> {v} [{nome_via(d)[:28]}]  d[{v}] = min({antigo_txt}, {d_u:,.1f} + {w:,.1f} = {novo:,.1f})"
                      f"  {'✔ atualiza' if melhora else '✘ mantém'}")
            if melhora:
                dist[v] = novo
                pred[v] = (u, d)
                heapq.heappush(fila, (novo, v))
    return dist, pred, len(fechados)


def a_estrela(adj, origem, destino, peso):
    lat_d, lon_d = COORD[destino]

    def h(n):
        dm = haversine_m(COORD[n][0], COORD[n][1], lat_d, lon_d)
        return dm if peso == "length" else dm / VMAX_MS

    g = {origem: 0.0}
    pred = {}
    fila = [(h(origem), 0.0, origem)]
    fechados = set()
    while fila:
        _, g_u, u = heapq.heappop(fila)
        if u in fechados:
            continue
        fechados.add(u)
        if u == destino:
            break
        for v, w, d in adj.get(u, ()):
            if v in fechados:
                continue
            novo = g_u + w
            if novo < g.get(v, INF):
                g[v] = novo
                pred[v] = (u, d)
                heapq.heappush(fila, (novo + h(v), novo, v))
    return g.get(destino, INF), pred, len(fechados)


def reconstruir(pred, origem, destino):
    trecho, atual = [], destino
    while atual != origem:
        u, d = pred[atual]
        trecho.append((u, atual, d))
        atual = u
    trecho.reverse()
    return trecho


def metricas(trecho):
    metros = sum(d["length"] for _, _, d in trecho)
    segundos = sum(d["travel_time"] for _, _, d in trecho)
    return metros / 1000, segundos


def mmss(segundos):
    m, s = divmod(int(round(segundos)), 60)
    return f"{m:02d}:{s:02d}"


coord_origem, fonte_o = obter_coordenada(ORIGEM)
print(f"{ORIGEM['nome']}: {coord_origem} [{fonte_o}]")
nos_dest, coords_dest = {}, {}
for h in HOSPITAIS:
    coords_dest[h["nome"]], fonte = obter_coordenada(h)
    nos_dest[h["nome"]] = no_mais_proximo(COORD, *coords_dest[h["nome"]])
    print(f"{h['nome']}: {coords_dest[h['nome']]} [{fonte}]")
no_origem = no_mais_proximo(COORD, *coord_origem)

adj_dist = construir_adjacencia(G, "length")
adj_tempo = construir_adjacencia(G, "travel_time")

primeiro = HOSPITAIS[0]["nome"]
print(f"\n=== Passo a passo (critério: TEMPO) até {primeiro} — 4 primeiras iterações ===")
dijkstra(adj_tempo, no_origem, [nos_dest[primeiro]], trace=4, unidade="s")


alvos = set(nos_dest.values())

t0 = time.perf_counter()
dist_l, pred_l, exp_l = dijkstra(adj_dist, no_origem, alvos)
t_l = time.perf_counter() - t0
t0 = time.perf_counter()
dist_t, pred_t, exp_t = dijkstra(adj_tempo, no_origem, alvos)
t_t = time.perf_counter() - t0
print(f"Dijkstra (distância): {t_l:.2f}s, {exp_l:,} nós fechados | Dijkstra (tempo): {t_t:.2f}s, {exp_t:,} nós fechados")

rotas, linhas = {}, []
for h in HOSPITAIS:
    nome, no = h["nome"], nos_dest[h["nome"]]
    curta = reconstruir(pred_l, no_origem, no)
    rapida = reconstruir(pred_t, no_origem, no)
    km_c, s_c = metricas(curta)
    km_r, s_r = metricas(rapida)
    reta = haversine_m(*coord_origem, *coords_dest[nome]) / 1000
    rotas[nome] = {"curta": curta, "rapida": rapida}
    linhas.append({
        "Hospital": nome,
        "Linha reta (km)": round(reta, 2),
        "Mais curta (km)": round(km_c, 2),
        "Tempo na mais curta": mmss(s_c),
        "Mais rápida (km)": round(km_r, 2),
        "Tempo na mais rápida": mmss(s_r),
        "Tempo ganho": mmss(s_c - s_r),
        "Ganho (%)": round(100 * (s_c - s_r) / s_c, 1) if s_c else 0.0,
        "_s_c": s_c, "_s_r": s_r, "_km_c": km_c, "_km_r": km_r,
    })
tabela = pd.DataFrame(linhas)
colunas_visiveis = [c for c in tabela.columns if not c.startswith("_")]
print()
print(tabela[colunas_visiveis].to_string(index=False))

melhor = tabela.loc[tabela["_s_r"].idxmin()]
print(f"\nHospital mais rápido de alcançar a partir do HC: {melhor['Hospital']} "
      f"({melhor['Tempo na mais rápida']} — {melhor['Mais rápida (km)']:.2f} km)")


comparacao = []
for h in HOSPITAIS:
    nome, no = h["nome"], nos_dest[h["nome"]]

    assert math.isclose(dist_t[no], nx.shortest_path_length(G, no_origem, no, weight="travel_time"), rel_tol=1e-9)
    assert math.isclose(dist_l[no], nx.shortest_path_length(G, no_origem, no, weight="length"), rel_tol=1e-9)

    t0 = time.perf_counter()
    _, _, n_dij = dijkstra(adj_tempo, no_origem, [no])
    t_dij = time.perf_counter() - t0
    t0 = time.perf_counter()
    custo_a, _, n_a = a_estrela(adj_tempo, no_origem, no, "travel_time")
    t_a = time.perf_counter() - t0
    assert math.isclose(custo_a, dist_t[no], rel_tol=1e-9), "A* deve achar o mesmo custo ótimo do Dijkstra"

    comparacao.append({
        "Hospital": nome,
        "Dijkstra: nós": n_dij, "Dijkstra: ms": round(t_dij * 1000, 1),
        "A*: nós": n_a, "A*: ms": round(t_a * 1000, 1),
        "Redução de nós (%)": round(100 * (1 - n_a / n_dij), 1),
    })
print("✔ Nosso Dijkstra bate com o networkx (distância e tempo) e o A* encontra o mesmo custo ótimo.\n")
print(pd.DataFrame(comparacao).to_string(index=False))


def instrucoes(trecho):
    passos = []
    for _, _, d in trecho:
        via = nome_via(d)
        if passos and passos[-1]["via"] == via:
            passos[-1]["m"] += d["length"]
            passos[-1]["s"] += d["travel_time"]
        else:
            passos.append({"via": via, "m": d["length"], "s": d["travel_time"]})
    acum_m = acum_s = 0
    saida = []
    for p in passos:
        acum_m += p["m"]
        acum_s += p["s"]
        saida.append({
            "Via": p["via"][:40],
            "Trecho (km)": round(p["m"] / 1000, 2),
            "Vel. média (km/h)": round(p["m"] / p["s"] * 3.6) if p["s"] else 0,
            "Tempo do trecho": mmss(p["s"]),
            "Acumulado (km)": round(acum_m / 1000, 2),
            "Acumulado (mm:ss)": mmss(acum_s),
        })
    return pd.DataFrame(saida)


print(f"Rota mais rápida: {ORIGEM['nome']} -> {melhor['Hospital']}\n")
print(instrucoes(rotas[melhor["Hospital"]]["rapida"]).to_string(index=False))


def linha_da_rota(trecho):
    pontos = []
    for u, v, d in trecho:
        geom = d.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            pontos.extend((y, x) for x, y in geom.coords)
        else:
            pontos.extend([COORD[u], COORD[v]])
    return pontos


mapa = folium.Map(location=coord_origem, zoom_start=13, tiles="OpenStreetMap")
fg_rapida = folium.FeatureGroup(name="Rotas mais RÁPIDAS (tempo)").add_to(mapa)
fg_curta = folium.FeatureGroup(name="Rotas mais CURTAS (km)", show=False).add_to(mapa)

for i, h in enumerate(HOSPITAIS):
    nome, r = h["nome"], rotas[h["nome"]]
    km_r, s_r = metricas(r["rapida"])
    km_c, s_c = metricas(r["curta"])
    folium.PolyLine(linha_da_rota(r["rapida"]), color=CORES[i], weight=6, opacity=0.9,
                    tooltip=f"{nome} — mais rápida: {km_r:.2f} km | {mmss(s_r)}").add_to(fg_rapida)
    folium.PolyLine(linha_da_rota(r["curta"]), color=CORES[i], weight=4, opacity=0.9, dash_array="8",
                    tooltip=f"{nome} — mais curta: {km_c:.2f} km | {mmss(s_c)}").add_to(fg_curta)
    folium.Marker(coords_dest[nome], tooltip=nome, popup=(f"<b>{nome}</b><br>Mais rápida: {km_r:.2f} km | {mmss(s_r)}"
                                                           f"<br>Mais curta: {km_c:.2f} km | {mmss(s_c)}"),
                  icon=folium.Icon(color="red", icon="h-square", prefix="fa")).add_to(mapa)
folium.Marker(coord_origem, tooltip=ORIGEM["nome"], popup=f"<b>{ORIGEM['nome']}</b><br>Ponto de partida",
              icon=folium.Icon(color="green", icon="ambulance", prefix="fa")).add_to(mapa)
folium.LayerControl().add_to(mapa)
mapa.save("mapa_rotas_curitiba.html")
print("Mapa interativo salvo em: mapa_rotas_curitiba.html")
mapa


SEGMENTOS = [[(COORD[u][1], COORD[u][0]), (COORD[v][1], COORD[v][0])] for u, v in G.edges()]


def nome_curto(nome):
    return nome.replace("Hospital Universitário ", "H. Univ. ").replace("Hospital ", "H. ")[:22]


fig = plt.figure(figsize=(18, 9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1])
ax_m, ax_t, ax_k = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

ax_m.add_collection(LineCollection(SEGMENTOS, colors="#cfcfcf", linewidths=0.3))
lats_all, lons_all = [coord_origem[0]], [coord_origem[1]]
for i, h in enumerate(HOSPITAIS):
    pts = linha_da_rota(rotas[h["nome"]]["rapida"])
    la, lo = zip(*pts)
    lats_all += la
    lons_all += lo
    ax_m.plot(lo, la, color=CORES[i], lw=2.6, label=h["nome"], zorder=3)
    ax_m.scatter(coords_dest[h["nome"]][1], coords_dest[h["nome"]][0], marker="H", s=170, color=CORES[i],
                 edgecolor="black", zorder=5)
ax_m.scatter(coord_origem[1], coord_origem[0], marker="*", s=420, color="gold", edgecolor="black",
             zorder=6, label=ORIGEM["nome"])
mg = 0.01
ax_m.set_xlim(min(lons_all) - mg, max(lons_all) + mg)
ax_m.set_ylim(min(lats_all) - mg, max(lats_all) + mg)
ax_m.set_aspect(1 / math.cos(math.radians(coord_origem[0])))
ax_m.set_title("Rotas mais rápidas a partir do Hospital de Clínicas (Dijkstra, peso = tempo)", fontweight="bold")
ax_m.legend(loc="best", fontsize=8)
ax_m.set_xticks([])
ax_m.set_yticks([])

x = np.arange(len(HOSPITAIS))
rotulos = [nome_curto(h["nome"]) for h in HOSPITAIS]
for ax, col_c, col_r, titulo, ylabel, fmt in [
    (ax_t, "_s_c", "_s_r", "ANTES × DEPOIS — tempo de deslocamento", "minutos", lambda v: mmss(v)),
    (ax_k, "_km_c", "_km_r", "Distância percorrida", "km", lambda v: f"{v:.1f}"),
]:
    escala = 60 if col_c == "_s_c" else 1
    vc, vr = tabela[col_c] / escala, tabela[col_r] / escala
    ax.bar(x - 0.2, vc, 0.4, color="#9e9e9e", label="Antes: rota de menor distância")
    ax.bar(x + 0.2, vr, 0.4, color="#d62728", label="Depois: rota de menor tempo")
    for xi, (a, b) in enumerate(zip(tabela[col_c], tabela[col_r])):
        ax.text(xi - 0.2, a / escala, fmt(a), ha="center", va="bottom", fontsize=7)
        ax.text(xi + 0.2, b / escala, fmt(b), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(rotulos, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontweight="bold")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("painel_executivo.png", dpi=150, bbox_inches="tight")
plt.show()


s_antes, s_depois = tabela["_s_c"].sum(), tabela["_s_r"].sum()
print("=== GANHO DA OTIMIZAÇÃO (rota de menor distância -> rota de menor tempo) ===")
print(f"Tempo somado dos 5 trajetos: {mmss(s_antes)} -> {mmss(s_depois)}  "
      f"(economia de {mmss(s_antes - s_depois)} = {100 * (s_antes - s_depois) / s_antes:.1f}%)")
print(f"Km somados:                  {tabela['_km_c'].sum():.2f} km -> {tabela['_km_r'].sum():.2f} km "
      f"(a rota mais rápida percorre {tabela['_km_r'].sum() - tabela['_km_c'].sum():+.2f} km)")
print(f"Maior ganho individual: {tabela.loc[tabela['Ganho (%)'].idxmax(), 'Hospital']} "
      f"({tabela['Ganho (%)'].max():.1f}%)")

FATOR_CONGESTIONAMENTO = 0.5
ARTERIAIS = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link",
             "secondary", "secondary_link"}
for _, _, _, d in G.edges(keys=True, data=True):
    arterial = bool(ARTERIAIS & set(re.findall(r"[a-z_]+", str(d.get("highway")).lower())))
    d["travel_time_cong"] = d["travel_time"] / (FATOR_CONGESTIONAMENTO if arterial else 1.0)

adj_cong = construir_adjacencia(G, "travel_time_cong")
dist_c, pred_c, _ = dijkstra(adj_cong, no_origem, alvos)

print(f"\n=== SENSIBILIDADE: vias arteriais a {FATOR_CONGESTIONAMENTO:.0%} da velocidade ===")
sens = []
for h in HOSPITAIS:
    nome, no = h["nome"], nos_dest[h["nome"]]
    s_livre_sob_cong = sum(d["travel_time_cong"] for _, _, d in rotas[nome]["rapida"])
    s_reotimizada = dist_c[no]
    sens.append({
        "Hospital": nome,
        "Rota antiga sob trânsito": mmss(s_livre_sob_cong),
        "Rota reotimizada": mmss(s_reotimizada),
        "Ganho de reotimizar (%)": round(100 * (s_livre_sob_cong - s_reotimizada) / s_livre_sob_cong, 1),
    })
print(pd.DataFrame(sens).to_string(index=False))
print("\nLeitura: se o ganho de reotimizar for > 0, a rota ótima muda com o trânsito — "
      "por isso um sistema real deve recalcular as rotas com dados de tráfego em tempo real.")