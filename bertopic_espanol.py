
# %% 0. Imports y parámetros --------------------------------------------------
import re
import random
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUT_DIR = BASE_DIR / "resultados"
OUT_DIR.mkdir(exist_ok=True)

CSV_PATH = BASE_DIR / "IMDB Dataset SPANISH.csv"
MIN_PALABRAS = 5       
MAX_RESENAS = 5_000    
TROZO_PALABRAS = 200     

EMBEDDING_MODEL = "hiiamsid/sentence_similarity_spanish_es"
BATCH_SIZE = 16


# %% 1. Cargar y limpiar el corpus --------------------------------------------
def limpiar(texto):
    texto = re.sub(r"<br\s*/?>", " ", texto, flags=re.IGNORECASE)   # etiquetas <br />
    return re.sub(r"\s+", " ", texto).strip()


resenas = pd.read_csv(CSV_PATH, usecols=["review_es"])["review_es"].dropna().astype(str)
resenas = resenas.map(limpiar)
resenas = resenas[resenas.str.split().str.len() >= MIN_PALABRAS]

total_validas = len(resenas)
resenas = resenas.drop_duplicates()
duplicados = total_validas - len(resenas)
print(f"Reseñas válidas: {total_validas:,} | duplicados eliminados: {duplicados:,}")

if len(resenas) > MAX_RESENAS:
    resenas = resenas.sample(MAX_RESENAS, random_state=SEED)   # muestra reproducible

docs_muestra = resenas.tolist()
PALABRAS_ES = {
    "el", "la", "los", "las", "de", "del", "que", "y", "en", "un", "una", "unos",
    "unas", "no", "lo", "se", "con", "por", "para", "como", "es", "son", "muy",
    "más", "pero", "esta", "este", "también", "su", "sus", "me", "ya", "al", "mi",
    "fue", "ser", "está", "están", "película", "películas", "historia", "porque",
    "aunque", "sobre", "todo", "bien", "mejor", "cuando", "después", "nunca",
    "donde", "quien", "quién",
}
PALABRAS_EN = set(ENGLISH_STOP_WORDS) - PALABRAS_ES


def parece_ingles(texto):
    palabras = re.findall(r"(?u)\b\w+\b", texto.lower())
    ingles = sum(p in PALABRAS_EN for p in palabras)
    espanol = sum(p in PALABRAS_ES for p in palabras)
    return ingles >= 8 and ingles > 1.5 * espanol and ingles >= 0.12 * len(palabras)


espanol = [not parece_ingles(doc) for doc in docs_muestra]
resenas_ingles = len(docs_muestra) - sum(espanol)
docs = [doc for doc, es in zip(docs_muestra, espanol) if es]
print(f"Reseñas en inglés detectadas en review_es y excluidas: {resenas_ingles:,}")
print(f"Reseñas en español para modelar: {len(docs):,}")

if len(docs) < 500:
    print("OJO: la actividad pide al menos 500 segmentos")

# Estadísticas de la segmentación (van al reporte)
longitudes = np.array([len(d.split()) for d in docs])
stats = pd.Series({
    "criterio": "muestra aleatoria reproducible; una reseña en español = un documento",
    "num_segmentos": len(docs),
    "resenas_ingles_excluidas": resenas_ingles,
    "duplicados_eliminados": duplicados,
    "palabras_media": round(longitudes.mean(), 1),
    "palabras_mediana": float(np.median(longitudes)),
    "palabras_min": int(longitudes.min()),
    "palabras_max": int(longitudes.max()),
    "palabras_p95": float(np.percentile(longitudes, 95)),
})
print(stats)
stats.to_csv(OUT_DIR / "estadisticas_segmentacion.csv")

n_trozos = [
    max(1, (len(d.split()) + TROZO_PALABRAS - 1) // TROZO_PALABRAS)
    for d in docs
]
pct_partidas = 100 * np.mean(np.array(n_trozos) > 1)
pd.Series({
    "trozo_palabras": TROZO_PALABRAS,
    "resenas_partidas_%": round(pct_partidas, 2),
    "trozos_totales": sum(n_trozos),
}).to_csv(OUT_DIR / "truncamiento.csv")
print(f"Reseñas divididas en varios trozos: {pct_partidas:.1f}%")

# Histograma de longitudes
plt.figure(figsize=(8, 4))
plt.hist(longitudes, bins=50, color="#2b6cb0", edgecolor="white")
plt.axvline(longitudes.mean(), color="red", linestyle="--", label=f"media = {longitudes.mean():.1f}")
plt.xlabel("Palabras por reseña")
plt.ylabel("Número de reseñas")
plt.title("Distribución de longitudes")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "histograma_longitudes.png", dpi=200)
plt.close()


# %% 2. Embeddings (se guardan para no recalcularlos) -------------------------
hash_muestra = hashlib.md5("".join(docs_muestra).encode("utf-8")).hexdigest()[:8]
hash_corpus = hashlib.md5("".join(docs).encode("utf-8")).hexdigest()[:8]
CACHE_MUESTRA = OUT_DIR / f"embeddings_{hash_muestra}.npy"
CACHE_PATH = OUT_DIR / f"embeddings_{hash_corpus}.npy"

if CACHE_PATH.is_file():
    embeddings = np.load(CACHE_PATH)
    print("Embeddings cargados desde", CACHE_PATH.name, embeddings.shape)
elif CACHE_MUESTRA.is_file():
    embeddings_muestra = np.load(CACHE_MUESTRA)
    if len(embeddings_muestra) != len(docs_muestra):
        raise ValueError("La caché de embeddings no coincide con la muestra.")
    embeddings = embeddings_muestra[np.asarray(espanol)]
    np.save(CACHE_PATH, embeddings)
    del embeddings_muestra
    print("Embeddings en español extraídos de la caché existente:", embeddings.shape)
else:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Calculando embeddings con {EMBEDDING_MODEL} en {device} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    embedder.max_seq_length = 512

    # Parto cada reseña en trozos de TROZO_PALABRAS palabras y luego promedio sus embeddings
    trozos = []
    for d in docs:
        palabras = d.split()
        partes = [" ".join(palabras[i:i + TROZO_PALABRAS]) for i in range(0, len(palabras), TROZO_PALABRAS)]
        trozos.extend(partes)

    emb_trozos = embedder.encode(trozos, batch_size=BATCH_SIZE, show_progress_bar=True)

    embeddings, inicio = [], 0
    for n in n_trozos:
        embeddings.append(emb_trozos[inicio:inicio + n].mean(axis=0))
        inicio += n
    embeddings = np.array(embeddings, dtype=np.float32)
    np.save(CACHE_PATH, embeddings)


# %% 3. Stopwords y función que arma el modelo --------------------------------
STOPWORDS_BASE = """
a al algo algún alguna algunas algunos ahí ahora allí ante antes aquel aquella aquello aquí así
aunque cada casi como cómo con contra cual cuál cuando de del desde donde dónde dos durante e el
él ella ellas ellos en entonces entre era eran es esa esas ese eso esos esta ésta estaba estaban
están estar estas este éste esto estos fue fueron fui ha había habían han hasta hay hacia la las
le les lo los más me mi mis mismo mucho muchos muy nada ni ninguno no nos nuestro o otra otras
otro otros para pero poco por porque que qué quien quienes se sea según ser si sí sido siendo sin
sino sobre solo son su sus también tan tanto te tener tiene tienen toda todas todo todos tras tu
tus un una unas uno unos usted vez veces y ya yo
""".split()

STOPWORDS_PELIS = ["película", "películas", "pelicula", "peliculas", "film", "filme", "cine",
                   "ver", "hace", "hacer", "parece", "pues", "después"]

STOPWORDS_ES = STOPWORDS_BASE + STOPWORDS_PELIS


def construir_modelo(cfg):
    umap_model = UMAP(n_neighbors=cfg["n_neighbors"], n_components=5, min_dist=0.0,
                      metric="cosine", random_state=SEED)      
    hdbscan_model = HDBSCAN(min_cluster_size=cfg["min_cluster_size"],
                            min_samples=cfg["min_samples"], metric="euclidean",
                            cluster_selection_method="eom")
    
    vectorizer = CountVectorizer(stop_words=STOPWORDS_ES, ngram_range=cfg["ngram_range"])
    representacion = MaximalMarginalRelevance(diversity=0.3)  

    modelo = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                      vectorizer_model=vectorizer, representation_model=representacion,
                      nr_topics=cfg["nr_topics"], top_n_words=10,
                      calculate_probabilities=False, verbose=True)
    return modelo, vectorizer, representacion


# %% 4. Dos configuraciones ---------------------------------------------------
CONFIGS = {
    "A_detallada": dict(n_neighbors=15, min_cluster_size=30, min_samples=5,
                        ngram_range=(1, 1), nr_topics=None),
    "B_general":   dict(n_neighbors=15, min_cluster_size=60, min_samples=5,
                        ngram_range=(1, 2), nr_topics=20),
}

modelos, resultados, filas = {}, {}, []
for nombre, cfg in CONFIGS.items():
    print(f"\n===== Configuración {nombre}: {cfg} =====")
    modelo, vectorizer, representacion = construir_modelo(cfg)
    topics, _ = modelo.fit_transform(docs, embeddings)
    print(f"Tópicos encontrados: {len(set(topics)) - (1 if -1 in topics else 0)} | "
          f"outliers: {100 * (np.array(topics) == -1).mean():.1f}%")
    modelos[nombre] = (modelo, vectorizer, representacion)
    resultados[nombre] = np.array(topics)

    # Métricas para comparar: # de tópicos, % outliers y diversidad de palabras
    palabras = [w for t in modelo.get_topics() if t != -1 for w, _ in modelo.get_topic(t)[:10]]
    filas.append({
        "config": nombre,
        "num_topicos": len(set(topics)) - (1 if -1 in topics else 0),
        "outliers_%": round(100 * (resultados[nombre] == -1).mean(), 2),
        "diversidad_top10": round(len(set(palabras)) / max(len(palabras), 1), 4),
    })

comparacion = pd.DataFrame(filas)
print(comparacion)
comparacion.to_csv(OUT_DIR / "comparacion_configuraciones.csv", index=False)


# %% 5. Elegir configuración y reducir outliers -------------------------------
MEJOR = "B_general"   
modelo, vectorizer, representacion = modelos[MEJOR]
topics_orig = resultados[MEJOR]
pct_antes = 100 * (topics_orig == -1).mean()

if np.any(topics_orig == -1):
    topics_finales = modelo.reduce_outliers(
        docs, topics_orig, strategy="embeddings",
        embeddings=embeddings, threshold=0.5
    )
    modelo.update_topics(
        docs, topics=topics_finales, vectorizer_model=vectorizer,
        representation_model=representacion
    )
    estrategia_outliers = "reasignación por embeddings (threshold=0.5)"
else:
    topics_finales = topics_orig.copy()
    estrategia_outliers = "no fue necesario reasignar: no se detectaron outliers"

topics_finales = np.asarray(topics_finales)
pct_despues = 100 * (topics_finales == -1).mean()

print(f"Configuración elegida: {MEJOR}")
print(f"Outliers antes: {pct_antes:.2f}% | después: {pct_despues:.2f}%")
pd.DataFrame([{"outliers_antes_%": round(pct_antes, 2), "outliers_despues_%": round(pct_despues, 2),
               "estrategia": estrategia_outliers}]
             ).to_csv(OUT_DIR / "outliers.csv", index=False)


# %% 6. Tabla de tópicos y nombres propios ------------------------------------
info = modelo.get_topic_info()

NOMBRES = {
    -1: "Reseñas atípicas sin asignación",
    0: "Guerra y memoria histórica",
    1: "Valoraciones generales de películas",
    2: "Expectativa y experiencia de visionado",
    3: "Calidad del guion y adaptación",
    4: "Cine clásico y películas de culto",
    5: "Géneros y producciones de culto",
    6: "Biografías y figuras cinematográficas",
    7: "Comedia y eficacia del guion",
    8: "Elogios a actuaciones y narrativa",
    9: "Crítica negativa y calificaciones",
}
automaticos = info["Representation"].map(lambda p: " / ".join(p[:3]))
info["Nombre_propio"] = info["Topic"].map(NOMBRES).fillna(automaticos)

print(info[["Topic", "Count", "Nombre_propio", "Representation"]].to_string(index=False))
info.drop(columns=["Representative_Docs"], errors="ignore").to_csv(
    OUT_DIR / "topic_info.csv", index=False
)


# %% 7. Segmentos representativos (los 3 más cercanos al centro de cada tópico)
def representativos(topico, k=3):
    idx = np.flatnonzero(topics_finales == topico)
    centro = embeddings[idx].mean(axis=0, keepdims=True)
    similitud = cosine_similarity(embeddings[idx], centro).ravel()
    return idx[np.argsort(similitud)[-k:][::-1]]


filas = []
for _, fila in info[info["Topic"] != -1].iterrows():
    for i in representativos(fila["Topic"]):
        filas.append({"Topic": fila["Topic"], "Nombre_propio": fila["Nombre_propio"],
                      "Segmento": docs[i][:700] + ("..." if len(docs[i]) > 700 else "")})
reps_df = pd.DataFrame(filas)
reps_df.to_csv(OUT_DIR / "segmentos_representativos.csv", index=False)
print(reps_df.head(15).to_string())

# Tabla reseña -> tópico (ya con los outliers reasignados)
modelo.get_document_info(docs).to_csv(OUT_DIR / "documentos_y_topicos.csv", index=False)


# %% 8. Visualizaciones (HTML, ábrelos en el navegador) -----------------------
modelo.visualize_barchart(top_n_topics=10, n_words=8).write_html(OUT_DIR / "barchart.html")
modelo.visualize_topics().write_html(OUT_DIR / "mapa_intertopico.html")
modelo.visualize_hierarchy().write_html(OUT_DIR / "jerarquia.html")

print("\nListo. Archivos guardados en:", OUT_DIR)