"""Núcleo compartilhado pela aplicação e pelos três notebooks."""
from datetime import date
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

MODELOS = ['sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
           'sentence-transformers/distiluse-base-multilingual-cased-v2']
CATEGORIAS = {'infraestrutura', 'saúde', 'segurança', 'educação', 'meio ambiente'}

def carregar_base(arquivo='manifestacoes.json'):
    dados = json.loads(Path(arquivo).read_text(encoding='utf-8-sig')) if isinstance(arquivo, (str, Path)) else json.load(arquivo)
    if not isinstance(dados, list) or not dados:
        raise ValueError('A base deve ser uma lista JSON não vazia de manifestações.')
    ids = set()
    for registro in dados:
        if not isinstance(registro, dict) or not {'id','data','categoria_oficial','texto'} <= registro.keys():
            raise ValueError('Cada registro precisa de id, data, categoria_oficial e texto.')
        if not all(isinstance(registro[k], str) for k in ['id','data','categoria_oficial','texto']):
            raise ValueError('Os quatro campos obrigatórios devem ser strings.')
        if registro['id'] in ids or not registro['id'].strip():
            raise ValueError('IDs devem ser únicos e não vazios.')
        ids.add(registro['id'])
        date.fromisoformat(registro['data'])
        if registro['categoria_oficial'] not in CATEGORIAS:
            raise ValueError(f"Categoria desconhecida: {registro['categoria_oficial']}")
        if not registro['texto'].strip():
            raise ValueError('Não são aceitos textos vazios.')
    return pd.DataFrame(dados).sort_values('id').reset_index(drop=True)

@lru_cache(maxsize=2)
def carregar_modelo(nome=MODELOS[0]):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(nome, device='cpu', trust_remote_code=False)

def dividir(texto, chunk_size=300, chunk_overlap=60, estrategia='Recursiva'):
    if chunk_size < 1 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError('Use tamanho positivo e 0 <= overlap < chunk_size.')
    separadores = ['\n\n', '\n', '. ', ' ', ''] if estrategia == 'Recursiva' else ['']
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        separators=separadores, length_function=len, add_start_index=True)
    docs = splitter.create_documents([texto])
    return [{'chunk': i+1, 'inicio': d.metadata['start_index'], 'caracteres': len(d.page_content),
             'texto': d.page_content} for i, d in enumerate(docs)]

def codificar(textos, modelo=MODELOS[0]):
    """Média normalizada de janelas de tokens; evita truncamento silencioso."""
    model = carregar_modelo(modelo)
    capacidade = model.max_seq_length - model.tokenizer.num_special_tokens_to_add(pair=False)
    partes, donos = [], []
    for i, texto in enumerate(textos):
        ids = model.tokenizer.encode(texto, add_special_tokens=False, truncation=False, verbose=False)
        if not ids:
            raise ValueError('Texto sem tokens.')
        if len(ids) <= capacidade:
            partes.append(texto); donos.append(i)
        else:
            # Janela conservadora para que a re-tokenização do texto decodificado também caiba.
            tamanho = max(1, capacidade // 2)
            for inicio in range(0, len(ids), tamanho):
                trecho = model.tokenizer.decode(ids[inicio:inicio+tamanho], skip_special_tokens=True)
                if len(model.tokenizer.encode(trecho)) > model.max_seq_length:
                    raise ValueError('Uma janela excedeu o limite do modelo; reduza seu tamanho.')
                partes.append(trecho); donos.append(i)
    if not partes:
        return np.empty((0, model.get_sentence_embedding_dimension()))
    vetores = model.encode(partes, normalize_embeddings=True, show_progress_bar=False)
    donos = np.array(donos)
    agregados = np.stack([vetores[donos == i].mean(axis=0) for i in range(len(textos))])
    normas = np.linalg.norm(agregados, axis=1, keepdims=True)
    return np.divide(agregados, normas, out=np.zeros_like(agregados), where=normas != 0)

def pares_da_matriz(matriz, limiar=0.85, ids=None):
    if not -1 <= limiar <= 1:
        raise ValueError('O limiar deve pertencer a [-1,1].')
    ids = list(range(len(matriz))) if ids is None else list(ids)
    return sorted([{'id_a': ids[i], 'id_b': ids[j], 'similaridade': float(matriz[i,j])}
                   for i in range(len(ids)) for j in range(i+1,len(ids)) if matriz[i,j] > limiar],
                  key=lambda p: (-p['similaridade'], str(p['id_a']), str(p['id_b'])))

def detectar_duplicatas(textos, limiar=0.85, modelo=MODELOS[0]):
    if len(textos) < 2:
        return []
    return pares_da_matriz(cosine_similarity(codificar(textos, modelo)), limiar)

def projetar(embeddings, metodo='PCA'):
    x = np.asarray(embeddings)
    if len(x) < 2:
        return np.zeros((len(x),2))
    if metodo == 'PCA':
        n = min(2, len(x), x.shape[1])
        y = PCA(n_components=n, random_state=42).fit_transform(x)
        return np.pad(y, ((0,0),(0,2-n)))
    if metodo != 't-SNE':
        raise ValueError('Método desconhecido.')
    return TSNE(n_components=2, perplexity=min(10, len(x)-1), init='random',
                learning_rate='auto', random_state=42).fit_transform(x)

def avaliar_pares(preditos, verdade, ids):
    """verdade: lista exaustiva de pares confirmados, não apenas exemplos positivos."""
    universo = set(ids)
    reais = set()
    for a,b in verdade:
        if a == b or a not in universo or b not in universo:
            raise ValueError('Gabarito contém IDs inválidos.')
        reais.add(tuple(sorted((a,b))))
    previstos = {tuple(sorted((p['id_a'],p['id_b']))) for p in preditos}
    tp, fp, fn = previstos & reais, previstos - reais, reais - previstos
    precision = len(tp)/len(previstos) if previstos else 0.0
    recall = len(tp)/len(reais) if reais else 0.0
    return {'TP': sorted(tp), 'FP': sorted(fp), 'FN': sorted(fn), 'precision': precision,
            'recall': recall, 'F1': 2*precision*recall/(precision+recall) if precision+recall else 0.0}
