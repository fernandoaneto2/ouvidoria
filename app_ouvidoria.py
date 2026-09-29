"""Execute: streamlit run app_ouvidoria.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from ouvidoria_core import MODELOS, carregar_base, carregar_modelo, codificar, dividir, projetar

@st.cache_resource
def modelo_em_cache(nome):
    return carregar_modelo(nome)

@st.cache_data
def embeddings_em_cache(nome, textos):
    modelo_em_cache(nome)
    return codificar(list(textos), nome)

@st.cache_data
def projecao_em_cache(embeddings, metodo):
    return projetar(embeddings, metodo)

def grafico_pontos(coords, grupos, ids):
    fig, ax = plt.subplots(figsize=(9,5))
    for grupo in sorted(set(grupos)):
        mascara = np.asarray(grupos) == grupo
        ax.scatter(coords[mascara,0], coords[mascara,1], label=grupo, alpha=.8)
    for ponto, rotulo in zip(coords, ids):
        ax.annotate(rotulo, ponto, fontsize=6, xytext=(3,3), textcoords='offset points')
    ax.legend(fontsize=8); ax.set_xlabel('Componente 1'); ax.set_ylabel('Componente 2')
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

def main():
    st.set_page_config(page_title='Ouvidoria Inteligente', page_icon='💬', layout='wide')
    st.title('Ouvidoria Inteligente')
    st.caption('Triagem semântica de manifestações • Fernando Amorim Pontes Neto')
    nome = st.sidebar.selectbox('Modelo de embedding', MODELOS)
    k = st.sidebar.slider('Top-k', 1, 20, 5)
    arquivo = st.sidebar.file_uploader('Carregar manifestacoes.json', type=['json'])
    caminho = Path(__file__).with_name('manifestacoes.json')
    if arquivo is None and not caminho.exists():
        st.info('Carregue a base original em JSON ou coloque manifestacoes.json ao lado deste script.')
        st.code('[{"id":"M001","data":"2026-01-01","categoria_oficial":"infraestrutura","texto":"Descrição da manifestação"}]', language='json')
        return
    try:
        base = carregar_base(arquivo if arquivo is not None else caminho)
    except Exception as exc:
        st.error(f'Base inválida: {exc}'); return
    if arquivo is None and Path(__file__).with_name('PROVENIENCIA.json').exists():
        st.info('Base demonstrativa sintética criada para o exercício, sem dados pessoais reais.')
    if len(base) != 40:
        st.warning(f'Base com {len(base)} registros; o enunciado prevê 40.')
    st.sidebar.metric('Manifestações', len(base))
    try:
        with st.spinner('Preparando representações vetoriais...'):
            e = embeddings_em_cache(nome, tuple(base.texto))
    except Exception as exc:
        st.error(f'Modelo indisponível: {exc}')
        st.info('O primeiro uso requer internet para baixar os pesos. A análise ocorre localmente.'); return
    abas = st.tabs(['🔍 Busca Semântica', '📋 Base Completa', '🌐 Espaço Vetorial', '🧩 Chunking'])
    with abas[0]:
        consulta = st.text_area('Descreva o problema que você procura', 'A rua está cheia de buracos')
        if consulta.strip():
            try:
                q = embeddings_em_cache(nome, (consulta,))[0]
                scores = e @ q
                for i in np.argsort(-scores, kind='stable')[:min(k,len(base))]:
                    s = float(scores[i]); r = base.iloc[i]
                    cor = '🟢' if s > .7 else ('🟡' if s > .5 else '🔴')
                    with st.container(border=True):
                        st.markdown(f'**{cor} {r.id} · score {s:.3f}**')
                        st.caption(f'{r.categoria_oficial} · {r.data}'); st.write(r.texto)
                st.caption('Verde >0,7; amarelo >0,5; vermelho demais. Scores indicam similaridade, não probabilidade de duplicata.')
            except Exception as exc:
                st.error(f'Não foi possível processar a consulta: {exc}')
        else:
            st.info('Digite uma descrição para buscar.')
    with abas[1]:
        st.dataframe(base, hide_index=True, width='stretch')
        if st.button('Gerar matriz de similaridade'):
            matriz = e @ e.T
            fig, ax = plt.subplots(figsize=(10,8))
            img = ax.imshow(matriz, vmin=-1, vmax=1, cmap='coolwarm')
            ax.set_xticks(range(len(base)), base.id, rotation=90, fontsize=6)
            ax.set_yticks(range(len(base)), base.id, fontsize=6)
            fig.colorbar(img, ax=ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
            st.download_button('Baixar matriz CSV', pd.DataFrame(matriz,index=base.id,columns=base.id).to_csv().encode('utf-8-sig'), 'similaridades.csv')
    with abas[2]:
        metodo = st.selectbox('Projeção', ['PCA', 't-SNE'])
        grafico_pontos(projecao_em_cache(e, metodo), base.categoria_oficial.tolist(), base.id.tolist())
        st.caption('A projeção 2D pode distorcer distâncias. A categoria serve apenas para colorir; não é entrada do modelo. Compare também similaridades no espaço original.')
    with abas[3]:
        texto = st.text_area('Manifestação longa', base.iloc[base.texto.str.len().argmax()].texto, height=180)
        estrategia = st.selectbox('Estratégia', ['Recursiva', 'Caracteres fixos'])
        tamanho = st.slider('chunk_size em caracteres', 50, 1000, 400, 10)
        overlap = st.slider('chunk_overlap máximo em caracteres', 0, tamanho-1, min(80,tamanho-1))
        if texto.strip():
            chunks = dividir(texto, tamanho, overlap, estrategia)
            st.dataframe(pd.DataFrame(chunks), hide_index=True, width='stretch')
            try:
                ce = embeddings_em_cache(nome, tuple(c['texto'] for c in chunks))
                with st.expander('Embeddings completos'):
                    st.dataframe(pd.DataFrame(ce), width='stretch')
                st.download_button('Baixar embeddings CSV',pd.DataFrame(ce).to_csv(index=False).encode(), 'embeddings_chunks.csv')
                if len(ce) > 1:
                    grafico_pontos(projecao_em_cache(ce,'PCA'), ['Manifestação']*len(ce), [str(c['chunk']) for c in chunks])
                    st.write('Cossenos entre vizinhos:', [round(float(ce[i]@ce[i+1]),3) for i in range(len(ce)-1)])
                else:
                    st.info('Apenas um chunk: aumente o texto ou reduza chunk_size para comparar vizinhos.')
            except Exception as exc:
                st.error(f'Erro ao gerar embeddings dos chunks: {exc}')
        st.caption('O overlap solicitado é um limite; separadores podem produzir sobreposição efetiva menor. Textos acima do limite de tokens usam média de janelas normalizada.')

if __name__ == '__main__':
    main()
