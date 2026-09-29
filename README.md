# Ouvidoria Inteligente
Autor: Fernando Amorim Pontes Neto. Trabalho individual.

## Execução
Requer Python 3.12. Na pasta deste projeto:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app_ouvidoria.py
```

No Windows, ative com `.venv\Scripts\activate`.
O primeiro uso de cada modelo requer internet para baixar pesos; a inferência é local.
A aplicação aceita upload JSON ou usa manifestacoes.json na mesma pasta.
Os quatro campos exigidos são id, data (AAAA-MM-DD), categoria_oficial e texto.
Categorias aceitas: infraestrutura, saúde, segurança, educação e meio ambiente.

## Notebooks
Abra e execute todos, na ordem abaixo, com o ambiente acima selecionado no editor:
1. análise_comparativa.ipynb — BoW, TF-IDF, dois modelos, pares e PCA/t-SNE.
2. deteccao_duplicatas.ipynb — heatmap, candidatos, limiares, FP e FN.
3. chunking_manifestacoes.ipynb — cinco textos mais longos, três configurações e PCA.

Os notebooks importam ouvidoria_core.py e gravam resultados na pasta resultados.
O gabarito duplicatas_reais.json é uma lista exaustiva de pares: [["M003","M017"],...].
Sem gabarito, precisão e recall não podem ser calculados honestamente.
A ausência do arquivo de dados não é suprida por uma base invisível no aplicativo.

## Modelos e análise
A interface seleciona MiniLM multilíngue ou DistilUSE multilíngue.
Embeddings são normalizados e textos acima do limite do modelo são representados
pela média normalizada de janelas de tokens. Não se usa categoria como entrada.
O limiar 0,85 é conservador e exploratório; similaridade temática não comprova
mesmo evento. O percentil 90 exclui diagonal e pares repetidos.
RecursiveCharacterTextSplitter usa caracteres, não tokens. Overlap efetivo pode
ser menor que o solicitado. As projeções 2D não preservam todas as distâncias.

## Materiais de apoio
Os quatro exemplos enviados foram consultados como referência didática.
As tarefas de e-commerce presentes no notebook de apoio não fazem parte deste
projeto. Corrigiu-se o uso de stop_words="portuguese", não aceito pelo scikit-learn,
e o caso de apenas um chunk nas projeções.

## Estado da entrega
A base é sintética, autorizada pelo solicitante, no esquema do enunciado: 40 registros, cinco textos
longos e três pares de duplicatas (ver PROVENIENCIA.json). Os três notebooks foram executados de ponta a
ponta com os dois modelos reais e gravam CSVs, figuras e resumos JSON em `resultados/`. Com o limiar
0,85 (MiniLM), a detecção de duplicatas obteve 2 verdadeiros positivos, 0 falsos positivos e 1 falso
negativo (precisão 1,0; recall 0,667; F1 0,8). RELATORIO.pdf apresenta esses resultados, a análise de
sensibilidade do limiar e a comparação das três configurações de chunking.

## Testes
Execute `python testes.py`. Os testes de lógica não baixam modelos.

## GitHub
Este diretório está versionado com código, relatório, dados, notebooks executados e a pasta `resultados/`.
Ambientes virtuais e pesos de modelos não são versionados (ver `.gitignore`).
