# Estado dos projetos e instruções para outra IA

## Solicitação e decisões confirmadas

O usuário pediu três projetos independentes seguindo os enunciados: AgroSearch,
HealthSearch e Ouvidoria Inteligente. A autoria é individual: Fernando Amorim
Pontes Neto. A entrega foi organizada em aplicação .py, relatório .pdf e pacote
.zip por projeto. O usuário priorizou economizar créditos e pediu material
suficiente para continuar com outra IA, caso necessário.

A entrega será pelo GitHub, em três repositórios:
- https://github.com/fernandoaneto2/AgroSearch
- https://github.com/fernandoaneto2/HealthSearch
- https://github.com/fernandoaneto2/ouvidoria

Nenhum commit ou push foi realizado. O guia COMO_PUBLICAR_NO_GITHUB.md contém
os comandos personalizados. Não publicar automaticamente sem solicitação.

## Estado real

### AgroSearch
- Aplicação completa em um arquivo, com corpus do enunciado.
- Normalização, tokenização, stopwords e stemming configuráveis.
- Índice invertido e TF-IDF implementados do zero; NLTK só fornece o stemmer.
- Bônus de cosseno implementado.
- Quatro testes de lógica passaram na execução anterior.
- Testes de interface da consulta, controles, cosseno e entradas vazias passaram.
- Relatório de duas páginas criado e revisado visualmente.

### HealthSearch
- Aplicação completa em um arquivo; seis documentos originais hardcoded.
- BM25 com k1 entre 0 e 3, b entre 0 e 1; embeddings multilíngues reais.
- RRF ponderada com alpha entre 0 e 1, constante 60, posições iniciadas em 1.
- Corrigido o caso k1=0: rank_bm25 0.2.2 gera 0/0 em termos ausentes; o código
  usa o limite correto, IDF por presença, nesse extremo.
- Quatro testes de lógica passaram; telas principais, consultas e sliders passaram.
- Resultados reais foram obtidos com paraphrase-multilingual-MiniLM-L12-v2.
- Relatório de duas páginas com gráfico de ranks, revisado visualmente.
- Bônus Cross-Encoder implementado, mas ainda NÃO testado com pesos reais.

### Ouvidoria Inteligente
- app_ouvidoria.py e ouvidoria_core.py implementados.
- Cinco testes de lógica passaram: pares, métricas, chunking, projeção mínima e JSON.
- Primeiro carregamento do app, modelo MiniLM e funções principais foram exercitados.
- A bateria completa de interface NÃO foi concluída. Houve um erro no teste ao
  selecionar widget por índice; o roteiro foi corrigido para selecionar por rótulo.
- Três notebooks foram criados, mas estão SEM SAÍDAS EXECUTADAS.
- O primeiro notebook falhou durante o download de DistilUSE; não há tabelas finais,
  heatmap ou métricas experimentais finais de duplicatas e chunking.
- RELATORIO.pdf é um relatório parcial de implementação e pendências. Deve ser
  substituído por um relatório de resultados, com até cinco páginas, após executar.

## Base sintética autorizada

O manifestacoes.json original não foi fornecido. O usuário consultou os materiais
que possuía e autorizou explicitamente criar uma base no mesmo esquema exigido.
Não voltar a pedir essa autorização.

A base criada contém exatamente os campos id, data, categoria_oficial e texto,
40 registros M001 a M040, cinco categorias, textos de 50 a 800 caracteres e cinco
textos acima de 500 caracteres: M010, M016, M024, M033 e M040.

Gabarito exaustivo: M003-M017, M008-M022 e M005-M026. São três pares disjuntos,
abrangendo seis registros (15%); contar só as cópias adicionais daria 7,5%.
PROVENIENCIA.json esclarece essa convenção. Não apresentar a base como real ou
como sendo a base original do professor. Categoria semelhante não implica duplicata.

## Como concluir a Ouvidoria

1. Trabalhar fora de uma pasta sincronizada pelo iCloud para evitar arquivos
   disponíveis apenas online. Usar Python 3.12 e criar um ambiente novo.
2. Instalar requirements.txt da Ouvidoria. Não reutilizar o .venv antigo do
   workspace se houver arquivos com o indicador macOS dataless.
3. Baixar os dois modelos públicos usados em ouvidoria_core.py. O download via
   Xet do segundo modelo falhou por conexão. Tentar HF_HUB_DISABLE_XET=1 antes
   de importar huggingface_hub/sentence_transformers. Não substituir por scores
   simulados nem inventar resultados. A comparação de dois modelos é recomendada;
   os objetivos centrais exigem embeddings reais neste projeto.
4. Executar análise_comparativa.ipynb, deteccao_duplicatas.ipynb e
   chunking_manifestacoes.ipynb, nesta ordem e a partir da pasta do projeto.
   O runner em apoio_continuidade/executar_notebooks.py preserva a estrutura
   ../projetos; pode ser ajustado ou substituído por execução normal no Jupyter.
5. Inspecionar os resultados reais e escrever conclusões específicas para os
   pares, falsos positivos/negativos e fronteiras dos chunks. Os notebooks já
   geram CSVs, figuras e resumos JSON em resultados/.
6. Usar apoio_continuidade/relatorio_ouvidoria.py e gerar_relatorios.py como
   ponto de partida para o relatório final. Os builders esperam a estrutura
   de diretórios desta entrega e podem exigir ajustar ROOT. Usam ReportLab.
7. Executar os testes locais e o roteiro de interface corrigido. Cobrir troca de
   modelo, top-k, PCA/t-SNE, matriz, consulta vazia e um único chunk.
8. Renderizar e revisar todas as páginas do PDF. Manter no máximo cinco páginas.
9. Se desejar validar o bônus do HealthSearch, baixar o Cross-Encoder e confirmar
   reordenação dos Top-3 sem confundir seus scores com os scores RRF.

## Cuidados de implementação

- AgroSearch não pode usar scikit-learn/TfidfVectorizer para índice ou TF-IDF.
- TF relativo e IDF suavizada estão explicitados no relatório.
- HealthSearch usa rankings completos e desempates por ID. Quando BM25 é todo
  zero, a RRF ainda recebe posições arbitradas; a interface informa a limitação.
- O corpus médico é exatamente o do enunciado, não orientação clínica validada.
- Ouvidoria usa janelas de tokens com média normalizada para evitar truncamento
  silencioso de textos longos. O chunking experimental usa caracteres.
- 0,85 é um limiar exploratório; excluir diagonal e contar apenas i<j.
- Percentil 90 de pares não equivale a 15% de registros duplicados.
- PCA/t-SNE são diagnósticos; medir também cossenos no espaço original.
- Os notebooks de apoio de e-commerce não são um quarto projeto a executar.

## Materiais originais fornecidos

Os enunciados estão na pasta Downloads do usuário:
- Desafio_Lab_AgroSearch.docx
- Desafio_Ouvidoria_Inteligente.docx
- Laboratório Prático 05 - Desafio Integrador HealthSearch (Motor de Busca Híbrido BM25 e Semântico).docx

Materiais de apoio disponíveis na mesma pasta:
- app_chunks_embeddings.py
- app_faq_semantico.py
- lab_representacoes.ipynb
- laborat_ecommerce_produtos.ipynb

## Incidentes e transparência

Um bloqueio temporário da revisão automática de aprovação ocorreu por limite de
uso; não foi uma rejeição por segurança. Após retomada, autorizações voltaram a
funcionar. O principal obstáculo local foi o iCloud remover o conteúdo de arquivos
para deixá-los disponíveis online, bloqueando leituras. O código desta entrega foi
recuperado do histórico desta própria tarefa e as correções confirmadas reaplicadas.
Os relatórios AgroSearch/HealthSearch preservam resultados já observados; suas
figuras podem ter sido regeneradas a partir das posições registradas. Os resumos
numéricos no apoio não substituem novos experimentos e indicam arredondamento.

Não afirmar que tudo foi executado ou que a Ouvidoria está concluída. A entrega
atual é suficiente para continuar sem recomeçar a implementação.
