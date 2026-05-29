# LABORATÓRIO 10: O Pipeline Definitivo (RAG, QLoRA e Otimização de Inferência na GPU)

## Objetivo

Este laboratório integra os conceitos de:
- recuperação de contexto massivo via RAG,
- ajuste fino eficiente com QLoRA 4-bit,
- otimização de inferência na GPU usando FlashAttention-2,
- e uso de KV Cache para geração autogressiva.

O objetivo é demonstrar um fluxo de produção que processa grandes volumes de texto médico e gera resumos clínicos sem travar por falta de VRAM.

## Contexto do Problema

A HealthTech precisa que o sistema leia capítulos inteiros de manuais médicos (simulação de 10.000 a 15.000 tokens) e gere relatórios clínicos automáticos. A solução deve evitar o erro de Out-Of-Memory causado pela complexidade O(n²) do mecanismo de Self-Attention do Transformer.

## Metodologia

### Passo 1: Ingestão eficiente com QLoRA 4-bit

O modelo base é carregado usando a biblioteca `bitsandbytes` em quantização 4-bit, com `load_in_4bit=True` e `bnb_4bit_compute_dtype=torch.float16`. Isso reduz drasticamente a memória de parâmetros do modelo antes mesmo da inferência, liberando VRAM para o contexto extenso.

### Passo 2: Simulação de RAG massivo

Um texto sintético de ~12.000 tokens é gerado para simular os capítulos médicos recuperados por um banco vetorial. O texto é tokenizado com `AutoTokenizer` e movido para o dispositivo da GPU para inferência.

### Passo 3: Benchmark baseline sem cache

O modelo é executado com `model.config.use_cache = False` e geração de 100 novos tokens. Nesse cenário, o modelo recalcula Q, K e V em cada passo de token, o que aumenta tempo e pico de VRAM e evidencia o gargalo do decoder.

### Passo 4: Otimização com KV Cache e FlashAttention-2

O modelo é recarregado com `attn_implementation="flash_attention_2"` e `use_cache = True`. O uso de KV Cache evita o recálculo redundante de estados de atenção durante a geração, e o FlashAttention-2 melhora o desempenho na GPU usando algoritmos hardware-aware.

## Resultados Esperados

Os principais indicadores de benchmark devem ser:
- tempo total de geração de 100 tokens,
- pico de memória VRAM durante inferência,
- redução significativa em comparação com o baseline.

## Análise Técnica

### Parte A: Como QLoRA, KV Cache e FlashAttention salvaram o Transformer
A combinação de QLoRA, KV Cache e FlashAttention evita que o Transformer tradicional exploda a VRAM. QLoRA reduz a memória ocupada pelos pesos do modelo ao quantizar para 4 bits, liberando espaço para o contexto enorme. O KV Cache evita recalcular as matrizes Q, K e V para tokens já processados durante a geração autogressiva, transformando o bottleneck do decoder de uma operação O(n²) em algo próximo de O(n) nos passos sucessivos. Por fim, o FlashAttention-2 aproveita as unidades de memória de alta largura de banda da GPU para calcular a atenção sem materializar matrizes inteiras, diminuindo ainda mais o consumo de memória e acelerando a execução.

### Parte B: Por que 2 milhões de tokens exigiriam State Space Models
Mesmo com essas otimizações, o Transformer ainda mantém complexidade quadrática em relação ao número de tokens para a fase de atenção global. Se o cliente requer processamento de 2 milhões de tokens, a quantidade de memória necessária para armazenar chaves e valores e realizar o produto de atenção torna-se impossível em GPUs atuais. Nessa escala, até o FlashAttention falharia porque o gargalo é fundamentalmente a natureza do self-attention. A indústria então precisa migrar para arquiteturas como State Space Models, por exemplo Mamba, que possuem complexidade de memória muito mais favorável, tipicamente O(1) ou O(n) com pequeno coeficiente, permitindo longas sequências sem explodir a VRAM.

## Uso

1. Executar o notebook `lab10_rag_qloRA_flashattention.ipynb`.
2. Verificar geração de `README.md`, `benchmarks.json` e `generated_clinical_summary.txt`.
3. Criar commit e tag `v1.0` para entrega no GitHub.
