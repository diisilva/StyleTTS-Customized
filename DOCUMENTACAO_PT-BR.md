# StyleTTS — Documentação Completa (Português BR)

> Autores originais: Yinghao Aaron Li, Cong Han, Nima Mesgarani
> Artigo: [https://arxiv.org/abs/2205.15439](https://arxiv.org/abs/2205.15439)
> Amostras de áudio: [https://styletts.github.io/](https://styletts.github.io/)

---
## Antes de  tudo 

Execute: pip install -r requirements.txt

estou a versao do Python 3.13.9 

### Teste com amostra menor dos dados

Para validar o pipeline antes do treino completo, criei listas reduzidas:

```powershell
Get-Content Data\train_list.txt | Select-Object -First 1000 | Set-Content Data\train_list_mini.txt
Get-Content Data\val_list.txt   | Select-Object -First 200  | Set-Content Data\val_list_mini.txt
```

Em seguida, no `Configs/config.yml`, alterei temporariamente:

```yaml
train_data: "Data/train_list_mini.txt"
val_data:   "Data/val_list_mini.txt"
epochs_1st: 5
epochs_2nd: 3
save_freq:  1
```

E executei normalmente (aqui agora vc pode passar via parametro):


## Training
First stage training:
```bash
# GPU (recommended)
python train_first.py --config_path ./Configs/config.yml --device cuda

# CPU
python train_first.py --config_path ./Configs/config.yml --device cpu
```
Second stage training:
```bash
# GPU (recommended)
python train_second.py --config_path ./Configs/config.yml --device cuda

# CPU
python train_second.py --config_path ./Configs/config.yml --device cpu
```

Com 1000 amostras e `batch_size: 16` → ~63 steps por época. As 5 épocas rodam em poucos minutos e permitem verificar se não há `CUDA out of memory`, se as perdas aparecem corretamente no log e se os checkpoints são salvos em `Models/LJSpeech/`. Após confirmar, basta restaurar os valores originais no `config.yml` para o treino completo.

### Monitorar o treino com TensorBoard

Enquanto o treino roda (ou após), abra o TensorBoard. Se `tensorboard` não for reconhecido no terminal, instale primeiro:

```powershell
pip install tensorboard
```

Depois inicie:
```powershell
# Opção 1 — direto (se PATH atualizado)
tensorboard --logdir Models/LJSpeech/tensorboard

# Opção 2 — via Python (sempre funciona)
python -m tensorboard.main --logdir Models/LJSpeech/tensorboard
```

Acesse **http://localhost:6006** no navegador.

**O que observar nas métricas:**

| Métrica | Comportamento saudável | O que significa |
|---|---|---|
| `mel_loss` | Caindo progressivamente (~0.7 → 0.2–0.3 ao final) | Principal indicador — modelo aprendendo a reconstruir o áudio |
| `adv_loss` | Estável ~0.69 | GAN equilibrado — gerador e discriminador empatados |
| `d_loss` | Estável ~1.38 | Discriminador saudável (≈ 2 × log(2)) |
| `mono_loss` | Zero até época 20, depois cai | Normal — TMA só ativa em `TMA_epoch: 20` |
| `s2s_loss` | Idem ao `mono_loss` | Alinhamento S2S, ativa junto com o TMA |

> **Alerta:** se qualquer perda virar `NaN` ou subir indefinidamente acima de 100, o treino está instável. Reduza o `batch_size` ou o `lr` no `config.yml`.

### Otimizações de Performance no Windows

Três correções críticas aplicadas a este repositório. Sem elas, cada época pode levar horas em vez de minutos no Windows:

| Arquivo | Problema | Correção |
|---|---|---|
| `utils.py` | `get_image()` criava figuras matplotlib sem `plt.close(fig)` → vazamento de RAM a cada época | Adicionado `plt.close(fig)` antes do `return` |
| `meldataset.py` | `DataLoader` sem `persistent_workers=True` → 4 workers destruídos e recriados do zero a cada época (Windows usa `spawn`, não `fork`) | Adicionado `persistent_workers=(num_workers > 0)` |
| `train_first.py` / `train_second.py` | `num_workers=8` excessivo no Windows — overhead de spawn supera o ganho | Reduzido para `num_workers=4` |

> **Por que isso importa no Windows:** diferente do Linux (que usa `fork` para clonar processos instantaneamente), o Windows precisa inicializar cada worker do zero com `spawn`. Com 8 workers e sem `persistent_workers`, o overhead de reinicialização por época chegava a dezenas de minutos — mais tempo do que o próprio treinamento.

### Desativar o sleep do Windows durante o treino

O treino completo leva horas. Se o Windows suspender o PC no meio (modo sleep/hibernação), o processo Python é interrompido e o progresso da época atual é perdido. Isso ocorreu em testes reais: o step 10 da época 2 ficou parado das 01:06 até as 09:31 — um gap de 8h24min — porque o PC dormiu.

Antes de iniciar o treino, execute no PowerShell **como Administrador**:

```powershell
# Desativa o sleep quando conectado à tomada (0 = nunca dormir)
powercfg /change standby-timeout-ac 0
```

Após o treino terminar, restaure o valor padrão:

```powershell
# Restaura sleep para 30 minutos (padrão Windows)
powercfg /change standby-timeout-ac 30
```

> **Por que só `ac`?** O parâmetro `-ac` (alternating current) afeta o comportamento conectado à tomada. Se quiser desativar também na bateria, use `-dc` com o mesmo valor — mas não é recomendado em notebooks, pois pode descarregar a bateria se o treino rodar sem supervisão.

---

### Após o Estágio 2 terminar

Os checkpoints ficam em `Models/LJSpeech/epoch_2nd_XXXXX.pth`. Abra o TensorBoard e identifique o checkpoint com o menor `mel_loss` de validação — **não necessariamente o último** é o melhor.

Baixe o vocoder HiFi-GAN pré-treinado (sem ele não há geração de áudio, pois os checkpoints do StyleTTS produzem apenas mel-espectrogramas):
- [HiFi-GAN LJSpeech](https://huggingface.co/yl4579/StyleTTS/blob/main/LJSpeech/Vocoder.zip) → descompactar em `Vocoder/`

Depois abra `Demo/Inference_LJSpeech.ipynb`, aponte para o seu `epoch_2nd_XXXXX.pth` com o menor `mel_loss` e execute. É aí que você ouvirá pela primeira vez se o modelo aprendeu algo.

---

## O que é o StyleTTS?

O **StyleTTS** é um modelo generativo de síntese de voz (Text-to-Speech — TTS) baseado em **estilo**. Ele é capaz de sintetizar fala com prosódianatural e diversificada a partir de um áudio de referência, imitando o estilo de fala, o tom emocional e as variações prosódicas sem precisar de rótulos explícitos para essas categorias.

### Principais características

- **Síntese paralela de alta qualidade**: gera áudio sem precisar executar a síntese passo a passo (como modelos autoregressivos), sendo muito mais rápido.
- **Transferência de estilo**: através de aprendizado auto-supervisionado, o modelo aprende a copiar o estilo de qualquer áudio de referência — prosódia, emoção, ritmo — automaticamente.
- **TMA (Transferable Monotonic Aligner)**: alinhador monotônico transferível que resolve um dos maiores problemas de TTS paralelo: encontrar o alinhamento monotônico correto entre texto e duração da fala.
- **Augmentação invariante de duração**: técnica de aumento de dados que garante robustez ao modelo durante o treinamento.
- **Suporte a dataset de um locutor (LJSpeech) e multi-locutor (LibriTTS)**.

---

## Estrutura do Repositório

```
StyleTTS/
│
├── train_first.py          # Script de treinamento — Estágio 1
├── train_second.py         # Script de treinamento — Estágio 2
├── models.py               # Definição de todas as arquiteturas de rede neural
├── meldataset.py           # Pré-processamento de áudio e pipeline de dados
├── utils.py                # Funções auxiliares (alinhamento, perdas, utilidades)
├── optimizers.py           # Definição dos otimizadores e schedulers
├── text_utils.py           # Utilitários de texto e mapeamento de fonemas
│
├── Configs/
│   └── config.yml          # Arquivo de configuração central do treinamento
│
├── Data/
│   ├── train_list.txt           # Lista de dados de treinamento (LJSpeech)
│   ├── val_list.txt             # Lista de dados de validação (LJSpeech)
│   ├── train_list_libritts.txt  # Lista de dados de treinamento (LibriTTS)
│   └── val_list_libritts.txt    # Lista de dados de validação (LibriTTS)
│
├── Demo/
│   ├── Inference_LJSpeech.ipynb     # Notebook de inferência para LJSpeech
│   ├── Inference_LibriTTS.ipynb     # Notebook de inferência para LibriTTS
│   └── hifi-gan/
│       ├── vocoder.py               # Arquitetura do vocoder HiFi-GAN
│       └── vocoder_utils.py         # Utilitários do vocoder
│
└── Utils/
    ├── ASR/
    │   ├── config.yml       # Configuração do modelo de alinhamento de texto (ASR)
    │   ├── epoch_00080.pth  # Pesos pré-treinados do alinhador de texto
    │   ├── layers.py        # Camadas das redes do modelo ASR
    │   └── models.py        # Arquitetura ASRCNN (alinhador de texto)
    └── JDC/
        ├── bst.t7           # Pesos pré-treinados do extrator de F0 (pitch)
        └── model.py         # Arquitetura JDCNet (extrator de frequência fundamental)
```

---

## Descrição Detalhada de Cada Arquivo

---

### `train_first.py` — Treinamento do Estágio 1

**Para que serve:** Treina a espinha dorsal do StyleTTS — o alinhador de texto, o codificador de texto, o codificador de estilo e o decodificador de mel-espectrograma. É o ponto de entrada obrigatório antes do estágio 2.

**O que faz internamente:**
- Carrega o arquivo de configuração (`config.yml`).
- Inicializa logs no TensorBoard e em arquivo de texto (`train.log`).
- Monta os dataloaders de treino e validação usando `meldataset.py`.
- Carrega os modelos auxiliares pré-treinados:
  - **Alinhador de texto (ASR):** `Utils/ASR/epoch_00080.pth`
  - **Extrator de F0 (pitch):** `Utils/JDC/bst.t7`
- Constrói o modelo completo via `build_model()` de `models.py`.
- Usa o **TMA (Transferable Monotonic Aligner)** para aprender o alinhamento texto-áudio de forma monotônica.
- Durante o loop de treinamento:
  - Passa os batches pelo alinhador ASR para obter as atenções texto-mel.
  - Computa o caminho monotônico com `maximum_path()`.
  - Calcula perdas de reconstrução de mel, adversarial, regularização, feature matching, alinhamento S2S e alinhamento monotônico.
  - Atualiza os parâmetros com o otimizador `AdamW` via `MultiOptimizer`.
- Salva checkpoints periodicamente como `epoch_1st_XXXXX.pth`.

**Como executar:**
```bash
# GPU (recomendado — RTX 4050 ou superior)
python train_first.py --config_path ./Configs/config.yml --device cuda

# CPU (lento, apenas para testes)
python train_first.py --config_path ./Configs/config.yml --device cpu
```

O parâmetro `--device` sobrescreve o campo `device` do `config.yml`. Valores aceitos: `cuda`, `cuda:0`, `cuda:1`, `cpu`. Se omitido, usa o valor do `config.yml` (padrão: `cuda`).

**Onde os resultados vão:** pasta definida em `log_dir` no `config.yml` (padrão: `Models/LJSpeech/`).

---

### `train_second.py` — Treinamento do Estágio 2

**Para que serve:** Treina o **Preditor de Prosódia** e o **Discriminador**, que são responsáveis por gerar durações, F0 (pitch/entonação) e energia a partir do estilo aprendido. Depende do resultado do estágio 1.

**O que faz internamente:**
- Carrega os pesos do estágio 1 a partir do arquivo `first_stage_path` definido no `config.yml`.
- Congela todos os módulos do estágio 1 (apenas `predictor` e `discriminator` são treinados).
- Usa **augmentação de tempo** (`TimeStrech`) para criar variações de duração durante o treinamento — é a "augmentação invariante de duração" descrita no artigo.
- Calcula perdas adicionais em relação ao estágio 1:
  - **Perda de F0 (pitch):** reconstrução da frequência fundamental.
  - **Perda de norma:** consistência de energia.
  - **Perda de duração:** alinhamento correto das durações de fonema.
- Salva checkpoints como `epoch_2nd_XXXXX.pth`.

**Como executar:**
```bash
# GPU (recomendado)
python train_second.py --config_path ./Configs/config.yml --device cuda

# CPU
python train_second.py --config_path ./Configs/config.yml --device cpu
```

**Onde os resultados vão:** mesmo `log_dir` do estágio 1.

---

### `models.py` — Arquiteturas das Redes Neurais

**Para que serve:** Define toda a arquitetura do StyleTTS. Contém todos os módulos PyTorch do sistema.

**Componentes principais:**

| Classe | Descrição |
|---|---|
| `StyleEncoder` | Codifica um mel-espectrograma de referência em um vetor de estilo de 128 dimensões. Usa blocos residuais com downsampling. |
| `TextEncoder` | Codifica a sequência de fonemas (texto) em representações contextuais usando convoluções 1D e normalização de camada. |
| `Decoder` | Decodifica as representações de texto com o vetor de estilo para gerar um mel-espectrograma. Usa blocos AdaIN residuais (Adaptive Instance Normalization). |
| `ProsodyPredictor` | Prediz F0, energia e duração a partir do estilo e do texto no estágio 2. Usa LSTMs bidirecionais com AdaLayerNorm. |
| `DurationEncoder` | Codifica informação de duração no contexto do estilo usando LSTMs bidirecionais. |
| `Discriminator2d` | Discriminador 2D para treinamento adversarial, avalia a qualidade do mel-espectrograma gerado. |
| `ResBlk` / `ResBlk1d` | Blocos residuais para downsampling/upsampling em 2D e 1D. |
| `AdaIN1d` / `AdaLayerNorm` | Normalização adaptativa que injeta o vetor de estilo nas camadas do decodificador e preditor. |
| `load_F0_models()` | Carrega o modelo JDCNet pré-treinado para extração de F0. |
| `load_ASR_models()` | Carrega o modelo ASRCNN pré-treinado para alinhamento de texto. |
| `build_model()` | Monta todos os componentes em um dicionário `Munch` usado durante o treinamento. |
| `load_checkpoint()` | Carrega pesos de um checkpoint salvo. |

---

### `meldataset.py` — Dataset e Pré-processamento de Áudio

**Para que serve:** Responsável por ler os arquivos de áudio e texto, converter áudio em mel-espectrograma, e fornecer batches ao treinamento.

**O que faz internamente:**
- Define o vocabulário de símbolos: pontuação, letras e fonemas IPA (International Phonetic Alphabet).
- `TextCleaner`: converte texto/fonemas em sequências de índices inteiros.
- `preprocess()`: converte waveform em mel-espectrograma normalizado (log-mel, normalizado com média -4 e desvio padrão 4).
- `FilePathDataset`: implementa o `Dataset` do PyTorch. Lê arquivos `.wav` com `soundfile`, reamostre se necessário, extrai mel-espectrograma, e retorna `(texto, comprimento_texto, mel, comprimento_mel)`.
- `build_dataloader()`: cria `DataLoader` do PyTorch com `persistent_workers=True` (mantém os workers vivos entre épocas — essencial no Windows) e `pin_memory=True` para GPU.

**Parâmetros de mel-espectrograma:**
- Sample rate: 24.000 Hz
- n_fft: 2048
- win_length: 1200
- hop_length: 300
- n_mels: 80

**Formato dos dados de entrada:**
```
caminho/para/arquivo.wav|transcrição em fonemas IPA|id_locutor
```
Exemplo real do dataset:
```
LJSpeech-1.1/wavs/LJ050-0234.wav|ɪt hɐz jˈuːzd ˈʌðɚ tɹˈɛʒɚɹi...|0
```

---

### `utils.py` — Funções Auxiliares

**Para que serve:** Agrupa funções utilitárias usadas nos scripts de treinamento.

**Funções principais:**

| Função | Descrição |
|---|---|
| `maximum_path(neg_cent, mask)` | Calcula o caminho monotônico ótimo entre texto e mel usando a implementação Cython de `monotonic_align`. Usado no TMA. |
| `get_data_path_list(train_path, val_path)` | Lê os arquivos `.txt` de lista de dados e retorna listas de linhas. |
| `length_to_mask(lengths)` | Converte comprimentos de sequência em máscaras booleanas para atenção com padding. |
| `adv_loss(logits, target)` | Calcula a perda adversarial (binary cross-entropy com logits, com clamp para evitar NaN). |
| `r1_reg(d_out, x_in)` | Regularização R1 (penalidade de gradiente zero-centrada) para o discriminador. |
| `log_norm(x)` | Calcula a norma logarítmica do mel-espectrograma para a perda de norma do estágio 2. |
| `get_image(arrs)` | Gera figura matplotlib de uma matriz 2D (usada para visualização no TensorBoard). Chama `plt.close(fig)` após uso para evitar vazamento de memória. |

---

### `optimizers.py` — Otimizadores e Schedulers

**Para que serve:** Define e gerencia múltiplos otimizadores simultaneamente (um por módulo do modelo).

**Componentes:**

| Classe/Função | Descrição |
|---|---|
| `MultiOptimizer` | Gerencia um dicionário de otimizadores `AdamW`, um para cada sub-módulo do modelo (text_encoder, decoder, predictor, etc.). Suporta `step()`, `zero_grad()` e `scheduler()` para todos de uma vez ou individualmente. |
| `define_scheduler(optimizer, params)` | Cria um `OneCycleLR` scheduler com os parâmetros fornecidos. |
| `build_optimizer(parameters_dict, scheduler_params_dict)` | Instancia um `AdamW` por módulo (lr=1e-4, weight_decay=1e-4, betas=(0.0, 0.99)) e retorna um `MultiOptimizer`. |

**Configurações do `AdamW`:**
- `lr`: 1e-4 (taxa de aprendizado)
- `weight_decay`: 1e-4
- `betas`: (0.0, 0.99) — sem momento de primeira ordem, comum em GANs
- `eps`: 1e-9

---

### `text_utils.py` — Utilitários de Texto

**Para que serve:** Define o vocabulário de símbolos fonéticos e a classe de limpeza de texto.

**O que contém:**
- `_pad`: símbolo de padding (`$`).
- `_punctuation`: pontuação suportada.
- `_letters`: letras maiúsculas e minúsculas do alfabeto latino.
- `_letters_ipa`: conjunto completo de símbolos IPA (International Phonetic Alphabet) para representar qualquer som da fala humana.
- `symbols`: lista completa de todos os símbolos suportados (178 no total).
- `dicts`: dicionário mapeando cada símbolo ao seu índice inteiro.
- `TextCleaner`: converte uma string de text/fonemas em lista de índices inteiros para alimentar o modelo.

> **Nota:** Este arquivo é a versão independente de `text_utils.py`, usada pelos notebooks de inferência. A mesma lógica está incorporada dentro de `meldataset.py` para o treinamento.

---

## Pasta `Utils/` — Modelos Auxiliares Pré-treinados

### `Utils/ASR/` — Alinhador de Texto (CTC + Attention)

**Para que serve:** Contém o modelo de reconhecimento de fala (ASR) que é usado como **alinhador de texto** no StyleTTS. Ele aprende a mapear mel-espectrogramas para sequências de fonemas, gerando matrizes de atenção que são o alinhamento texto-áudio utilizado pelo TMA.

#### `Utils/ASR/models.py`

| Classe | Descrição |
|---|---|
| `ASRCNN` | Rede convolucional ASR. Entrada: mel-espectrograma 80-dim. Processa com MFCC → CNN inicializador → blocos convolucionais → projeção → saída CTC. Quando o texto de entrada é fornecido, também computa atenção S2S via `ASRS2S`. |
| `ASRS2S` | Decodificador sequência-para-sequência com atenção de localização. Gera a matriz de atenção entre texto e mel usada pelo TMA. |

#### `Utils/ASR/layers.py`
Contém camadas reutilizáveis: `MFCC`, `Attention` (atenção baseada em localização), `LinearNorm`, `ConvNorm`, `ConvBlock`.

#### `Utils/ASR/config.yml`
Configuração do modelo ASR (dimensões, número de tokens, camadas, etc.).

#### `Utils/ASR/epoch_00080.pth`
Pesos pré-treinados do alinhador de texto. Treinado com mel-espectrogramas de 24 kHz. **Não deve ser modificado** a menos que você vá re-treinar o alinhador com seus próprios dados.

> Para treinar seu próprio alinhador: [https://github.com/yl4579/AuxiliaryASR](https://github.com/yl4579/AuxiliaryASR)

---

### `Utils/JDC/` — Extrator de Pitch/F0

**Para que serve:** Contém o modelo **JDCNet** (Joint Detection and Classification Network), usado para extrair a frequência fundamental (F0 / pitch) dos mel-espectrogramas durante o treinamento.

#### `Utils/JDC/model.py`

| Classe | Descrição |
|---|---|
| `JDCNet` | Rede convolucional-recorrente para detecção e classificação conjunta da melodia de voz. Usa blocos residuais CNN + BiLSTM. Retorna classe de F0 e se o frame é voz ou silêncio. Adaptada do artigo de Kum et al. (2019). |
| `ResBlock` | Bloco residual com duas convoluções e max-pooling, usado dentro do JDCNet. |

#### `Utils/JDC/bst.t7`
Pesos pré-treinados do extrator de F0. Formato `.t7` (legado do Torch7, carregado via PyTorch).

> Para treinar seu próprio extrator de F0: [https://github.com/yl4579/PitchExtractor](https://github.com/yl4579/PitchExtractor)

---

## Pasta `Demo/` — Inferência

### `Demo/Inference_LJSpeech.ipynb`

**Para que serve:** Notebook Jupyter para sintetizar fala usando o modelo treinado no dataset LJSpeech (locutor único).

**Fluxo de inferência:**
1. Carrega o modelo StyleTTS treinado e o vocoder HiFi-GAN.
2. Recebe um texto de entrada.
3. Converte o texto em fonemas IPA usando o `phonemizer`.
4. Usa um áudio de referência para extrair o vetor de estilo via `StyleEncoder`.
5. Passa os fonemas pelo `TextEncoder` → `Decoder` para gerar o mel-espectrograma.
6. Converte o mel-espectrograma em áudio usando o HiFi-GAN.

**Modelos pré-treinados necessários:**
- [StyleTTS LJSpeech](https://huggingface.co/yl4579/StyleTTS/blob/main/LJSpeech/Models.zip) → descompactar em `Models/`
- [HiFi-GAN LJSpeech](https://huggingface.co/yl4579/StyleTTS/blob/main/LJSpeech/Vocoder.zip) → descompactar em `Vocoder/`

---

### `Demo/Inference_LibriTTS.ipynb`

**Para que serve:** Notebook Jupyter para replicação de voz zero-shot (ZSL) no dataset LibriTTS (multi-locutor). Permite sintetizar fala com a voz de qualquer locutor do conjunto de teste, sem ter visto esse locutor durante o treinamento.

**Modelos pré-treinados necessários:**
- [StyleTTS LibriTTS](https://huggingface.co/yl4579/StyleTTS/blob/main/LibriTTS/Models.zip) → descompactar em `Models/`
- [HiFi-GAN LibriTTS](https://huggingface.co/yl4579/StyleTTS/blob/main/LibriTTS/Vocoder.zip) → descompactar em `Vocoder/`
- Baixar `test-clean` do LibriTTS para o zero-shot demo.

---

### `Demo/hifi-gan/vocoder.py` — Vocoder HiFi-GAN

**Para que serve:** Implementa a arquitetura do vocoder **HiFi-GAN (High Fidelity GAN)**, que converte mel-espectrogramas em forma de onda de áudio de alta fidelidade (24 kHz).

**Arquitetura:**
- `Generator`: rede totalmente convolucional com upsampling progressivo. Recebe um mel-espectrograma (80 bands) e gera amostras de áudio via transposição de convoluções + blocos residuais multi-escala.
- `ResBlock1` / `ResBlock2`: blocos residuais com dilações múltiplas para capturar padrões de diferentes resoluções temporais.
- `MultiScaleDiscriminator` / `MultiPeriodDiscriminator`: discriminadores do HiFi-GAN original (usados durante o treinamento do vocoder, não do StyleTTS em si).

---

### `Demo/hifi-gan/vocoder_utils.py` — Utilitários do Vocoder

**Para que serve:** Funções de suporte para o treinamento e carregamento do HiFi-GAN.

| Função | Descrição |
|---|---|
| `init_weights(m)` | Inicializa pesos convolucionais com distribuição normal. |
| `get_padding(kernel_size, dilation)` | Calcula o padding para manter o tamanho da sequência nas convoluções dilatadas. |
| `load_checkpoint(filepath, device)` | Carrega um checkpoint de arquivo `.pth` no dispositivo especificado. |
| `save_checkpoint(filepath, obj)` | Salva um objeto como checkpoint. |
| `scan_checkpoint(cp_dir, prefix)` | Encontra o checkpoint mais recente em um diretório. |
| `plot_spectrogram(spectrogram)` | Gera figura matplotlib do espectrograma (para visualização). |

---

## Pasta `Configs/`

### `Configs/config.yml` — Configuração Central

**Para que serve:** Arquivo YAML que controla todos os hiperparâmetros do treinamento. É lido por `train_first.py` e `train_second.py` via argumento `--config_path`.

**Seções e parâmetros:**

```yaml
# Diretórios e arquivos
log_dir: "Models/LJSpeech"         # Onde salvar checkpoints e logs
first_stage_path: "first_stage.pth" # Nome do checkpoint do estágio 1

# Frequência e controle
save_freq: 2        # Salvar checkpoint a cada N épocas
log_interval: 10    # Logar métricas a cada N iterações
device: "cuda"      # Dispositivo padrão (cuda ou cpu) — sobrescrito por --device na linha de comando
multigpu: false     # Suporte a múltiplas GPUs (DataParallel)

# Épocas (treino completo)
epochs_1st: 200     # Épocas do estágio 1
epochs_2nd: 100     # Épocas do estágio 2
batch_size: 16      # Tamanho do batch

# Dados (treino completo)
train_data: "Data/train_list.txt"
val_data: "Data/val_list.txt"
# Para mini-teste, descomente os blocos comentados no config.yml

# Modelos auxiliares
F0_path: "Utils/JDC/bst.t7"
ASR_config: "Utils/ASR/config.yml"
ASR_path: "Utils/ASR/epoch_00080.pth"

# Pré-processamento
preprocess_params:
  sr: 24000             # Sample rate
  spect_params:
    n_fft: 2048
    win_length: 1200
    hop_length: 300

# Modelo
model_params:
  hidden_dim: 512    # Dimensão oculta principal
  n_token: 178       # Tamanho do vocabulário fonético
  style_dim: 128     # Dimensão do vetor de estilo
  n_layer: 3         # Número de camadas
  dim_in: 64         # Dimensão de entrada do discriminador
  max_conv_dim: 512  # Dimensão máxima das convoluções
  n_mels: 80         # Número de bandas mel
  dropout: 0.2

# Pesos das perdas
loss_params:
  lambda_mel: 5.     # Reconstrução de mel (estágios 1 e 2)
  lambda_adv: 1.     # Perda adversarial GAN (estágios 1 e 2)
  lambda_reg: 1.     # Regularização adversarial R1 (estágios 1 e 2)
  lambda_fm: 0.1     # Feature matching (estágios 1 e 2)
  lambda_mono: 1.    # Alinhamento monotônico TMA (estágio 1)
  lambda_s2s: 1.     # Alinhamento S2S (estágio 1)
  TMA_epoch: 20      # Époa de início do TMA
  TMA_CEloss: false  # Usar cross-entropy no TMA (ver issue #7)
  lambda_F0: 1.      # Reconstrução de F0 (estágio 2)
  lambda_norm: 1.    # Reconstrução de norma/energia (estágio 2)
  lambda_dur: 1.     # Perda de duração (estágio 2)

# Otimizador
optimizer_params:
  lr: 0.0001
```

---

## Pasta `Data/` — Listas de Dados

### Formato dos arquivos de lista

Cada linha tem o formato:
```
caminho/para/audio.wav|transcrição_em_fonemas_IPA|id_locutor
```
- O `id_locutor` é `0` para datasets de locutor único (LJSpeech).
- Para LibriTTS o ID do locutor varia por locutor.

### Arquivos presentes

| Arquivo | Conteúdo |
|---|---|
| `train_list.txt` | Lista de treino do LJSpeech com fonemas IPA (locutor 0). |
| `val_list.txt` | Lista de validação do LJSpeech com fonemas IPA. |
| `train_list_libritts.txt` | Lista de treino do LibriTTS (multi-locutor). |
| `val_list_libritts.txt` | Lista de validação do LibriTTS. |

---

## Fluxo Completo: Do Dado ao Áudio

```
Texto de entrada (fonemas IPA)
        │
        ▼
  [TextEncoder]
  Representação contextual do texto
        │
        ├───── Alinhamento TMA ────► [text_aligner (ASRCNN)]
        │       (estágio 1)                   │
        │                                     ▼
        │                           Matriz de atenção
        │                           (alinhamento texto↔mel)
        │
        ├───── Vetor de estilo ────► [StyleEncoder]
        │                           (extraído do áudio de referência)
        │
        ▼
    [Decoder]
  Mel-espectrograma sintetizado
        │
        ├──── Estágio 2 ──────────► [ProsodyPredictor]
        │                           Predição de F0, norma, duração
        │
        ▼
  [HiFi-GAN Vocoder]
        │
        ▼
  Áudio de saída (waveform 24 kHz)
```

---

## Pré-requisitos e Instalação

### Requisitos de sistema
- Python >= 3.7
- GPU com CUDA (recomendado — o treinamento em CPU é muito lento)
- Espaço em disco para datasets (LJSpeech ~2.6 GB, LibriTTS train-460 ~50 GB)

### Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/yl4579/StyleTTS.git
cd StyleTTS

# 2. Instalar dependências
pip install SoundFile torchaudio munch torch pydub pyyaml librosa git+https://github.com/resemble-ai/monotonic_align.git

# 3. Para inferência, instalar também o phonemizer
pip install phonemizer
```

> **Atenção para o `monotonic_align`:** este pacote compila código Cython durante a instalação. É necessário ter um compilador C disponível (no Windows: Visual C++ Build Tools; no Linux: `gcc`).

---

## Como Treinar

### 1. Preparar os dados

- Baixar e extrair o [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).
- Upsample o áudio para 24 kHz.
- Os arquivos de lista (`Data/train_list.txt`, `Data/val_list.txt`) já estão prontos para o LJSpeech com fonemas IPA.

Para **LibriTTS**:
- Combinar `train-clean-360` e `train-clean-100` → renomear pasta para `train-clean-460`.
- Usar os arquivos `Data/train_list_libritts.txt` e `Data/val_list_libritts.txt`.

### 2. Estágio 1

```bash
# GPU (recomendado)
python train_first.py --config_path ./Configs/config.yml --device cuda

# CPU (somente para testes, muito lento)
python train_first.py --config_path ./Configs/config.yml --device cpu
```

- Treina por `epochs_1st` (padrão: 200) épocas.
- Salva checkpoints como: `Models/LJSpeech/epoch_1st_00002.pth`, `epoch_1st_00004.pth`, etc.
- Ao final, salva o melhor modelo como `first_stage.pth`.

### 3. Estágio 2

```bash
# GPU (recomendado)
python train_second.py --config_path ./Configs/config.yml --device cuda

# CPU
python train_second.py --config_path ./Configs/config.yml --device cpu
```

- Carrega automaticamente o `first_stage.pth` da pasta `log_dir`.
- Treina por `epochs_2nd` (padrão: 100) épocas.
- Salva checkpoints como: `Models/LJSpeech/epoch_2nd_00002.pth`, etc.

### 4. Monitorar com TensorBoard

```powershell
# Opção 1
tensorboard --logdir Models/LJSpeech/tensorboard

# Opção 2 (se tensorboard não for reconhecido no PATH)
python -m tensorboard.main --logdir Models/LJSpeech/tensorboard
```

Acesse **http://localhost:6006**. Métricas a acompanhar: `mel_loss` caindo, `adv_loss` e `d_loss` estáveis em ~0.69 e ~1.38 respectivamente. `mono_loss` e `s2s_loss` ficam em zero até a época 20 (`TMA_epoch: 20`) — isso é normal.

---

## Como Fazer Inferência

1. Baixar os modelos pré-treinados (links acima).
2. Descompactar em `Models/` e `Vocoder/`.
3. Instalar o `phonemizer`:
   ```bash
   pip install phonemizer
   ```
4. No TensorBoard, identificar o checkpoint `epoch_2nd_XXXXX.pth` com o menor `mel_loss` de validação — não necessariamente o último é o melhor.
5. Abrir `Demo/Inference_LJSpeech.ipynb`, apontar para esse checkpoint e executar célula a célula. É aí que você ouvirá pela primeira vez se o modelo aprendeu algo.
   - `Demo/Inference_LJSpeech.ipynb` para locutor único (LJSpeech).
   - `Demo/Inference_LibriTTS.ipynb` para zero-shot multi-locutor.

---

## Como Personalizar o Pré-processamento

O arquivo `meldataset.py` define como o áudio é convertido em mel-espectrograma. Se você mudar os parâmetros (sample rate, n_fft, hop_length, etc.):

1. Os modelos pré-treinados de alinhamento (`Utils/ASR/`) e extração de F0 (`Utils/JDC/`) **não funcionarão mais**.
2. Você precisará re-treinar:
   - O **alinhador de texto**: código em [https://github.com/yl4579/AuxiliaryASR](https://github.com/yl4579/AuxiliaryASR)
   - O **extrator de F0**: código em [https://github.com/yl4579/PitchExtractor](https://github.com/yl4579/PitchExtractor)

---

## Dependências Externas Utilizadas

| Biblioteca | Uso |
|---|---|
| `torch` / `torchaudio` | Framework de deep learning e processamento de áudio |
| `librosa` | Processamento de áudio adicional |
| `soundfile` | Leitura/escrita de arquivos de áudio `.wav` |
| `pyyaml` / `munch` | Leitura do config e acesso por atributo |
| `click` | Interface de linha de comando para os scripts de treino |
| `monotonic_align` | Cálculo eficiente do caminho monotônico (Cython) |
| `phonemizer` | Conversão de texto em fonemas IPA (somente inferência) |
| `tensorboard` | Visualização das métricas de treinamento |
| `numpy` | Operações numéricas |

---

## Referências

- **Artigo StyleTTS:** Li, Y. A., Han, C., & Mesgarani, N. (2022). *StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis.* [arXiv:2205.15439](https://arxiv.org/abs/2205.15439)
- **HiFi-GAN:** Kong, J. et al. (2020). *HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis.* [GitHub](https://github.com/jik876/hifi-gan)
- **JDCNet:** Kum, S., & Nam, J. (2019). *Joint Detection and Classification of Singing Voice Melody.*
- **monotonic_align:** [resemble-ai/monotonic_align](https://github.com/resemble-ai/monotonic_align)
- **phonemizer:** [bootphon/phonemizer](https://github.com/bootphon/phonemizer)
