# Em busca de ondas gravitacionais primordiais
<figure>
  <img src="images/WMAP_image_of_the_CMB_anisotropy.jpg" alt="CMBR map" width="500">
  <figcaption align="center">
    <b>Figure 1: WMAP (Wilkinson Microwave Anisotropy Probe) image of the CMB (Cosmic microwave background radiation) anisotropy.</b>
  </figcaption>
</figure>

![WMAP (Wilkinson Microwave Anisotropy Probe) image of the CMB (Cosmic microwave background radiation) anisotropy.](images/WMAP_image_of_the_CMB_anisotropy.jpg)

Este repositório contém o projeto de aprendizado de máquina, que utiliza o *framework* [Kedro](https://www.google.com/search?q=https://kedro.readthedocs.io/en/stable/).

**Importante:** Este arquivo README fornece as instruções para clonar e configurar o ambiente de desenvolvimento usando **Poetry**. Para instruções detalhadas sobre como interagir com o projeto Kedro (comandos de *pipeline*, estrutura de pastas, etc.), consulte o `README.md` principal localizado na subpasta do Kedro.

**Observação:** Para a capacitação básica em IA do instituto HBR, não é necessário utilizar nenhum comando do Kedro. Basta seguir as etapas abaixo para que os notebooks rodem corretamente.

## 🛠️ Configuração e Instalação

O projeto utiliza **Poetry** para gerenciamento de dependências e ambientes virtuais, garantindo um controle mais rigoroso de conflitos de pacotes fora do *template* padrão do Kedro.

Siga os passos abaixo para configurar o ambiente e rodar o projeto:

### Pré-requisitos

Certifique-se de ter o [Git](https://git-scm.com/) e o [Poetry](https://www.google.com/search?q=https://python-poetry.org/docs/%23installation) instalados em seu sistema.

### 1\. Clonar o Repositório

Abra o terminal e use o `git clone` para baixar o código:

```bash
git clone https://github.com/RenatodaCostaSantos/capacitacao.git
```

### 2\. Entrar na Pasta do Projeto

Navegue até a pasta raiz do projeto clonado:

```bash
cd <NOME_DA_PASTA_DO_PROJETO>
```

### 3\. Instalar as Dependências com Poetry

O Poetry irá criar um ambiente virtual (venv) e instalar todas as dependências especificadas no arquivo `pyproject.toml`.

```bash
poetry install --no-root
```

O argumento `--no-root` é usado pois, neste caso, a raiz do projeto principal é onde se encontra o Poetry, enquanto o pacote Kedro (que é o que queremos rodar) está em uma subpasta.

### 4\. Ativar o Ambiente Virtual (Opcional, mas Recomendado)

Se você planeja usar o terminal para comandos Kedro, scripts ou notebooks, ative o ambiente virtual:

```bash
eval $(poetry env actiave)
```

**Nota:** Se você for usar apenas os notebooks via VS Code ou outro IDE, o Poetry pode ser configurado para usar o venv automaticamente.

### 5\. Adicionar Dados Brutos

O projeto espera que os dados de entrada brutos estejam localizados na pasta `data/01_raw/` **dentro da subpasta `aeroespacial`**.

Adicione seus arquivos de dados (ex: CSVs, JSONs, etc.) nesta localização:

```
# Exemplo da estrutura:
<NOME_DA_PASTA_DO_PROJETO>/
├── .venv/ # venv criado pelo poetry
├── aeroespacial/
│
│   ├── data/
│   │   ├── 01_raw/
│   │   │   └── seus_dados_aqui.csv # <== Adicione aqui
│   │   └── ...
│   ├── README.md # README principal do Kedro
│    └── ...
└── README.md # Este arquivo
```

### 6\. Rodar os Notebooks

Os *notebooks* que interagem com o contexto do Kedro estão localizados dentro da subpasta do Kedro (em `aeroespacial/notebooks/`).

1.  Certifique-se de que o ambiente virtual do Poetry está **ativo** (veja passo 4).
2.  Navegue até a pasta dos notebooks: `cd aeroespacial/notebooks/`
3.  Abra e execute os notebooks usando seu editor preferido (Jupyter Lab, VS Code, etc.). Eles deverão carregar as dependências corretamente do ambiente Poetry.

-----

*Para comandos avançados do Kedro, como `kedro run`, `kedro test` ou visualização do pipeline, consulte o `README.md` principal na pasta do framework.*
