%pip -q install google-genai
import os
from google.colab import userdata

os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

from google import genai

client = genai.Client()

MODEL_ID = "gemini-2.0-flash"

from IPython.display import HTML, Markdown

!pip install -q google-adk

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types  # Para criar conteúdos (Content e Part)
from datetime import date
import textwrap # Para formatar melhor a saída de texto
from IPython.display import display, Markdown # Para exibir texto formatado no Colab
import requests # Para fazer requisições HTTP
import warnings

warnings.filterwarnings("ignore")

# Função auxiliar que envia uma mensagem para um agente via Runner e retorna a resposta final
def call_agent(agent: Agent, message_text: str) -> str:
    # Cria um serviço de sessão em memória
    session_service = InMemorySessionService()
    # Cria uma nova sessão (você pode personalizar os IDs conforme necessário)
    session = session_service.create_session(app_name=agent.name, user_id="user1", session_id="session1")
    # Cria um Runner para o agente
    runner = Runner(agent=agent, app_name=agent.name, session_service=session_service)
    # Cria o conteúdo da mensagem de entrada
    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    final_response = ""
    # Itera assincronamente pelos eventos retornados durante a execução do agente
    for event in runner.run(user_id="user1", session_id="session1", new_message=content):
        if event.is_final_response():
          for part in event.content.parts:
            if part.text is not None:
              final_response += part.text
              final_response += "\n"
    return final_response

# Função auxiliar para exibir texto formatado em Markdown no Colab
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

##########################################
# --- Agente 1: Buscador de Notícias --- #
##########################################
def agente_buscador(topico, data_de_hoje):
    buscador = Agent(
        name="agente_buscador",
        model="gemini-2.0-flash",
        description="Agente que busca informação no google",
        tools=[google_search],
        instruction="""
        Função: Agente especializado em buscar e coletar dados e informações
        relevantes em (Google Search).Fornecer pesquisa detalhados
        e relevantes para o planejamento e organização de conteúdo.
        Identificar e extrair informações chave, dados, tendências e exemplos.
        Summarizar os achados de forma clara e concisa.
        """
    )
    entrada_do_agente_buscador = f"Tópico: {topico}\Data de hoje: {data_de_hoje}"
    # Executa o agente
    resposta_do_agente_buscador = call_agent(buscador, entrada_do_agente_buscador)
    return resposta_do_agente_buscador

################################################
# --- Agente 2: Planejador de posts --- #
################################################
def agente_planejador(topico, lancamentos_buscados):
    planejador = Agent(
        name="agente_planejador",
        model="gemini-1.5-flash",
        instruction="""
        Função: Agente especializado em estruturar e organizar informações para
        criar planos de conteúdo detalhados e acionáveis para um post.
        Baseado no Tópico e nos dados de Lançamentos buscados, crie um plano
        de conteúdo organizado em SEÇÕES ENUMERADAS em Markdown.

        Exemplo de formato desejado:
        1. Título da Primeira Seção (ex: Introdução)
        2. Título da Segunda Seção (ex: Detalhes do Assunto A)
        3. Título da Terceira Seção (ex: Aspectos Relevantes do Assunto B)
        4. Conclusão

        Certifique-se de que cada ponto numerado é um título de seção claro e conciso.
        Acrescentar mais SEÇÕES ENUMERADAS caso seja necessario para melhor organização.
        Use os dados de 'Lançamentos buscados' para informar o conteúdo de cada seção no plano.
        O plano deve ser uma lista numerada que delineia a estrutura completa do post.
        """,
        description="Planejador de conteúdo",
    )

    entrada_do_agente_planejador = f"Tópico:{topico}\nLançamentos buscados: {lancamentos_buscados}"
    # Executa o agente
    plano_do_post = call_agent(planejador, entrada_do_agente_planejador)
    return plano_do_post

######################################
# --- Agente 3: Redator do Post --- #
######################################
# Modificado para apenas criar e retornar o objeto Agent
def agente_redator():
    redator = Agent(
        name="agente_redator",
        model="gemini-1.5-flash",
        instruction="""
        VOCÊ é o Agente Redator de Conteúdo, um redator criativo, meticuloso,técnico
        especializado..
        Sua tarefa é escrever UMA ÚNICA SEÇÃO do rascunho do post, conforme especificado.
        Use o Plano de Conteúdo fornecido para entender o contexto geral.
        Escreva apenas a seção indicada no prompt. Não escreva o post inteiro.
        Analisar o Plano de Conteúdo para entender a estrutura.
        Incorporar os Pontos Mais Relevantes/Dados no rascunho para esta seção.
        Manter um tom de um especialista e tecnico da área.
        """,
        description="Agente redator"
    )
    # Removido o passo de call_agent aqui
    return redator    

##########################################
# --- Agente 4: Revisor de Qualidade --- #
##########################################
# Modificado para apenas criar e retornar o objeto Agent
def agente_revisor():
    revisor = Agent(
        name="agente_revisor",
        model="gemini-1.5-flash",
        instruction="""
        Você é um Editor e Revisor de Conteúdo meticuloso, especializado e técnico.
        Use um tom de escrita tecnica e de um profissional especialista no assunto.
        Comparar o Rascunho (do Agente 3) com o Plano Original (do Agente 2).
        Revise o rascunho de post abaixo sobre o tópico indicado, verificando clareza, concisão, correção e tom.
        Se o rascunho estiver bom, responda apenas 'O rascunho está ótimo e pronto!'.
        Caso haja problemas, aponte-os e sugira melhorias.
        """,
        description="Agente revisor"
    )
    # Removido o passo de call_agent aqui
    return revisor


# Esta é uma célula de código protótipo para demonstrar a lógica.
# Ela substitui a última célula do seu código original.
# Certifique-se de ter as funções agente_buscador, agente_planejador,
# as novas versões de agente_redator e agente_revisor (que retornam apenas o Agent),
# call_agent, to_markdown
# e as configurações da API (client, MODEL_ID) prontas em células anteriores.

from datetime import date
import re # Vamos precisar de regex para extrair as seções do plano
from IPython.display import display, Markdown

# --- Funções Auxiliares (adapte conforme o formato exato do seu plano) ---
# Esta função tenta extrair títulos de seção de um texto de plano formatado.
# Você pode precisar ajustar esta função dependendo de como o agente_planejador
# formata o plano de post. Idealmente, instrua o agente_planejador a usar
# um formato consistente (por exemplo, lista numerada ou títulos H2 em markdown).
def extrair_secoes_do_plano(texto_plano: str) -> list[str]:
    secoes = []
    # Tenta encontrar títulos de seção comuns ou itens de lista numerada
    # Ex: ## Introdução, 1. Introdução, - Introdução
    padroes_secao = [
        r'^\s*##\s*(.+)',  # Títulos H2 em Markdown
        r'^\s*\d+\.\s*(.+)', # Listas numeradas (e.g., "1. Introdução")
        r'^\s*-\s*(.+)'     # Itens de lista com hífen (e.g., "- Introdução")
    ]

    # Divide o plano em linhas para facilitar a busca
    linhas = texto_plano.splitlines()

    for linha in linhas:
        for padrao in padroes_secao:
            match = re.match(padrao, linha.strip())
            if match:
                secao_titulo = match.group(1).strip()
                # Evita adicionar linhas vazias ou que não são títulos reais
                if secao_titulo and len(secao_titulo) > 3: # Filtro básico por tamanho
                    secoes.append(secao_titulo)
                break # Passa para a próxima linha se encontrou uma seção

    # Se nenhuma seção foi encontrada usando padrões comuns,
    # talvez o plano seja um bloco de texto simples.
    # Neste caso, pode ser difícil dividir automaticamente.
    # Uma alternativa seria gerar o texto como um todo (se couber no limite)
    # ou pedir ao usuário para identificar as seções.
    # Para este protótipo, vamos retornar uma seção única se nada for encontrado.
    if not secoes and texto_plano.strip():
         # Cria uma seção padrão se não conseguiu parsear nada
         return ["Conteúdo Principal"]
    elif not secoes and not texto_plano.strip():
        return [] # Retorna lista vazia se o plano estiver vazio

    return secoes

# --- Lógica Principal de Orquestração ---

data_de_hoje = date.today().strftime("%d/%m/%Y")

print("🚀 Iniciando o Sistema de planejamento de conteúdo com 4 Agentes 🚀")

# --- Obter o Tópico do Usuário ---
topico = input("❓ Por favor, digite o TÓPICO sobre o qual você quer criar o conteúdo: ")

if not topico:
  print("Por favor, insira um tópico válido.")
else:
  print(f"Maravilha! Vamos criar o conteúdo sobre {topico}")

  # --- Etapa 1: Agente Buscador ---
  # Aqui agente_buscador é chamado para *executar* a busca e retornar o resultado
  lançamentos_buscados = agente_buscador(topico, data_de_hoje)
  print("\n---   Resultado do Agente 1 (Buscador) ----\n")
  display(to_markdown(lançamentos_buscados))
  print("-------------------------------")

  # --- Etapa 2: Agente Planejador ---
  # Aqui agente_planejador é chamado para *executar* o planejamento e retornar o resultado
  plano_de_post = agente_planejador(topico, lançamentos_buscados)
  print("\n---   Resultado do Agente 2 (Planejador) ----\n")
  display(to_markdown(plano_de_post))
  print("-------------------------------") # Separador após exibir o plano

  # --- Etapa 3: Agente Redator (Gerando por Seção) ---
  print("\n---   Iniciando Geração do Rascunho (por Seção) ----\n")

  # Chame a nova função para OBTER o objeto Agent Redator configurado (com a nova instrução)
  # FAÇA ISSO APENAS UMA VEZ ANTES DO LOOP!
  redator_agent_obj = agente_redator() # <--- Obtém o objeto Agent usando a nova função

  secoes_do_plano = extrair_secoes_do_plano(plano_de_post)

  if not secoes_do_plano:
      print("Não foi possível extrair seções do plano. Não será possível gerar o rascunho por seção.")
      rascunho_do_post = ""
  else:
      rascunho_do_post_partes = []
      for i, secao in enumerate(secoes_do_plano):
          print(f"✏️ Gerando Seção {i+1}/{len(secoes_do_plano)}: {secao}")

          # Prepare a entrada ESPECÍFICA para esta chamada de call_agent...
          entrada_para_redator_secao = f"""
          Tópico Geral: {topico}

          Plano de Conteúdo Completo:
          {plano_de_post}

          ---
          POR FAVOR, ESCREVA APENAS O CONTEÚDO DA SEÇÃO ESPECÍFICA ABAIXO.
          Não inclua o título da seção no início da sua resposta.
          Seção a escrever: {secao}

          Instruções Adicionais para esta Seção:
          (Se houver instruções específicas para esta seção do plano, adicione aqui)
          """

          # Chame call_agent usando o objeto Agent e a entrada específica para a seção
          rascunho_da_secao = call_agent(redator_agent_obj, entrada_para_redator_secao) # <--- Use redator_agent_obj e a entrada específica!

          # Armazene a parte gerada...
          rascunho_do_post_partes.append(f"## {secao}\n\n{rascunho_da_secao}")

          # Opcional: Exibir a seção assim que ela for gerada...
          display(to_markdown(f"--- Seção Gerada: {secao} ---\n" + rascunho_da_secao))
          print("-" * 30)

      rascunho_do_post = "\n\n".join(rascunho_do_post_partes)

      print("\n--- Geração do Rascunho Completa ----\n")

  # --- Etapa 4: Agente Revisor ---
  if rascunho_do_post:
      print("\n---   Iniciando Revisão do Rascunho Completo ----\n")
      # Chame a nova função para OBTER o objeto Agent Revisor configurado
      revisor_agent_obj = agente_revisor() # <--- Obtém o objeto Agent usando a nova função
      entrada_para_revisor = f"Tópico: {topico}\nRascunho: {rascunho_do_post}"
      # Chame call_agent usando o objeto Agent e a entrada específica
      post_final = call_agent(revisor_agent_obj, entrada_para_revisor) # <--- Use revisor_agent_obj e a entrada!
      print("\n---   Resultado do Agente 4 (Revisor) ----\n")
      display(to_markdown(post_final))
      print("-------------------------------")
  else:
      print("\nNão foi possível gerar um rascunho para revisão.")

print("\n✅ Sistema de planejamento de conteúdo finalizado ✅")
