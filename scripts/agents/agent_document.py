import os
import glob
import logging
import pandas as pd
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class AgentDocument:
    """
    Agente responsável por recuperar informações de documentos armazenados localmente.
    """

    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    def __init__(self, path_pattern: str):
        """
        path_pattern: padrão glob do arquivo, ex:
            'srag_2025_final_processed.csv'
            '*.csv'
        O sistema automaticamente vai buscar no datalake/gold/
        """

        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            current_dir = os.getcwd()

        project_root = os.path.dirname(os.path.dirname(current_dir))
        gold_dir = os.path.join(project_root, "datalake", "gold")

        self.gold_dir = gold_dir
        self.path = os.path.join(gold_dir, path_pattern)

        self.retriever = self.rag_retriever()

    def _build_summary_documents(self, df: pd.DataFrame) -> List[Document]:
        docs = []
        df_clean = df.copy()
        df_clean['DT_NOTIFIC_DT'] = pd.to_datetime(df_clean['DT_NOTIFIC'], errors='coerce')
        valid_df = df_clean[df_clean['DT_NOTIFIC_DT'].notna()]

        total_rows = len(valid_df)
        min_date = valid_df['DT_NOTIFIC_DT'].min().strftime('%Y-%m-%d')
        max_date = valid_df['DT_NOTIFIC_DT'].max().strftime('%Y-%m-%d')

        # 1. Documento Geral do Dataset
        febre_tot = (valid_df['FEBRE'] == 'Sim').sum()
        dispneia_tot = (valid_df['DISPNEIA'] == 'Sim').sum()
        uti_tot = (valid_df['UTI'] == 'Sim').sum()
        obito_tot = (valid_df['EVOLUCAO'] == 'Óbito').sum()
        vacina_tot = (valid_df['VACINA'] == 'Sim').sum()
        antiviral_tot = (valid_df['ANTIVIRAL'] == 'Sim').sum()
        fator_risc_tot = (valid_df['FATOR_RISC'] == 'Sim').sum()
        ave_suino_tot = valid_df['AVE_SUINO'].isin(['Sim, aves e/ou suínos', 'Sim, outros']).sum()
        surto_tot = (valid_df['SURTO_SG'] == 'Sim').sum()

        overall_text = (
            f"Visão Geral Completa do Banco de Dados SRAG:\n"
            f"- Total de Notificações Registradas: {total_rows}\n"
            f"- Período Epidemiológico: De {min_date} a {max_date}\n"
            f"- Taxa de Febre: {febre_tot} casos ({(febre_tot/total_rows)*100:.2f}%)\n"
            f"- Taxa de Dispneia: {dispneia_tot} casos ({(dispneia_tot/total_rows)*100:.2f}%)\n"
            f"- Taxa de Internação em UTI: {uti_tot} casos ({(uti_tot/total_rows)*100:.2f}%)\n"
            f"- Taxa de Óbito (Mortalidade): {obito_tot} casos ({(obito_tot/total_rows)*100:.2f}%)\n"
            f"- Cobertura Vacinal: {vacina_tot} pessoas ({(vacina_tot/total_rows)*100:.2f}%)\n"
            f"- Uso de Antiviral: {antiviral_tot} pacientes ({(antiviral_tot/total_rows)*100:.2f}%)\n"
            f"- Presença de Fatores de Risco: {fator_risc_tot} pacientes ({(fator_risc_tot/total_rows)*100:.2f}%)\n"
            f"- Contato com Aves/Suínos: {ave_suino_tot} casos ({(ave_suino_tot/total_rows)*100:.2f}%)\n"
            f"- Surtos de Síndrome Gripal (SG): {surto_tot} casos ({(surto_tot/total_rows)*100:.2f}%)"
        )
        docs.append(Document(page_content=overall_text, metadata={"type": "overall_summary"}))

        # 2. Documentos de Resumo Mensal
        valid_df['MONTH'] = valid_df['DT_NOTIFIC_DT'].dt.strftime('%Y-%m')
        for month, group in valid_df.groupby('MONTH'):
            tot = len(group)
            if tot == 0:
                continue
            febre = (group['FEBRE'] == 'Sim').sum()
            dispneia = (group['DISPNEIA'] == 'Sim').sum()
            uti = (group['UTI'] == 'Sim').sum()
            obito = (group['EVOLUCAO'] == 'Óbito').sum()
            vacina = (group['VACINA'] == 'Sim').sum()
            antiviral = (group['ANTIVIRAL'] == 'Sim').sum()
            fator_risc = (group['FATOR_RISC'] == 'Sim').sum()
            ave_suino = group['AVE_SUINO'].isin(['Sim, aves e/ou suínos', 'Sim, outros']).sum()
            surto = (group['SURTO_SG'] == 'Sim').sum()

            month_text = (
                f"Resumo Epidemiológico do Mês {month}:\n"
                f"- Total Notificações no mês: {tot}\n"
                f"- Febre: {febre} ({(febre/tot)*100:.2f}%)\n"
                f"- Dispneia: {dispneia} ({(dispneia/tot)*100:.2f}%)\n"
                f"- UTI: {uti} ({(uti/tot)*100:.2f}%)\n"
                f"- Óbitos: {obito} ({(obito/tot)*100:.2f}%)\n"
                f"- Vacina: {vacina} ({(vacina/tot)*100:.2f}%)\n"
                f"- Antiviral: {antiviral} ({(antiviral/tot)*100:.2f}%)\n"
                f"- Fator de Risco: {fator_risc} ({(fator_risc/tot)*100:.2f}%)\n"
                f"- Contato Aves/Suínos: {ave_suino} ({(ave_suino/tot)*100:.2f}%)\n"
                f"- Surtos SG: {surto} ({(surto/tot)*100:.2f}%)"
            )
            docs.append(Document(page_content=month_text, metadata={"type": "monthly_summary", "month": month}))

        # 3. Documentos por Semana Epidemiológica
        valid_df['WEEK'] = valid_df['DT_NOTIFIC_DT'].dt.strftime('%Y-W%U')
        for week, group in valid_df.groupby('WEEK'):
            tot = len(group)
            if tot == 0:
                continue
            uti = (group['UTI'] == 'Sim').sum()
            obito = (group['EVOLUCAO'] == 'Óbito').sum()
            week_text = (
                f"Semana Epidemiológica {week}: {tot} casos notificados de SRAG. "
                f"Internações em UTI: {uti} ({(uti/tot)*100:.1f}%). Óbitos: {obito} ({(obito/tot)*100:.1f}%)."
            )
            docs.append(Document(page_content=week_text, metadata={"type": "weekly_summary", "week": week}))

        # 4. Amostragem Estratificada de Registros Individuais
        columns_from_doc = [
            "DT_NOTIFIC", "AVE_SUINO", "FEBRE", "DISPNEIA",
            "DESC_RESP", "FATOR_RISC", "VACINA", "ANTIVIRAL",
            "UTI", "EVOLUCAO", "SURTO_SG"
        ]
        sample_size = min(500, len(valid_df))
        sample_df = valid_df.sample(n=sample_size, random_state=42)

        for idx, row in sample_df.iterrows():
            text_line = " | ".join(
                [f"{col}: {row[col]}" for col in columns_from_doc if col in row.index]
            )
            docs.append(
                Document(
                    page_content=f"Registro de Notificação SRAG ID {idx}: {text_line}",
                    metadata={"row_index": int(idx), "type": "sample_record"}
                )
            )

        return docs

    def rag_retriever(self):
        files = glob.glob(self.path)
        if not files:
            raise FileNotFoundError(f"Nenhum arquivo encontrado para o padrão: {self.path}")

        index_dir = os.path.join(self.gold_dir, "faiss_index")
        embeddings = OpenAIEmbeddings()

        if os.path.exists(index_dir) and os.path.isdir(index_dir):
            try:
                vector_store = FAISS.load_local(
                    index_dir, embeddings, allow_dangerous_deserialization=True
                )
                logging.info(f"Loaded FAISS index from disk at {index_dir}")
                return vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}
                )
            except Exception as e:
                logging.warning(f"Could not load FAISS index from {index_dir}: {e}. Rebuilding...")

        df = pd.concat(
            [pd.read_csv(arq, sep=";", on_bad_lines="skip") for arq in files],
            ignore_index=True
        )

        docs = self._build_summary_documents(df)

        vector_store = FAISS.from_documents(docs, embedding=embeddings)
        try:
            vector_store.save_local(index_dir)
            logging.info(f"Saved FAISS index to disk at {index_dir}")
        except Exception as e:
            logging.warning(f"Could not save FAISS index locally: {e}")

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

        logging.info(f"RAG Retriever created with {len(docs)} documents covering full dataset of {len(df)} rows")

        return retriever

    def create_new_state(self, state: dict) -> dict:
        information = self.retriever.invoke(state["question"])
        split_information = "\n".join([doc.page_content for doc in information])
        state["retrived_docs"] = split_information

        logging.info("Stored retrieved documents inside state['retrived_docs'].")

        return state
