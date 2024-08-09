from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


class NodeExtractor:

    def __init__(self) -> None:
        self.graph = Neo4jGraph()

    def _get_model(self, model="gpt-4o"):
        llm = ChatOpenAI(temperature=0, model_name=model)
        graph_model = LLMGraphTransformer(
            llm=llm,
            node_properties=["description"],
            relationship_properties=["description"],
        )
        return graph_model

    def extract(self, pdf_file):
        graph_model = self._get_model()
        graph_documents = []
        # ToDO Currently we chunk by pages. It might be relevant to make one text, and then chunk smaller with overlapp.
        for page_num in range(len(pdf_file)):
            print(f"Page Number: {page_num}")
            page = pdf_file.load_page(page_num)
            documents = [Document(page_content=page.get_text())]
            graph_documents.extend(graph_model.convert_to_graph_documents(documents))
        self.graph.add_graph_documents(
            graph_documents, baseEntityLabel=True, include_source=True
        )
