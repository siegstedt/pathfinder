from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()
from typing import List, Optional
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import fitz  # PyMuPDF

examples = [
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "WORKS_FOR",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": (
            "Adam is a software engineer in Microsoft since 2009, "
            "and last year he got an award as the Best Talent"
        ),
        "head": "Adam",
        "head_type": "Person",
        "relation": "HAS_AWARD",
        "tail": "Best Talent",
        "tail_type": "Award",
    },
    {
        "text": (
            "Microsoft is a tech company that provide "
            "several products such as Microsoft Word"
        ),
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "PRODUCED_BY",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "lightweight app",
        "tail_type": "Characteristic",
    },
    {
        "text": "Microsoft Word is a lightweight app that accessible offline",
        "head": "Microsoft Word",
        "head_type": "Product",
        "relation": "HAS_CHARACTERISTIC",
        "tail": "accessible offline",
        "tail_type": "Characteristic",
    },
]


class UnstructuredRelation(BaseModel):
    head: str = Field(
        description=(
            "extracted head entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    head_type: str = Field(
        description="type of the extracted head entity like Person, Company, etc"
    )
    relation: str = Field(description="relation between the head and the tail entities")
    tail: str = Field(
        description=(
            "extracted tail entity like Microsoft, Apple, John. "
            "Must use human-readable unique identifier."
        )
    )
    tail_type: str = Field(
        description="type of the extracted tail entity like Person, Company, etc"
    )


class NodeExtractor:
    def create_unstructured_prompt_with_generated_description(
        node_labels: Optional[List[str]] = None,
        rel_types: Optional[List[str]] = None,
        previous_nodes: Optional[List[str]] = None,  # List of previous nodes
        previous_edges: Optional[List[str]] = None,  # List of previous edges
        existing_node_concepts: Optional[
            List[str]
        ] = None,  # List of existing concepts for nodes
    ) -> ChatPromptTemplate:

        node_labels_str = str(node_labels) if node_labels else ""
        rel_types_str = str(rel_types) if rel_types else ""

        # If previous_nodes is provided, convert it to a string to include in the prompt
        previous_nodes_str = (
            "\nThe following entities have been previously extracted. Ensure to use these entities if relevant, "
            "and prefer the most complete version of their names:\n"
            + ", ".join(previous_nodes)  # Join the list of previous nodes into a string
            if previous_nodes
            else ""
        )

        # If previous_edges are provided, convert it to a string to include in the prompt
        previous_edges_str = (
            "\nThe following concepts ofrelationships have been previously extracted. Ensure to use these relationships if relevant, "
            "and prefer these relationships where possible:\n"
            + ", ".join(previous_edges)  # Join the list of previous edges into a string
            if previous_edges
            else ""
        )

        # If existing node concepts are provided, include them in the prompt
        existing_node_concepts_str = (
            "\nThe following existing node concepts are available. If an extracted entity fits into one of these, "
            "use the existing concept type where appropriate:\n"
            + ", ".join(
                existing_node_concepts
            )  # Join the list of existing node concepts into a string
            if existing_node_concepts
            else ""
        )

        base_string_parts = [
            "You are a top-tier algorithm designed for extracting information in "
            "structured formats to build a knowledge graph. Your task is to identify "
            "the entities and relations requested with the user prompt from a given "
            "text. You must generate the output in a JSON format containing a list "
            'with JSON objects. Each object should have the keys: "head", '
            '"head_type", "relation", "tail", "tail_type", and "description". The "head" '
            "key must contain the text of the extracted entity with one of the types "
            "from the provided list in the user prompt.",
            (
                f'The "head_type" key must contain the type of the extracted head entity, '
                f"which must be one of the types from {node_labels_str}."
                if node_labels
                else ""
            ),
            (
                f'The "relation" key must contain the type of relation between the "head" '
                f'and the "tail", which must be one of the relations from {rel_types_str}.'
                if rel_types
                else ""
            ),
            (
                f'The "tail" key must represent the text of an extracted entity which is '
                f'the tail of the relation, and the "tail_type" key must contain the type '
                f"of the tail entity from {node_labels_str}."
                if node_labels
                else ""
            ),
            "IMPORTANT: Ensure that the chosen relation type accurately reflects the description of the relationship "
            "between the head and tail entities. For example, if the description indicates that someone is a director, "
            "the relation should be 'directs' rather than a generic or incorrect term like 'employs'. Always strive to select "
            "the most precise relation type based on the provided context and description.",
            "Attempt to extract as many entities and relations as you can. Maintain "
            "Entity Consistency: When extracting entities, it's vital to ensure "
            "consistency. If an entity is mentioned multiple times in the text but is referred "
            "to by different names or abbreviations (e.g., a short form or an alias), always "
            "use the most complete and descriptive identifier for that entity. For example, if "
            'a company is referred to by both its short name and full name (e.g., "XYZ Ltd" and "XYZ Limited"), '
            "make sure to use the full, expanded version.",
            "IMPORTANT: Every entity (head and tail) and every relationship must "
            'include a "description" field. If the description is not explicitly stated in the text, '
            "generate a meaningful description based on the available content. The description "
            "should provide useful context about the entity or relationship and explain its role or function.",
            "For relationships, ensure that the description explains the nature of the connection "
            "between the two entities. Do not use generic terms like 'relationship'—the description "
            "must be specific and explain how the entities are related.",
            previous_nodes_str,  # Include previous nodes in the system prompt
            previous_edges_str,  # Include previous edges in the system prompt
            existing_node_concepts_str,  # Include existing node concepts in the system prompt
            "When extracting nodes, first check if the extracted entity matches an existing node concept or any previous node "
            "extracted within the current document. If a match is found, use that concept or previous node instead of creating a new one.",
            "IMPORTANT NOTES:\n- Don't add any explanation or extra text.",
        ]

        system_prompt = "\n".join(filter(None, base_string_parts))

        system_message = SystemMessage(content=system_prompt)
        parser = JsonOutputParser(pydantic_object=UnstructuredRelation)

        human_string_parts = [
            "Based on the following example, extract entities and "
            "relations from the provided text.\n\n",
            (
                "Use the following entity types, don't use other entity "
                "that is not defined below:"
                "# ENTITY TYPES:"
                "{node_labels}"
                if node_labels
                else ""
            ),
            (
                "Use the following relation types, don't use other relation "
                "that is not defined below:"
                "# RELATION TYPES:"
                "{rel_types}"
                if rel_types
                else ""
            ),
            "Below are a number of examples of text and their extracted "
            "entities and relationships."
            "{examples}\n"
            "For the following text, extract entities and relations as "
            "in the provided example."
            "{format_instructions}\nText: {input}",
        ]

        human_prompt_string = "\n".join(filter(None, human_string_parts))
        human_prompt = PromptTemplate(
            template=human_prompt_string,
            input_variables=["input"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "node_labels": node_labels,
                "rel_types": rel_types,
                "examples": examples,
            },
        )

        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message, human_message_prompt]
        )

        return chat_prompt

    def __init__(self) -> None:
        self.graph = Neo4jGraph()

    def _get_model(
        self,
        model="gpt-4o",
        previous_nodes=None,
        previous_edges=None,
        existing_node_concepts=None,
    ):
        llm = ChatOpenAI(temperature=0, model_name=model)
        graph_model = LLMGraphTransformer(
            llm=llm,
            prompt=self.create_unstructured_prompt_with_generated_description(
                previous_nodes=previous_nodes,
                previous_edges=previous_edges,
                existing_node_concepts=existing_node_concepts,
            ),
            node_properties=["description"],
            relationship_properties=["description"],
        )
        return graph_model

    def extract(self, pdf_file: fitz.Document, pdf_name: str, **kwargs):
        # ToDo We should also add already extrected edges

        pdf_name_key = kwargs.get("pdf_name_key", "document_name")
        result = self.graph.query(
            """
            MATCH ()-[r]->()
            WHERE type(r) <> 'MENTIONS'
            RETURN DISTINCT type(r) AS relationshipType
            """
        )
        all_edges = set()
        for record in result:
            all_edges.add(record["relationshipType"])

        # Query to get all unique nodes with their node type
        result_nodes = self.graph.query(
            """
            MATCH (n)
            WITH n, [label IN labels(n) WHERE label <> 'Document' AND label <> '__Entity__'][0] AS validLabel
            RETURN DISTINCT n.id AS nodeId, validLabel AS nodeType
            """
        )
        # Add node id and node type to a set of dictionaries
        all_node_concepts = set()
        for record in result_nodes:
            if (
                not record["nodeType"] in ["Document", "__Entity__"]
                and not record["nodeType"] is None
            ):
                all_node_concepts.add(record["nodeType"])

        previous_nodes = set()
        graph_documents = []

        # Loop through pages in the pdf_file
        for page_num in range(len(pdf_file)):
            # We add old concepts and old nodes from this document to reduce duplicates
            graph_model = self._get_model(
                existing_node_concepts=all_node_concepts,
                previous_edges=all_edges,
                previous_nodes=previous_nodes,
            )
            print(f"Page Number: {page_num}")
            page = pdf_file.load_page(page_num)

            metadata = {pdf_name_key: pdf_name, "page_number": page_num + 1}
            documents = [Document(page_content=page.get_text(), metadata=metadata)]

            garp_doc = graph_model.convert_to_graph_documents(documents)
            graph_documents.extend(garp_doc)

            # For each graph document, extract and add new nodes to the set
            for document in garp_doc:
                for node in document.nodes:
                    # Add the node ID to the set of nodes
                    previous_nodes.add(node.id)
            for relationship in document.relationships:
                all_edges.add(relationship.type)
        self.graph.add_graph_documents(
            graph_documents, baseEntityLabel=True, include_source=True
        )
