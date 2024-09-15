import os
import pandas as pd
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from graphdatascience import GraphDataScience
from retry import retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from agents import BaseAgent
from services.entity_resolution.prompts import (
    SYSTEM_PROMPT_ENTITY,
    USER_PROMPT_ENTITY,
)
from services.entity_resolution.output_classes import Disambiguate


class CommunityBuilder(BaseAgent):
    def __init__(self) -> None:
        self.graph = Neo4jGraph()
        self.gds = GraphDataScience(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        if self.gds.graph.exists("communities")["exists"]:
            self.gds.graph.drop("communities")

        self.G, self.result = self.gds.graph.project(
            "communities",  #  Graph name
            "__Entity__",  #  Node projection
            {
                "_ALL_": {
                    "type": "*",
                    "orientation": "UNDIRECTED",
                    "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
                }
            },
        )

    @staticmethod
    def _prepare_string(data):
        nodes_str = "Nodes are:\n"
        for node in data["nodes"]:
            node_id = node["id"]
            node_type = node["type"]
            if "description" in node and node["description"]:
                node_description = f", description: {node['description']}"
            else:
                node_description = ""
            nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

        rels_str = "Relationships are:\n"
        for rel in data["rels"]:
            start = rel["start"]
            end = rel["end"]
            rel_type = rel["type"]
            if "description" in rel and rel["description"]:
                description = f", description: {rel['description']}"
            else:
                description = ""
            rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

        return nodes_str + "\n" + rels_str

    def _get_summary_agent(self):
        llm = self.get_llm()
        community_template = """Based on the provided nodes and relationships that belong to the same graph community,
        generate a natural language summary of the provided information:
        {community_info}

        Summary:"""  # noqa: E501

        community_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given an input triples, generate the information summary. No pre-amble.",
                ),
                ("human", community_template),
            ]
        )

        community_chain = community_prompt | llm | StrOutputParser()
        return community_chain

    def _get_weakly_connected_components(self):
        print("Run _get_weakly_connected_components ")
        wcc = self.gds.wcc.stats(self.G)
        print(f"Component count: {wcc['componentCount']}")
        print(f"Component distribution: {wcc['componentDistribution']}")

    def _run_leiden_algorithm(self):
        self.leiden_commuities = self.gds.leiden.write(
            self.G,
            writeProperty="communities",
            includeIntermediateCommunities=True,
            relationshipWeightProperty="weight",
        )
        print(self.leiden_commuities)

    def _create_communities(self):
        self.graph.query(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
        )
        check = self.graph.query(
            """
            MATCH (e:`__Entity__`)
            UNWIND range(0, size(e.communities) - 1 , 1) AS index
            CALL {
            WITH e, index
            WITH e, index
            WHERE index = 0
            MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
            ON CREATE SET c.level = index
            MERGE (e)-[:IN_COMMUNITY]->(c)
            RETURN count(*) AS count_0
            }
            CALL {
            WITH e, index
            WITH e, index
            WHERE index > 0
            MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
            ON CREATE SET current.level = index
            MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
            ON CREATE SET previous.level = index - 1
            MERGE (previous)-[:IN_COMMUNITY]->(current)
            RETURN count(*) AS count_1
            }
            RETURN count(*)
            """
        )
        print(f"Made communitiy nodes: {check}")

        # Here we Link the community node to the text chunk
        self.graph.query(
            """
        MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(:__Entity__)<-[:MENTIONS]-(d:Document)
        WITH c, count(distinct d) AS rank
        SET c.community_rank = rank;
        """
        )

    def _perform_community_check(self):
        community_size = self.graph.query(
            """
        MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(e:__Entity__)
        WITH c, count(distinct e) AS entities
        RETURN split(c.id, '-')[0] AS level, entities
        """
        )
        community_size_df = pd.DataFrame.from_records(community_size)
        percentiles_data = []
        for level in community_size_df["level"].unique():
            subset = community_size_df[community_size_df["level"] == level]["entities"]
            num_communities = len(subset)
            percentiles = np.percentile(subset, [25, 50, 75, 90, 99])
            percentiles_data.append(
                [
                    level,
                    num_communities,
                    percentiles[0],
                    percentiles[1],
                    percentiles[2],
                    percentiles[3],
                    percentiles[4],
                    max(subset),
                ]
            )

        # Create a DataFrame with the percentiles
        percentiles_df = pd.DataFrame(
            percentiles_data,
            columns=[
                "Level",
                "Number of communities",
                "25th Percentile",
                "50th Percentile",
                "75th Percentile",
                "90th Percentile",
                "99th Percentile",
                "Max",
            ],
        )
        print(f"Check community sizes")
        print(percentiles_df)

    def _prepare_community_info(self):
        self.community_info = self.graph.query(
            """
            MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
            WHERE c.level IN [0,1,2,3,4]
            WITH c, collect(e ) AS nodes
            WHERE size(nodes) > 1
            CALL apoc.path.subgraphAll(nodes[0], {
                whitelistNodes:nodes
            })
            YIELD relationships
            RETURN c.id AS communityId,
                [n in nodes | {id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '__Entity__'][0]}] AS nodes,
                [r in relationships | {start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}] AS rels
            """
        )
        print(self.community_info[2])

    def _process_community(self, community):
        community_chain = self._get_summary_agent()
        stringify_info = self._prepare_string(community)
        summary = community_chain.invoke({"community_info": stringify_info})
        return {"community": community["communityId"], "summary": summary}

    def _process_communuty_summaries(self):
        summaries = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._process_community, community): community
                for community in self.community_info
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing communities"
            ):
                summaries.append(future.result())
        print(summaries)
        # Store summaries
        self.graph.query(
            """
        UNWIND $data AS row
        MERGE (c:__Community__ {id:row.community})
        SET c.summary = row.summary
        """,
            params={"data": summaries},
        )

    def make_communities(self):
        self._get_weakly_connected_components()
        self._run_leiden_algorithm()
        self._create_communities()
        self._perform_community_check()
        self._prepare_community_info()
        self._process_communuty_summaries()
        self.G.drop()
