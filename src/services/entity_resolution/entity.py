import os
from typing import List, Optional
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
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


class EntityResolution(BaseAgent):
    def __init__(self) -> None:
        self.vector = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            node_label="__Entity__",
            text_node_properties=["id", "description"],
            embedding_node_property="embedding",
        )
        self.gds = GraphDataScience(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        self.graph = Neo4jGraph()

    def _process_knearest_graph(self, similarity_threshold=0.95):
        """We use an algorithmen to check for enteties which are close, and make potential candidates

        Args:
            similarity_threshold (float, optional): How closly related should the enteties be. Defaults to 0.95.
        """
        # Create the graph
        self.G, result = self.gds.graph.project(
            "entities",  #  Graph name
            "__Entity__",  #  Node projection
            "*",  #  Relationship projection
            nodeProperties=["embedding"],  #  Configuration parameters
        )
        # Check similarity
        self.gds.knn.mutate(
            self.G,
            nodeProperties=["embedding"],
            mutateRelationshipType="SIMILAR",
            mutateProperty="score",
            similarityCutoff=similarity_threshold,
        )

        # Write results back
        self.gds.wcc.write(self.G, writeProperty="wcc", relationshipTypes=["SIMILAR"])

    def _get_potetial_duplicates(self):
        word_edit_distance = 3
        return self.graph.query(
            """MATCH (e:`__Entity__`)
            WHERE size(e.id) > 4 // longer than 4 characters
            WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
            WHERE count > 1
            UNWIND nodes AS node
            // Add text distance
            WITH distinct
            [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] AS intermediate_results
            WHERE size(intermediate_results) > 1
            WITH collect(intermediate_results) AS results
            // combine groups together if they share elements
            UNWIND range(0, size(results)-1, 1) as index
            WITH results, index, results[index] as result
            WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
                    CASE WHEN index <> index2 AND
                        size(apoc.coll.intersection(acc, results[index2])) > 0
                        THEN apoc.coll.union(acc, results[index2])
                        ELSE acc
                    END
            )) as combinedResult
            WITH distinct(combinedResult) as combinedResult
            // extra filtering
            WITH collect(combinedResult) as allCombinedResults
            UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
            WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
            WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
                WHERE x <> combinedResultIndex
                AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
            )
            RETURN combinedResult
        """,
            params={"distance": word_edit_distance},
        )

    @retry(tries=3, delay=2)
    def _entity_resolution(self, entities: List[str]) -> Optional[List[str]]:
        extraction_chain = self.structured_model(
            user_prompt=USER_PROMPT_ENTITY,
            system_prompt=SYSTEM_PROMPT_ENTITY,
            return_class=Disambiguate,
        )
        return [
            el.entities
            for el in extraction_chain.invoke({"entities": entities}).merge_entities
        ]

    def _save_merge(self, enteties):
        self.graph.query(
            """
            UNWIND $data AS candidates
            CALL {
            WITH candidates
            MATCH (e:__Entity__) WHERE e.id IN candidates
            RETURN collect(e) AS nodes
            }
            CALL apoc.refactor.mergeNodes(nodes, {properties: {
                `.*`: 'discard'
            }})
            YIELD node
            RETURN count(*)
            """,
            params={"data": enteties},
        )
        self.G.drop()

    def run_entity_resolution(self, max_workers=5):
        self._process_knearest_graph()
        merged_entities = []
        potential_duplicate_candidates = self._get_potetial_duplicates()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submitting all tasks and creating a list of future objects
            futures = [
                executor.submit(self._entity_resolution, el["combinedResult"])
                for el in potential_duplicate_candidates
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing documents"
            ):
                to_merge = future.result()
                if to_merge:
                    merged_entities.extend(to_merge)
        print(merged_entities[:10])
        self._save_merge(merged_entities)
