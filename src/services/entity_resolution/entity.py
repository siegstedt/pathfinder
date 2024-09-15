from ast import Raise
import os
import numpy as np
from .output_classes import MergedEntity, MergedEdges, Disambiguate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
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

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class EntityResolution(BaseAgent):
    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings()
        self.vector = Neo4jVector.from_existing_graph(
            self.embeddings,
            node_label="__Entity__",
            text_node_properties=["id", "description"],
            embedding_node_property="embedding",
        )
        self.gds = GraphDataScience(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        )
        self.graph = Neo4jGraph()

    def _get_embedding(self, text):
        embedding = self.embeddings.embed_query(text)

        # Ensure the embedding is a list of floats
        if isinstance(embedding, list):
            return [float(value) for value in embedding]

    def _cosine_similarity_matrix(self, embeddings):
        embedding_list = np.array(list(embeddings.values()))
        similarity_matrix = cosine_similarity(embedding_list)
        return similarity_matrix

    def _find_similar_entities(self, entities, similarity_matrix, threshold=0.75):
        similar_pairs = []
        entity_list = list(entities.keys())
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                if similarity_matrix[i][j] > threshold:
                    similar_pairs.append(
                        (entity_list[i], entity_list[j], similarity_matrix[i][j])
                    )
        return similar_pairs

    def _refine_results(
        self, combined_results, similarity_threshold=0.9, return_as_list=False
    ):
        """
        Refine and regroup results based on the cosine similarity of embeddings of `nodeId`.

        Args:
            combined_results (list): List of groups containing entities (list of dicts with `nodeId` and `description`).
            similarity_threshold (float, optional): Similarity threshold to group entities. Defaults to 0.9.
            return_as_list (bool, optional): If True, input and output will be lists of lists. Defaults to False.
        """
        all_entities = []

        # Step 1: Flatten all entities into a single list of dictionaries depending on the input format
        for group in combined_results:
            all_entities.extend(group["combinedResult"])

        # Step 2: Get embeddings for all unique `nodeId` entities and map them to the full dictionary
        entity_embeddings = {
            entity["nodeId"]: self._get_embedding(entity["nodeId"])
            for entity in all_entities
        }

        # Step 3: Compute the cosine similarity matrix for all unique entities
        unique_node_ids = list(entity_embeddings.keys())
        similarity_matrix = self._cosine_similarity_matrix(entity_embeddings)

        # Step 4: Group entities based on the passed similarity_threshold
        grouped_entities = []
        visited = set()  # Track nodeIds that have already been grouped

        for i in range(len(unique_node_ids)):
            if unique_node_ids[i] in visited:
                continue
            current_group = {unique_node_ids[i]}
            visited.add(unique_node_ids[i])

            for j in range(i + 1, len(unique_node_ids)):
                if (
                    unique_node_ids[j] not in visited
                    and similarity_matrix[i][j] > similarity_threshold
                ):
                    current_group.add(unique_node_ids[j])
                    visited.add(unique_node_ids[j])

            # Step 5: Only add groups with more than one entity
            if len(current_group) > 1:
                # Collect the full dictionaries for the grouped nodeIds
                grouped_result = []
                added_node_ids = (
                    set()
                )  # Track which nodeIds have been added to avoid duplicates

                for entity in all_entities:
                    if (
                        entity["nodeId"] in current_group
                        and entity["nodeId"] not in added_node_ids
                    ):
                        grouped_result.append(entity)
                        added_node_ids.add(entity["nodeId"])  # Mark nodeId as added

                grouped_entities.append(grouped_result)

        # Step 6: Return the refined and regrouped results as list of lists or list of dicts
        if return_as_list:
            return grouped_entities  # Return list of lists
        else:
            return [{"combinedResult": group} for group in grouped_entities]

    def _process_knearest_graph(self, similarity_threshold=0.8):
        """Processes entities or relationships and checks for similarity using KNN.

        Args:
            type (str): Specifies whether to process "node" or "relationship". Defaults to "node".
            similarity_threshold (float, optional): The similarity cutoff threshold. Defaults to 0.8.
        """

        # Create the graph projection
        if self.gds.graph.exists("entities")["exists"]:
            self.gds.graph.drop("entities")

        result = self.graph.query(
            """
            MATCH (n:`__Entity__`)
            RETURN n.id AS nodeId, n.embedding IS NOT NULL AS hasEmbedding, n.embedding AS embedding
            LIMIT 10
            """
        )

        # For nodes: Project node embeddings
        self.G, result = self.gds.graph.project(
            "entities",  # Graph name
            "__Entity__",  # Node projection
            "*",  # Relationship projection
            nodeProperties=["embedding"],  # Use node embeddings for similarity
        )

        # Check similarity using KNN for nodes
        self.gds.knn.mutate(
            self.G,
            nodeProperties=["embedding"],
            mutateRelationshipType="SIMILAR",  # Create similar relationships
            mutateProperty="score",  # Store the similarity score
            similarityCutoff=similarity_threshold,  # Threshold for similarity
        )

        # Write results back using weakly connected components (WCC)
        self.gds.wcc.write(self.G, writeProperty="wcc", relationshipTypes=["SIMILAR"])

    def _get_entity_text(self, entity_id):
        # Query the Neo4j graph to retrieve the relevant text (e.g., name or description) for the entity
        result = self.graph.query(
            """
            MATCH (e:`__Entity__` {id: $entity_id})
            RETURN e.id AS id, e.description AS description
            """,
            entity_id=entity_id,
        )

        # Check if the entity has a description; if not, fall back to the id
        if result and result[0].get("description"):
            return result[0]["description"]
        else:
            return result[0]["id"]  # Use the ID if no description is found

    def _generate_embeddings_for_entities(self, entity_ids):
        # Assuming we have a method to generate embeddings for the given list of entities
        for entity_id in entity_ids:
            # Retrieve the text data for the entity (e.g., name, description, etc.)
            entity_text = self._get_entity_text(entity_id)
            embedding = self.embeddings.embed_query(entity_text)
            # Store the embedding in the Neo4j graph
            self.graph.run(
                """
                MATCH (e:`__Entity__` {id: $entity_id})
                SET e.embedding = $embedding
                """,
                entity_id=entity_id,
                embedding=embedding,
            )

    def _get_potential_duplicates(self, method=False):
        if method == "ngrams":
            return self.graph.query(
                """
        WITH 20 AS word_edit_distance
        MATCH (e:__Entity__)
        WHERE size(e.id) > 4  // Ensure entities have meaningful IDs
        WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
        WHERE count > 1
        UNWIND nodes AS node

        // Tokenize the id fields by splitting on spaces
        WITH node, apoc.text.split(toLower(node.id), " ") AS tokens1, nodes

        // Unwind all nodes again for comparison
        UNWIND nodes AS otherNode

        // Include both nodes and otherNode in the WITH clause
        WITH node, otherNode, tokens1

        // Avoid comparing a node with itself
        WHERE node.id <> otherNode.id

        // Tokenize the other node's id field
        WITH node, otherNode, tokens1, apoc.text.split(toLower(otherNode.id), " ") AS tokens2

        // Find intersection of tokens between the two nodes
        WITH node, otherNode, apoc.coll.intersection(tokens1, tokens2) AS commonTokens

        // Only keep nodes that have common tokens
        WHERE size(commonTokens) > 0

        // Collect distinct otherNode ids and descriptions
        WITH node, otherNode, collect(DISTINCT {nodeId: otherNode.id, description: otherNode.description}) AS otherNodes, 
             {nodeId: node.id, description: node.description} AS currentNode

        // Combine currentNode and otherNodes into a list
        WITH collect(currentNode) AS currentNodeList, otherNodes
        WITH currentNodeList + otherNodes AS combinedResult

        // Filter out groups that have only one element
        WHERE size(combinedResult) > 1

        // Ensure distinct results and return in the required format
        RETURN DISTINCT combinedResult
        """
            )
        word_edit_distance = 30
        return self.graph.query(
            """
            MATCH (e:`__Entity__`)
            WHERE size(e.id) > 4 // longer than 4 characters
            WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
            WHERE count > 1
            UNWIND nodes AS node
            // Add text distance
            WITH distinct
            [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | {id: n.id, description: n.description}] AS intermediate_results, node.description AS nodeDescription
            WHERE size(intermediate_results) > 1
            WITH collect({id: node.id, description: nodeDescription} + intermediate_results) AS results
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
            WHERE NOT any(x IN range(0, size(allCombinedResults)-1, 1)
                WHERE x <> combinedResultIndex
                AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
            )
            RETURN combinedResult
            """,
            params={"distance": word_edit_distance},
        )

    @staticmethod
    def _clean_list_of_lists(input_list):
        # Remove lists with only one item
        filtered_list = [lst for lst in input_list if len(lst) > 1]

        # Remove duplicate lists
        unique_list = []
        for lst in filtered_list:
            if lst not in unique_list:
                unique_list.append(lst)

        return unique_list

    @retry(tries=3, delay=2)
    def _entity_resolution(self, entities: List[dict]) -> Optional[List[str]]:
        extraction_chain = self.structured_model(
            user_prompt=USER_PROMPT_ENTITY,
            system_prompt=SYSTEM_PROMPT_ENTITY,
            return_class=Disambiguate,
        )
        return extraction_chain.invoke({"entities": entities})

    def _save_merge(self, entities):
        prompt = """
        You are provided with a list of dictionaries, referred to as {entities}, each representing an entity with an 'id' and a 'description'. Your task is to consolidate these entities into a single object of the class `MergedEntity`. The goal is to merge the 'id' and 'description' fields based on the following criteria:

        1. id: Identify the most comprehensive and descriptive 'id' from the list. This should reflect the broadest and most informative name related to the entity.
        
        2. description: Combine all non-empty descriptions into a cohesive and comprehensive summary. Ensure that the final description contains all relevant information from the individual descriptions in a logical and flowing manner.

        The 'id' should be the most comprehensive and representative name, and the 'description' should contain all relevant details in a concise yet complete manner.
        """
        model = self.structured_model(return_class=MergedEntity, user_prompt=prompt)

        for entity_group in entities:
            # Step 1: Query all nodes from the graph for the current entity group

            result_node = model.invoke({"entities": entity_group})
            candidate_ids = [entity.nodeId for entity in entity_group]
            # Step 3: Use apoc.refactor.mergeNodes to merge nodes into the result_node
            self.graph.query(
                """
                MATCH (e:__Entity__) WHERE e.id IN $candidate_ids
                WITH collect(e) AS nodes
                // Identify the node to keep based on id_to_keep
                MATCH (keep_node:__Entity__ {id: $id_to_keep})
                CALL apoc.refactor.mergeNodes([keep_node] + [node IN nodes WHERE node.id <> $id_to_keep], 
                    {properties: {
                        `.*`: 'discard'  // Discard all other nodes' properties
                    }})
                YIELD node
                RETURN node
                """,
                params={"candidate_ids": candidate_ids, "id_to_keep": result_node.id},
            )

            # Step 4: Set the id and description of the merged node to match result_node
            self.graph.query(
                """
                MATCH (n:__Entity__)
                WHERE n.id = $result_node_id
                SET n.description = $new_description
                """,
                params={
                    "result_node_id": result_node.id,
                    "new_description": result_node.description,
                },
            )

        # Optionally drop temporary data structures
        self.G.drop()

    def get_all_nodes(self):
        return self.graph.query(
            """
            MATCH (n)
            RETURN collect(n.id) AS allNodes
            """
        )

    def run_entity_resolution(self, max_workers=5):

        self._process_knearest_graph(similarity_threshold=0.9)
        merged_entities = []

        potential_duplicate_candidates = self._get_potential_duplicates(method="ngrams")

        potential_duplicate_candidates = self._refine_results(
            potential_duplicate_candidates, similarity_threshold=0.9
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submitting all tasks and creating a list of future objects
            futures = [
                executor.submit(self._entity_resolution, el["combinedResult"])
                for el in potential_duplicate_candidates
            ]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing entities"
            ):
                to_merge = future.result()
                if to_merge:
                    merged_entities.append(to_merge)
        # cleaned_entities = self._clean_list_of_lists(merged_entities)
        merged_entities = [
            entity.merge_entities
            for entity in merged_entities
            if entity.to_be_merged == "yes"
        ]

        self._save_merge(merged_entities)

    def check_edges(self, candidate_list):
        """
        Use an LLM to generate a new description from a list of descriptions.
        This is a placeholder for your LLM integration.
        """
        # Combine all descriptions into a single input strin

        # Call to an LLM (this can be done via OpenAI API, GPT, etc.)
        # For example, assuming there's an LLM client that generates text

        prompt = """
        You are given a list of candidate edges. Each edge is represented by a dictionary with the keys: `edgeId`, `description`, `relType`, `sourceId`, and `targetId`. Your task is to evaluate pairs of edges and determine whether they represent the same underlying concept or entity. Follow these steps to complete the task:

        1. **Analyze potential candidate edges**: Compare the `description`, `sourceId`, and `targetId` fields of each edge.
        
        2. **Determine similarity**: Assess whether the `sourceId`, `targetId`, and `description` refer to the same or closely related concepts across different edges.
        - If unsure, **do not merge** the edges.
        - Be cautious; merge only when confident that the edges represent the same concept.

        3. **If edges should be merged**, provide the following:
        - `sourceId`: The selected `sourceId` from the merged edge, or an empty string if not merging.
        - `targetId`: The selected `targetId`, or an empty string if not merging.
        - `relType`: Adjust the `relType` if needed for accuracy. If unchanged, keep it as is.
        - `description`: Create a merged description that combines relevant details without repetition. Provide an empty string if no description applies.
        - `rationale`: Explain why the edges should be merged.
        - `to_be_merged`: Yes if merging; No otherwise.

        4. **If edges should not be merged**, return:
        - `rationale`: Briefly explain why the edges should **not** be merged.

        Please ensure precision in your decisions and provide clear justifications. Here is the candidate list for review: {candidate_list}
        """
        llm = self.structured_model(
            user_prompt=prompt, return_class=MergedEdges, include_raw=False
        )
        new_description = llm.invoke({"candidate_list": candidate_list})
        return new_description

    def add_embeddings_to_edges_without_embeddings(self, excluded_relationships):
        # Step 1: Query all edges that do not have embeddings
        query = """
        MATCH (src)-[r]->(tgt)
        WHERE r.embedding IS NULL AND NOT TYPE(r) IN $excluded_relationships
        RETURN r, r.description AS description, id(r) AS relId
        """

        # Run the query to get edges without embeddings
        result = self.graph.query(
            query, params={"excluded_relationships": excluded_relationships}
        )

        # Step 2: Generate embeddings for descriptions
        for record in result:
            edge_id = record["relId"]
            description = record["description"]

            if description:
                # Generate the embedding for the description
                embedding = self._get_embedding(description)

                # Step 3: Add the embedding back to the edge
                update_query = """
                MATCH ()-[r]->() 
                WHERE id(r) = $edge_id
                SET r.embedding = $embedding
                """

                # Update the edge with the new embedding
                self.graph.query(
                    update_query, params={"edge_id": edge_id, "embedding": embedding}
                )

    def process_similar_edges(self, excluded_relationships, similarity_threshold):
        # Step 1: Query all edges excluding excluded_relationships and return source/target descriptions
        query = """
        MATCH (a)-[r]->(b)
        WHERE NOT TYPE(r) IN $excluded_relationships
        RETURN id(r) AS edgeId, r.description AS description, r.embedding AS embedding, TYPE(r) AS relType, a.id AS sourceId, b.id AS targetId
        """

        # Execute the query and fetch edge data
        result = self.graph.query(
            query, params={"excluded_relationships": excluded_relationships}
        )

        # Step 2: Extract embeddings and edge information grouped by edge type
        edges_by_type = {}

        for record in result:
            rel_type = record["relType"]
            if rel_type not in edges_by_type:
                edges_by_type[rel_type] = {"edge_data": [], "embeddings": []}

            edges_by_type[rel_type]["edge_data"].append(
                {
                    "edgeId": record["edgeId"],
                    "description": record["description"],
                    "relType": rel_type,
                    "sourceId": record["sourceId"],  # Source node's 'id' property
                    "targetId": record["targetId"],  # Target node's 'id' property
                }
            )
            edges_by_type[rel_type]["embeddings"].append(record["embedding"])

        # Step 3: Process each group of edges independently based on their type and direction
        all_clusters = []

        for rel_type, data in edges_by_type.items():
            # Skip if there is only one edge, since clustering requires at least 2
            if len(data["embeddings"]) < 2:
                continue

            # Convert embeddings to numpy array
            embeddings = np.array(data["embeddings"])

            # Step 4: Compute Cosine Similarity using sklearn for each type
            similarity_matrix = cosine_similarity(embeddings)

            # Step 5: Perform clustering based on similarity
            clustering_model = AgglomerativeClustering(
                metric="precomputed",  # Replaces affinity, used for custom distances
                linkage="average",
                distance_threshold=1 - similarity_threshold,
                n_clusters=None,  # Let the model decide the number of clusters based on the threshold
            )

            # Convert cosine similarity to distance (1 - similarity)
            distance_matrix = 1 - similarity_matrix
            labels = clustering_model.fit_predict(distance_matrix)

            # Step 6: Group similar edges based on clustering labels
            similar_edges_groups = {}
            for idx, label in enumerate(labels):
                if label not in similar_edges_groups:
                    similar_edges_groups[label] = []

                edge = data["edge_data"][idx]
                # Ensure the edge is not added multiple times to the same cluster
                if edge not in similar_edges_groups[label]:
                    similar_edges_groups[label].append(edge)

            # Step 7: Filter out clusters with only one member
            filtered_similar_edges_groups = {
                label: edges
                for label, edges in similar_edges_groups.items()
                if len(edges) > 1
            }

            # Append the results for this relation type to the overall results
            all_clusters.extend(filtered_similar_edges_groups.values())

        # Return the clusters of similar edges with more than one member
        return all_clusters

    def edge_resolution(self):

        excluded_relationships = ["MENTIONS"]
        self.add_embeddings_to_edges_without_embeddings(excluded_relationships)
        potential_candidates = self.process_similar_edges(
            excluded_relationships=excluded_relationships, similarity_threshold=0.95
        )

        for candidates in potential_candidates:
            # Step 2: Combine descriptions using an LLM to generate a new description
            new_description = self.check_edges(candidates)
            if new_description.to_be_merged == "yes":
                self._merge_edges(
                    new_description.edge_ids,
                    new_description.description,
                    new_description.sourceId,
                    new_description.targetId,
                    new_description.relType,
                )

    def _merge_edges(self, edge_ids, description, sourceId, targetId, relType):
        embedding = self._get_embedding(description)

        # Step 2: First query to create the new edge
        self.graph.query(
            f"""
                // Match the source and target nodes by their IDs
                MATCH (source), (target)
                WHERE source.id = $sourceId
                AND target.id = $targetId

                // Create a new edge with the provided relationship type, description, embedding, and relType as a property
                CREATE (source)-[new_rel:{relType}]->(target)
                SET new_rel.description = $description,
                    new_rel.embedding = $embedding,
                    new_rel.type = $relType
                """,
            params={
                "sourceId": sourceId,
                "targetId": targetId,
                "description": description,
                "embedding": embedding,
                "relType": relType,
            },
        )

        # Step 3: Second query to delete the old edges by their IDs
        self.graph.query(
            """
                // Match the edges to be deleted by their edge IDs
                MATCH ()-[r]->()
                WHERE id(r) IN $edge_ids

                // Delete the old edges
                DELETE r
                """,
            params={"edge_ids": edge_ids},
        )
