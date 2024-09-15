from pydantic import BaseModel, Field
from typing import List, Optional


class Entity(BaseModel):
    nodeId: str = Field(description="The node id of the entity")
    description: Optional[str] = Field(description="The description of the node")


class Disambiguate(BaseModel):
    "A list of enteties which should be merged"
    merge_entities: Optional[List[Entity]] = Field(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )
    rationale: str = Field(
        description="A rationale why this entities should be merged or why none should be merged"
    )
    to_be_merged: str = Field(
        description="Either yes or no, depending if entities should be merged"
    )


class MergedEntity(BaseModel):
    id: str = Field(
        description=("The new node id for the node. The most comprehensive id")
    )
    description: str = Field(
        description="The description of the node, containing the relevant information"
    )
    rationale: str = Field(
        description="A rationale why these nodes should be merged or why not"
    )
    merged_ids: list[str] = Field(description="A list of nodeIds to be merged")


class MergedEdges(BaseModel):
    description: str = Field(
        description="The description of the new edge, containing the relevant information"
    )
    sourceId: str = Field(
        description="The choosen source node of the relationship, an empty string if not to be merged"
    )
    targetId: str = Field(
        description="The choosen target node of the relationship, an empty string if not to be merged"
    )
    relType: str = Field(description="A relType which fits the new description")
    rationale: str = Field(
        description="A rationale why these edges should be merged or why not"
    )
    edge_ids: list = Field(
        description="A list of the edge ids as integer, to be merged"
    )
    to_be_merged: str = Field(
        description="Either yes or no, depending if entities should be merged"
    )
