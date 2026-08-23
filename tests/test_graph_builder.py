import numpy as np

from pivotrag.graph_builder import find_similar_entities
from pivotrag.models import EntityNode


def _entity(
    entity_id: str,
    name: str,
    embedding: list[float],
    chunk_id: str,
) -> EntityNode:
    return EntityNode(
        id=entity_id,
        name=name,
        description=name,
        embedding=np.array(embedding, dtype=np.float32),
        source_chunk_ids=[chunk_id],
    )


def test_different_names_in_same_chunk_receive_similarity_edge() -> None:
    entities = {
        "e1": _entity("e1", "Alpha", [1.0, 0.0], "chunk-1"),
        "e2": _entity("e2", "Beta", [1.0, 0.0], "chunk-1"),
    }

    edges, merge_map = find_similar_entities(
        entities,
        threshold=0.8,
        top_l=1,
    )

    assert merge_map == {}
    assert len(edges) == 1
    assert {edges[0].source, edges[0].target} == {"e1", "e2"}
    assert edges[0].type == "similarity"


def test_same_name_merging_uses_threshold_graph_connectivity() -> None:
    entities = {
        "e1": _entity("e1", "Alpha", [1.0, 0.0], "chunk-1"),
        "e2": _entity("e2", " alpha ", [0.9, 0.4359], "chunk-2"),
        "e3": _entity("e3", "Alpha", [0.62, 0.7846], "chunk-3"),
    }

    edges, merge_map = find_similar_entities(
        entities,
        threshold=0.8,
        top_l=2,
    )

    # e1~e2 and e2~e3 exceed the threshold, while e1~e3 does not.
    # Connected-component merging therefore combines all three mentions.
    assert edges == []
    assert merge_map == {"e2": "e1", "e3": "e1"}
