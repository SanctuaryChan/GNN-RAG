import re
from typing import Optional

QUESTION_OPS = [
    (re.compile(r"^\s*when\b", re.I), "when"),
    (re.compile(r"^\s*where\b", re.I), "where"),
    (re.compile(r"^\s*who\b", re.I), "who"),
    (re.compile(r"^\s*which\b", re.I), "which"),
    (re.compile(r"^\s*how\s+many\b", re.I), "count"),
    (re.compile(r"^\s*how\s+much\b", re.I), "count"),
    (re.compile(r"^\s*how\s+old\b", re.I), "when"),
    (re.compile(r"\bfirst\b", re.I), "first"),
    (re.compile(r"\blast\b", re.I), "last"),
    (re.compile(r"\bcompare\b|\bmore\s+than\b|\bless\s+than\b", re.I), "compare"),
    (re.compile(r"^\s*what\b", re.I), "what"),
]


def guess_operator(question: str) -> str:
    if not question:
        return "other"
    for pattern, op in QUESTION_OPS:
        if pattern.search(question):
            return op
    return "other"


def relation_to_phrase(rel: str) -> str:
    if rel is None:
        return ""
    rel = rel.strip()
    if rel == "":
        return rel
    rel = rel.replace("/", " ")
    if "." in rel:
        rel = rel.split(".")[-1]
    rel = rel.replace("_", " ")
    rel = rel.replace("-", " ")
    rel = re.sub(r"\s+", " ", rel)
    return rel.strip()


def entity_to_text(entity: str, entity_map: Optional[dict] = None) -> str:
    if entity is None:
        return ""
    if entity_map and entity in entity_map:
        return entity_map[entity]
    return str(entity)


def verbalize_path(
    path: list,
    question: Optional[str] = None,
    operator: Optional[str] = None,
    entity_map: Optional[dict] = None,
    answer: Optional[str] = None,
) -> str:
    if not path:
        return ""

    if operator is None:
        operator = guess_operator(question)

    entities = [entity_to_text(path[0][0], entity_map=entity_map)]
    relations = []
    for h, r, t in path:
        relations.append(relation_to_phrase(r))
        entities.append(entity_to_text(t, entity_map=entity_map))

    if len(path) == 1:
        rel = relations[0]
        e0, e1 = entities[0], entities[1]
        if operator in {"when", "where", "who", "which", "what", "count", "first", "last"}:
            text = f"The {rel} of {e0} is {e1}."
        else:
            text = f"{e0} {rel} {e1}."
    elif len(path) == 2:
        rel1, rel2 = relations[0], relations[1]
        e0, e2 = entities[0], entities[2]
        text = f"The {rel2} of the {rel1} of {e0} is {e2}."
    else:
        chunks = [f"Starting from {entities[0]}" ]
        for idx in range(len(relations)):
            chunks.append(f"follow {relations[idx]} to {entities[idx + 1]}")
        text = ", then ".join(chunks) + "."

    if answer:
        return f"This path supports answer \"{answer}\": {text}"
    return text
