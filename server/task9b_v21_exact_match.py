"""Exact, fail-closed semantic-negative query matching for Task 9B v2.1."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable


class ExactMatchInfeasible(RuntimeError):
    def __init__(self, report: dict[str, Any]):
        super().__init__(
            f"exact matching infeasible for {report['stratum']}: deficit={report['deficit']}"
        )
        self.report = report


@dataclass
class _Edge:
    to: int
    reverse: int
    capacity: int
    original_capacity: int


class _Dinic:
    def __init__(self, node_count: int):
        self.graph: list[list[_Edge]] = [[] for _ in range(node_count)]

    def add_edge(self, source: int, target: int, capacity: int) -> _Edge:
        forward = _Edge(target, len(self.graph[target]), capacity, capacity)
        reverse = _Edge(source, len(self.graph[source]), 0, 0)
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return forward

    def max_flow(self, source: int, sink: int) -> int:
        total = 0
        size = len(self.graph)
        while True:
            level = [-1] * size
            level[source] = 0
            queue = deque([source])
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge.capacity > 0 and level[edge.to] < 0:
                        level[edge.to] = level[node] + 1
                        queue.append(edge.to)
            if level[sink] < 0:
                return total
            cursor = [0] * size

            def send(node: int, amount: int) -> int:
                if node == sink:
                    return amount
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    if edge.capacity > 0 and level[node] + 1 == level[edge.to]:
                        pushed = send(edge.to, min(amount, edge.capacity))
                        if pushed:
                            edge.capacity -= pushed
                            reverse = self.graph[edge.to][edge.reverse]
                            reverse.capacity += pushed
                            return pushed
                    cursor[node] += 1
                return 0

            while True:
                pushed = send(source, 10**18)
                if not pushed:
                    break
                total += pushed


def _exact_match_stratum(
    families: list[dict[str, Any]], split: str, template_id: str
) -> tuple[dict[str, int], dict[str, Any]]:
    ordered = sorted(families, key=lambda row: str(row["family_id"]))
    quotas = Counter(int(row["positive_query_class_id"]) for row in ordered)
    classes = sorted(quotas)
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        forbidden = tuple(sorted({int(value) for value in row["present_class_ids"]}))
        grouped[forbidden].append(row)
    groups = sorted(grouped)

    source = 0
    group_node = {group: 1 + index for index, group in enumerate(groups)}
    class_node = {
        class_id: 1 + len(groups) + index for index, class_id in enumerate(classes)
    }
    sink = 1 + len(groups) + len(classes)
    flow = _Dinic(sink + 1)
    source_edges = {}
    assignment_edges = {}
    sink_edges = {}
    for group in groups:
        source_edges[group] = flow.add_edge(source, group_node[group], len(grouped[group]))
        for class_id in classes:
            if class_id not in group:
                assignment_edges[(group, class_id)] = flow.add_edge(
                    group_node[group], class_node[class_id], len(grouped[group])
                )
    for class_id in classes:
        sink_edges[class_id] = flow.add_edge(class_node[class_id], sink, quotas[class_id])

    required = len(ordered)
    achieved = flow.max_flow(source, sink)
    if achieved != required:
        unmet = {
            str(class_id): edge.capacity
            for class_id, edge in sink_edges.items()
            if edge.capacity
        }
        blocked_groups = [
            {
                "present_class_ids": list(group),
                "family_count": len(grouped[group]),
                "unassigned_count": source_edges[group].capacity,
                "allowed_quota_classes": [class_id for class_id in classes if class_id not in group],
            }
            for group in groups
            if source_edges[group].capacity
        ]
        raise ExactMatchInfeasible(
            {
                "version": "task9b-v21-exact-match-infeasible-1",
                "decision": "BLOCK",
                "reason": "no exact integer flow satisfies per-class quota and absent-query constraints",
                "stratum": {"split": split, "template_id": template_id},
                "required_flow": required,
                "achieved_flow": achieved,
                "deficit": required - achieved,
                "class_quotas": {str(key): value for key, value in sorted(quotas.items())},
                "unmet_class_quotas": unmet,
                "blocked_forbidden_groups": blocked_groups,
            }
        )

    assignments: dict[str, int] = {}
    for group in groups:
        available = iter(sorted(grouped[group], key=lambda row: str(row["family_id"])))
        for class_id in classes:
            edge = assignment_edges.get((group, class_id))
            count = 0 if edge is None else edge.original_capacity - edge.capacity
            for _ in range(count):
                family = next(available)
                assignments[str(family["family_id"])] = class_id
        try:
            next(available)
        except StopIteration:
            pass
        else:
            raise AssertionError("flow did not assign every family in a feasible group")

    actual = Counter(assignments.values())
    if actual != quotas:
        raise AssertionError("exact flow output does not preserve class quotas")
    return assignments, {
        "split": split,
        "template_id": template_id,
        "family_count": required,
        "class_quotas": {str(key): value for key, value in sorted(quotas.items())},
        "matched_class_counts": {str(key): value for key, value in sorted(actual.items())},
        "total_variation": 0.0,
        "exact": True,
    }


def match_all_strata(families: Iterable[dict[str, Any]]) -> dict[str, Any]:
    families = [dict(row) for row in families]
    identifiers = [str(row.get("family_id", "")) for row in families]
    if not families or any(not value for value in identifiers):
        raise ValueError("families must be non-empty and have non-empty IDs")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("family IDs must be unique")
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    family_stratum = {}
    for row in families:
        key = (str(row["split"]), str(row["template_id"]))
        strata[key].append(row)
        family_stratum[str(row["family_id"])] = {
            "split": key[0],
            "template_id": key[1],
        }

    assignment: dict[str, int] = {}
    reports = []
    for (split, template_id), rows in sorted(strata.items()):
        matched, report = _exact_match_stratum(rows, split, template_id)
        assignment.update(matched)
        reports.append(report)
    if set(assignment) != set(identifiers):
        raise AssertionError("global exact assignment is not a family bijection")
    return {
        "version": "task9b-v21-exact-match-1",
        "decision": "PASS",
        "assignment": dict(sorted(assignment.items())),
        "family_stratum": dict(sorted(family_stratum.items())),
        "family_count_before": len(families),
        "family_count_after": len(assignment),
        "family_bijection": True,
        "strata": reports,
    }
