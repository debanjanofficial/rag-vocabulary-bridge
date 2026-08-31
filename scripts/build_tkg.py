"""CLI tool to build and serialize data/terminology_graph.json."""

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import DATA_DIR
from src.tkg.builder import build_terminology_kg
from src.tkg.traversal import SubgraphExtractor

DEFAULT_OUT = os.path.join(DATA_DIR, "terminology_graph.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct Terminology Knowledge Graph."
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT, help="Target JSON output path"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run validation and sample expansions"
    )
    args = parser.parse_args()

    print("Building Terminology Knowledge Graph...")
    kg = build_terminology_kg()
    summary = kg.summary()

    print(f"\n=======================================================")
    print(f"Terminology Knowledge Graph Built Successfully")
    print(f"=======================================================")
    print(f"Total Nodes: {summary['total_nodes']}")
    print(f"Total Edges: {summary['total_edges']}")
    print("\nNode Counts by Entity Type:")
    for etype, count in summary["entity_types"].items():
        print(f"  • {etype:20s}: {count:5d}")
    print("\nEdge Counts by Relation Type:")
    for rel, count in summary["relation_types"].items():
        print(f"  • {rel:22s}: {count:5d}")

    kg.save_json(args.out)
    print(f"\nSaved graph to -> {args.out}")

    if args.validate:
        print("\n--- Validation & Sample Subgraph Expansions ---")
        extractor = SubgraphExtractor(kg)
        test_queries = [
            "How does this metal-look material burn?",
            "How do I stop the laminate from bending when I press it?",
            "What's the allowed length variation for these shade matte parts?",
            "How do I drill holes in my crystal glass panels without breaking them?",
        ]
        for q in test_queries:
            sub = extractor.extract_subgraph(q)
            print(f"\nQuery: {q}")
            print(f"  • Matched Entry: {[n.label for n in sub.matched_nodes]}")
            print(f"  • Preferred Terms: {sub.preferred_terms}")
            print(f"  • Product Families: {sub.product_families}")
            print(f"  • Processes: {sub.processes}")
            print(f"  • Standards: {sub.standards}")
            print(f"  • Expanded: {sub.expansion_query}")


if __name__ == "__main__":
    main()
