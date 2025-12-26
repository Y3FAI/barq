#!/usr/bin/env python3
"""
Barq Search - Command Line Interface

Orchestrates search across different engines with full timing.

Usage:
    python search.py --query "رخصة قيادة"
    python search.py --query "تدريب" --top 5
    python search.py --query "صندوق" --verbose
    python search.py --query "test" --engine keyword
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from search.core.orchestrator import SearchOrchestrator
from search.core.factory import EngineFactory
from search.evaluation.ground_truth import GroundTruth
from search.evaluation.metrics import SearchMetrics
from search.evaluation.benchmark import Benchmark


def format_time(ms):
    """Format milliseconds for display"""
    if ms < 1:
        return f"{ms*1000:.0f}μs"
    elif ms < 1000:
        return f"{ms:.1f}ms"
    else:
        return f"{ms/1000:.2f}s"


def main():
    parser = argparse.ArgumentParser(
        description='Barq Search - Search Arabic government services',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  python search.py --query "رخصة قيادة"

  # Get top 5 results
  python search.py --query "تدريب" --top 5

  # Show detailed timing breakdown
  python search.py --query "صندوق" --verbose

  # Use different engine (when available)
  python search.py --query "test" --engine keyword

  # Use different database
  python search.py --query "test" --db path/to/barq.db
        """
    )

    # Required arguments (unless listing engines or benchmarking)
    parser.add_argument(
        '--query',
        required='--list-engines' not in sys.argv and '--benchmark' not in sys.argv,
        type=str,
        help='Search query (Arabic or English)'
    )

    # Optional arguments
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        metavar='N',
        help='Number of results to return (default: 10)'
    )

    parser.add_argument(
        '--db',
        type=str,
        default=str(Path(__file__).parent / 'db' / 'barq.db'),
        help='Path to database (default: project/db/barq.db)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed timing breakdown for optimization'
    )

    parser.add_argument(
        '--engine',
        type=str,
        default='keyword',
        help='Search engine to use (default: keyword). Use comma-separated for comparison: keyword,bm25'
    )

    parser.add_argument(
        '--list-engines',
        action='store_true',
        help='List all available search engines'
    )

    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run benchmark on all ground truth queries'
    )

    parser.add_argument(
        '--engines',
        type=str,
        default=None,
        metavar='ENGINE1,ENGINE2',
        help='Compare multiple engines (comma-separated, requires --benchmark)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Barq Search v0.1.0'
    )

    args = parser.parse_args()

    # List engines and exit
    if args.list_engines:
        print("\nAvailable search engines:")
        for engine_name in EngineFactory.list_engines():
            print(f"  - {engine_name}")
        print()
        return

    # Benchmark mode
    if args.benchmark:
        ground_truth_path = Path(__file__).parent / 'data' / 'ground_truth.json'
        ground_truth = GroundTruth(str(ground_truth_path))

        if ground_truth.count() == 0:
            print("\n✗ Error: No ground truth queries found")
            print(f"   Expected file: {ground_truth_path}")
            sys.exit(1)

        print()
        print("=" * 80)
        print(f"BENCHMARK MODE: {ground_truth.count()} ground truth queries")
        print("=" * 80)

        # Determine which engines to test
        if args.engines:
            # Multiple engines comparison
            engine_names = [name.strip() for name in args.engines.split(',')]
            engines = []
            for name in engine_names:
                try:
                    engines.append(EngineFactory.create(name))
                except ValueError as e:
                    print(f"\n✗ Error: {e}")
                    sys.exit(1)

            print(f"Comparing {len(engines)} engines: {', '.join(engine_names)}")
        else:
            # Single engine
            engines = [EngineFactory.create(args.engine)]
            print(f"Testing engine: {args.engine}")

        print(f"Database: {args.db}")
        print(f"Top results per query: {args.top}")
        print()

        # Run benchmark
        benchmark = Benchmark(ground_truth, args.db)

        if len(engines) == 1:
            # Single engine mode
            results = benchmark.run_single_engine(engines[0], top_k=args.top, verbose=True)

            # Display summary
            print()
            print("=" * 80)
            print("BENCHMARK RESULTS")
            print("=" * 80)
            print(f"Engine: {results.engine_name}")
            print(f"Queries tested: {len(results.query_results)}")
            print()
            print("Service Discovery Metrics:")
            print(f"  Hit@1: {results.avg_hit1:.1%}  (correct service is #1)")
            print(f"  Hit@3: {results.avg_hit3:.1%}  (correct service in top 3)")
            print(f"  Hit@5: {results.avg_hit5:.1%}  (correct service in top 5)")
            print(f"  MRR:   {results.avg_mrr:.3f}  (average position of correct service)")
            print()
            print(f"Average Speed:")
            print(f"  Per-query: {format_time(results.avg_time)}")
            print()

            if args.verbose:
                print("Per-Query Results:")
                print("-" * 80)
                for qr in results.query_results:
                    hit = "✓" if qr.metrics.success_at_5 > 0 else "✗"
                    print(f"{hit} {qr.query[:35]:35s} | MRR={qr.metrics.mrr:.2f}")

            print("=" * 80)
            print()

        else:
            # Multiple engines comparison
            all_results = benchmark.run_multiple_engines(engines, top_k=args.top, verbose=True)

            # Display comparison table
            print()
            print("=" * 80)
            print("ENGINE COMPARISON")
            print("=" * 80)
            print(f"Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR   | Speed  | Engine")
            print("-" * 80)

            for engine_name, results in all_results.items():
                print(f"{results.avg_hit1:>5.1%} | {results.avg_hit3:>5.1%} | "
                      f"{results.avg_hit5:>5.1%} | {results.avg_hit10:>6.1%} | {results.avg_mrr:.3f} | "
                      f"{format_time(results.avg_time):>6s} | {engine_name}")

            if args.verbose:
                # Per-query comparison
                print()
                print("Per-Query Comparison (MRR):")
                print("-" * 80)
                header = f"{'Query':<30s} |"
                for engine_name in all_results.keys():
                    header += f" {engine_name[:10]:>10s} |"
                print(header)
                print("-" * 80)

                queries = list(all_results.values())[0].query_results
                for i, qr in enumerate(queries):
                    row = f"{qr.query[:30]:30s} |"
                    for results in all_results.values():
                        mrr = results.query_results[i].metrics.mrr
                        row += f" {mrr:>10.2f} |"
                    print(row)

            print("=" * 80)
            print()

        return

    # Check if comparing multiple engines
    if ',' in args.engine:
        # Comparison mode
        engine_names = [name.strip() for name in args.engine.split(',')]

        print()
        print("=" * 80)
        print(f"Barq Search v0.1.0 - ENGINE COMPARISON")
        print("=" * 80)
        print(f"Query: '{args.query}'")
        print(f"Comparing: {', '.join(engine_names)}")
        print(f"Database: {args.db}")
        print(f"Top results: {args.top}")
        print("=" * 80)

        # Load ground truth
        ground_truth_path = Path(__file__).parent / 'data' / 'ground_truth.json'
        ground_truth = GroundTruth(str(ground_truth_path))

        comparison_results = {}

        for engine_name in engine_names:
            try:
                engine = EngineFactory.create(engine_name)
                orchestrator = SearchOrchestrator(engine, args.db)
                response = orchestrator.search_with_full_timing(args.query, top_k=args.top)

                # Evaluate if ground truth exists
                metrics = None
                if ground_truth.has_query(args.query):
                    gt = ground_truth.get(args.query)
                    result_ids = [r.doc_id for r in response.results]
                    metrics = SearchMetrics.evaluate_all(
                        results=result_ids,
                        correct_docs=gt.relevant_docs,
                        response_time_ms=response.timing.search_ms
                    )

                comparison_results[engine_name] = {
                    'response': response,
                    'metrics': metrics
                }

            except Exception as e:
                print(f"\n✗ Error with {engine_name}: {e}")
                continue

        # Display comparison
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)

        # Metrics table
        if all(r['metrics'] for r in comparison_results.values()):
            print(f"\nHit@1 | Hit@3 | Hit@5 | Hit@10 | MRR   | Speed  | Engine")
            print("-" * 80)
            for engine_name, data in comparison_results.items():
                m = data['metrics']
                t = data['response'].timing.search_ms
                h1 = "✓" if m.success_at_1 > 0 else "✗"
                h3 = "✓" if m.success_at_3 > 0 else "✗"
                h5 = "✓" if m.success_at_5 > 0 else "✗"
                h10 = "✓" if m.success_at_10 > 0 else "✗"
                print(f"  {h1}   |   {h3}   |   {h5}   |   {h10}    | {m.mrr:.3f} | {format_time(t):>6s} | {engine_name}")

        # Results side-by-side
        print("\n" + "=" * 80)
        print("TOP 5 RESULTS COMPARISON")
        print("=" * 80)

        for i in range(min(5, args.top)):
            print(f"\n[Position {i+1}]")
            print("-" * 80)
            for engine_name, data in comparison_results.items():
                results = data['response'].results
                if i < len(results):
                    r = results[i]
                    print(f"{engine_name:15s} | Doc {r.doc_id:4d} | Score: {r.score:6.1f}")
                    print(f"                | {r.content_snippet[:60]}...")
                else:
                    print(f"{engine_name:15s} | No result")

        print("\n" + "=" * 80)
        return

    # Single engine mode
    print()
    print("=" * 80)
    print(f"Barq Search v0.1.0")
    print("=" * 80)
    print(f"Query: '{args.query}'")
    print(f"Engine: {args.engine}")
    print(f"Database: {args.db}")
    print(f"Top results: {args.top}")
    print("-" * 80)

    try:
        # Create engine via factory
        engine = EngineFactory.create(args.engine)

        # Create orchestrator
        orchestrator = SearchOrchestrator(engine, args.db)

        # Execute search with full timing
        response = orchestrator.search_with_full_timing(args.query, top_k=args.top)

        # Display timing
        print()
        print("⏱  TIMING")
        print("-" * 80)
        print(f"Search time:  {format_time(response.timing.search_ms)}")

        if args.verbose:
            print()
            print("Step breakdown (for optimization):")
            print(f"  1. Engine init:    {format_time(response.timing.engine_init_ms)}")
            print(f"  2. Index load:     {format_time(response.timing.index_load_ms)}")
            print(f"  3. Search:         {format_time(response.timing.search_ms)}")
            print(f"  Total:             {format_time(response.timing.total_ms)}")
            print()
            print(f"Indexed documents: {response.total_indexed}")

        print("-" * 80)

        # Check for ground truth and evaluate
        ground_truth_path = Path(__file__).parent / 'data' / 'ground_truth.json'
        ground_truth = GroundTruth(str(ground_truth_path))

        if ground_truth.has_query(args.query):
            # Evaluate with ground truth
            print()
            print("📈 EVALUATION (with ground truth)")
            print("-" * 80)

            gt = ground_truth.get(args.query)
            result_ids = [r.doc_id for r in response.results]

            metrics = SearchMetrics.evaluate_all(
                results=result_ids,
                correct_docs=gt.relevant_docs,
                response_time_ms=response.timing.total_ms
            )

            # Service discovery metrics
            hit1_status = "✓" if metrics.success_at_1 > 0 else "✗"
            print(f"Hit@1:  {hit1_status}  (correct service as #1 result)")

            hit3_status = "✓" if metrics.success_at_3 > 0 else "✗"
            print(f"Hit@3:  {hit3_status}  (correct service in top 3)")

            hit5_status = "✓" if metrics.success_at_5 > 0 else "✗"
            print(f"Hit@5:  {hit5_status}  (correct service in top 5)")

            print(f"MRR:    {metrics.mrr:.3f}  ", end="")
            if metrics.mrr > 0:
                position = int(1 / metrics.mrr)
                print(f"(found at position #{position})")
            else:
                print("(service not found)")

            print()
            print(f"Ground truth: {gt.description}")
            print(f"Relevant docs: {sorted(gt.relevant_docs)}")

            print("-" * 80)

        # Display results
        print()
        print(f"📊 RESULTS ({len(response.results)} found)")
        print("-" * 80)

        if response.results:
            for i, result in enumerate(response.results, 1):
                print()
                print(f"[{i}] Score: {result.score:.0f} | Doc ID: {result.doc_id}")
                print(f"    URL: {result.url}")
                print(f"    {result.content_snippet[:150]}...")
        else:
            print()
            print("No results found.")
            print()
            print("Tips:")
            print("  - Try different keywords")
            print("  - Check spelling")
            print("  - Use simpler terms")

        print()
        print("=" * 80)
        print(f"✓ Search complete in {format_time(response.timing.search_ms)}")
        print(f"  Engine: {response.engine_name}")
        print(f"  Found: {len(response.results)} results")
        print("=" * 80)
        print()

    except FileNotFoundError:
        print()
        print(f"✗ Error: Database not found at {args.db}")
        print()
        print("Make sure you've run the crawler first:")
        print("  python crawl.py --bot mygovsa --max-pages 100")
        print()
        sys.exit(1)

    except KeyboardInterrupt:
        print()
        print("\nSearch interrupted by user.")
        sys.exit(0)

    except Exception as e:
        print()
        print(f"✗ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
