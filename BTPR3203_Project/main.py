from src.rq1_analysis import run_rq1
from src.rq2_analysis import run_rq2
from src.rq3_analysis import run_rq3


def main():
    print("=" * 70)
    print("BTPR3203 GROUP PROJECT")
    print("Graduate Underemployment in Malaysia")
    print("=" * 70)

    print("\nRunning RQ1...")
    run_rq1()

    print("\nRunning RQ2...")
    run_rq2()

    print("\nRunning RQ3...")
    run_rq3()

    print("\nAll analyses completed successfully.")


if __name__ == "__main__":
    main()