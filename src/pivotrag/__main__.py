"""Allow running as: python -m pivotrag.pipeline"""

from pivotrag.pipeline import _main

if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
