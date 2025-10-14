#!/usr/bin/env python3
"""
Verilog GitHub Scraper

Scrapes high-quality Verilog/SystemVerilog repositories from GitHub
for training data collection.

Focuses on:
- CPU designs (RISC-V, ARM implementations)
- SoC designs
- Academic projects
- Well-documented codebases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import requests
from pathlib import Path
import time
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerilogGitHubScraper:
    """
    Scrape Verilog repositories from GitHub for training data.
    """

    def __init__(self, output_dir: str = "./data/training/verilog_repos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Curated list of high-quality open-source chip design repos
        self.repositories = [
            # RISC-V Cores
            {
                'url': 'https://github.com/riscv/riscv-boom',
                'name': 'BOOM - Berkeley Out-of-Order Machine',
                'description': 'High-performance RISC-V core'
            },
            {
                'url': 'https://github.com/openhwgroup/cv32e40p',
                'name': 'CV32E40P',
                'description': 'RISC-V 32-bit processor core'
            },
            {
                'url': 'https://github.com/SpinalHDL/VexRiscv',
                'name': 'VexRiscv',
                'description': 'RISC-V CPU in SpinalHDL'
            },
            {
                'url': 'https://github.com/chipsalliance/rocket-chip',
                'name': 'Rocket Chip',
                'description': 'RISC-V SoC generator'
            },

            # ARM Implementations
            {
                'url': 'https://github.com/ultraembedded/core_arm',
                'name': 'ARM Cortex-M0 compatible core',
                'description': 'Open-source ARM core'
            },

            # GPUs
            {
                'url': 'https://github.com/jbush001/NyuziProcessor',
                'name': 'Nyuzi Processor',
                'description': 'Experimental GPGPU processor'
            },

            # NPUs/Accelerators
            {
                'url': 'https://github.com/nvdla/hw',
                'name': 'NVIDIA Deep Learning Accelerator',
                'description': 'Open-source DLA'
            },

            # SoC Platforms
            {
                'url': 'https://github.com/lowRISC/ibex',
                'name': 'Ibex',
                'description': 'Small 32-bit RISC-V CPU core'
            },
            {
                'url': 'https://github.com/pulp-platform/pulpissimo',
                'name': 'PULPissimo',
                'description': 'Multi-core SoC for IoT'
            },

            # Academic/Educational
            {
                'url': 'https://github.com/ZipCPU/zipcpu',
                'name': 'ZipCPU',
                'description': 'Small RISC CPU core'
            },

            # Standard Cell Libraries & PDKs
            {
                'url': 'https://github.com/google/skywater-pdk',
                'name': 'SkyWater PDK',
                'description': 'Open-source 130nm PDK'
            },
        ]

    def clone_repositories(self):
        """Clone all repositories"""
        logger.info("="*60)
        logger.info("Cloning Verilog Repositories")
        logger.info("="*60)

        cloned_count = 0
        skipped_count = 0

        for repo_info in self.repositories:
            url = repo_info['url']
            name = repo_info['name']
            repo_name = url.split('/')[-1]
            target_dir = self.output_dir / repo_name

            logger.info(f"\n{'─'*60}")
            logger.info(f"Repository: {name}")
            logger.info(f"URL: {url}")

            if target_dir.exists():
                logger.info(f"  ⊙ Already cloned: {repo_name}")
                skipped_count += 1
                continue

            try:
                logger.info(f"  ⟳ Cloning...")

                # Clone with depth=1 for speed (no full history needed)
                result = subprocess.run(
                    ['git', 'clone', '--depth', '1', url, str(target_dir)],
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode == 0:
                    logger.info(f"  ✓ Cloned successfully: {repo_name}")
                    cloned_count += 1

                    # Count verilog files
                    v_files = list(target_dir.rglob('*.v'))
                    sv_files = list(target_dir.rglob('*.sv'))
                    total_files = len(v_files) + len(sv_files)

                    logger.info(f"    Files: {len(v_files)} .v, {len(sv_files)} .sv (total: {total_files})")

                    time.sleep(2)  # Be polite to GitHub
                else:
                    logger.error(f"  ✗ Clone failed: {result.stderr}")

            except subprocess.TimeoutExpired:
                logger.error(f"  ✗ Clone timeout")
            except Exception as e:
                logger.error(f"  ✗ Error cloning: {e}")

        logger.info("\n" + "="*60)
        logger.info("Cloning Summary")
        logger.info("="*60)
        logger.info(f"Newly cloned: {cloned_count}")
        logger.info(f"Skipped (already exists): {skipped_count}")
        logger.info(f"Total repositories: {len(self.repositories)}")
        logger.info(f"Repository directory: {self.output_dir}")

    def collect_statistics(self):
        """Collect statistics about cloned repos"""
        logger.info("\n" + "="*60)
        logger.info("Repository Statistics")
        logger.info("="*60)

        total_v_files = 0
        total_sv_files = 0
        total_lines = 0

        for repo_dir in self.output_dir.iterdir():
            if not repo_dir.is_dir() or repo_dir.name.startswith('.'):
                continue

            v_files = list(repo_dir.rglob('*.v'))
            sv_files = list(repo_dir.rglob('*.sv'))

            # Count lines
            repo_lines = 0
            for file in v_files + sv_files:
                try:
                    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                        repo_lines += sum(1 for line in f)
                except Exception:
                    pass

            total_v_files += len(v_files)
            total_sv_files += len(sv_files)
            total_lines += repo_lines

            logger.info(f"\n{repo_dir.name}:")
            logger.info(f"  .v files: {len(v_files)}, .sv files: {len(sv_files)}")
            logger.info(f"  Total lines: {repo_lines:,}")

        logger.info("\n" + "="*60)
        logger.info("Overall Statistics")
        logger.info("="*60)
        logger.info(f"Total .v files: {total_v_files}")
        logger.info(f"Total .sv files: {total_sv_files}")
        logger.info(f"Total lines of code: {total_lines:,}")


def main():
    """Run the scraper"""
    scraper = VerilogGitHubScraper()

    logger.info("\n🚀 Starting Verilog Repository Collection\n")

    # Clone repos
    scraper.clone_repositories()

    # Show stats
    scraper.collect_statistics()

    logger.info("\n✅ Repository collection complete!")
    logger.info(f"\nRepositories saved to: {scraper.output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review code in: data/training/verilog_repos/")
    logger.info("2. Extract code snippets and comments for training")
    logger.info("3. Combine with EDA documentation for comprehensive dataset")


if __name__ == "__main__":
    main()
