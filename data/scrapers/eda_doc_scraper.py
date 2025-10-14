#!/usr/bin/env python3
"""
EDA Documentation Scraper

Scrapes public chip design documentation from open-source projects:
- Yosys (synthesis)
- OpenROAD (placement, routing)
- DREAMPlace (GPU placement)
- OpenSTA (timing analysis)
- Magic (VLSI layout)

Saves documentation to data/knowledge_base/ for RAG indexing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import logging
import requests
from pathlib import Path
from typing import List, Dict
import time
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EDADocScraper:
    """Scrape EDA tool documentation from public sources"""

    def __init__(self, output_dir: str = "./data/knowledge_base"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Documentation sources
        self.sources = {
            'yosys': {
                'name': 'Yosys Synthesis Suite',
                'urls': [
                    'https://yosyshq.readthedocs.io/projects/yosys/en/latest/',
                    'https://raw.githubusercontent.com/YosysHQ/yosys/master/README.md',
                ],
                'github_docs': 'https://github.com/YosysHQ/yosys/tree/master/docs',
            },
            'openroad': {
                'name': 'OpenROAD',
                'urls': [
                    'https://openroad.readthedocs.io/en/latest/',
                    'https://raw.githubusercontent.com/The-OpenROAD-Project/OpenROAD/master/README.md',
                ],
                'github_docs': 'https://github.com/The-OpenROAD-Project/OpenROAD/tree/master/docs',
            },
            'dreamplace': {
                'name': 'DREAMPlace',
                'urls': [
                    'https://raw.githubusercontent.com/limbo018/DREAMPlace/master/README.md',
                ],
                'github_docs': 'https://github.com/limbo018/DREAMPlace/tree/master/docs',
            },
            'openlane': {
                'name': 'OpenLane',
                'urls': [
                    'https://openlane.readthedocs.io/en/latest/',
                    'https://raw.githubusercontent.com/The-OpenROAD-Project/OpenLane/master/README.md',
                ],
            },
            'magic': {
                'name': 'Magic VLSI Layout Tool',
                'urls': [
                    'http://opencircuitdesign.com/magic/index.html',
                ],
            },
        }

    def scrape_all(self):
        """Scrape all EDA documentation sources"""
        logger.info("Starting EDA documentation scraping...")

        for tool_key, tool_info in self.sources.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Scraping {tool_info['name']}")
            logger.info(f"{'='*60}")

            tool_dir = self.output_dir / tool_key
            tool_dir.mkdir(exist_ok=True)

            # Scrape URLs
            for url in tool_info.get('urls', []):
                self.scrape_url(url, tool_dir)
                time.sleep(1)  # Be polite

            # Try to get GitHub docs
            if 'github_docs' in tool_info:
                self.scrape_github_docs(tool_info['github_docs'], tool_dir)

        logger.info("\n" + "="*60)
        logger.info("Scraping complete!")
        logger.info(f"Documentation saved to: {self.output_dir}")
        logger.info("="*60)

    def scrape_url(self, url: str, output_dir: Path):
        """Scrape a single URL"""
        try:
            logger.info(f"Fetching {url}")

            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # Determine file type
            if url.endswith('.md') or 'README' in url:
                # Markdown file
                filename = self._url_to_filename(url, '.md')
                filepath = output_dir / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)

                logger.info(f"  ✓ Saved markdown: {filename}")

            else:
                # HTML page
                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove script and style
                for script in soup(["script", "style", "nav", "footer"]):
                    script.decompose()

                # Extract main content (try common patterns)
                main_content = (
                    soup.find('main') or
                    soup.find('article') or
                    soup.find('div', class_='document') or
                    soup.find('div', id='content') or
                    soup.body
                )

                if main_content:
                    text = main_content.get_text(separator='\n', strip=True)

                    filename = self._url_to_filename(url, '.txt')
                    filepath = output_dir / filename

                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(f"Source: {url}\n\n")
                        f.write(text)

                    logger.info(f"  ✓ Saved HTML content: {filename}")

        except Exception as e:
            logger.error(f"  ✗ Error scraping {url}: {e}")

    def scrape_github_docs(self, github_url: str, output_dir: Path):
        """
        Scrape documentation from a GitHub docs directory.

        Note: This is a basic implementation. For better results,
        consider cloning the repo directly.
        """
        try:
            logger.info(f"Fetching GitHub docs from {github_url}")

            # Convert GitHub tree URL to API URL
            parts = github_url.replace('https://github.com/', '').split('/tree/')
            if len(parts) == 2:
                repo_path = parts[0]
                branch_and_path = parts[1].split('/', 1)
                branch = branch_and_path[0] if len(branch_and_path) > 0 else 'master'
                path = branch_and_path[1] if len(branch_and_path) > 1 else ''

                api_url = f"https://api.github.com/repos/{repo_path}/contents/{path}"

                response = requests.get(api_url, timeout=30)
                response.raise_for_status()

                files = response.json()

                for file_info in files:
                    if file_info['type'] == 'file' and file_info['name'].endswith(('.md', '.rst', '.txt')):
                        # Download file
                        file_url = file_info['download_url']
                        self.scrape_url(file_url, output_dir)
                        time.sleep(0.5)

        except Exception as e:
            logger.error(f"  ✗ Error scraping GitHub docs: {e}")

    def _url_to_filename(self, url: str, extension: str = '.txt') -> str:
        """Convert URL to safe filename"""
        # Extract meaningful part
        if 'README' in url:
            name = 'README'
        else:
            parts = url.split('/')
            name = '_'.join(parts[-3:])  # Last 3 parts

        # Clean up
        name = name.replace('https:', '').replace('http:', '')
        name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
        name = name[:100]  # Limit length

        if not name.endswith(extension):
            name += extension

        return name


def main():
    """Run the scraper"""
    scraper = EDADocScraper()
    scraper.scrape_all()

    print("\n✅ Documentation scraping complete!")
    print(f"\nNext steps:")
    print("1. Review downloaded docs in: data/knowledge_base/")
    print("2. Run: python data/scrapers/index_knowledge_base.py")
    print("3. This will index all docs into the RAG system")


if __name__ == "__main__":
    main()
