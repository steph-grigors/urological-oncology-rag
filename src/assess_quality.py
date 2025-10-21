"""
Data Quality Assessment Module
Analyzes collected papers for completeness and quality
"""

import json
import os
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any


class DataQualityAssessor:
    """Assesses quality of collected medical papers"""

    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.papers = []

    def load_papers(self) -> List[Dict[Any, Any]]:
        """Load all JSON papers from data directory"""
        print(f"📂 Loading papers from {self.data_dir}...")

        for file in self.data_dir.glob("*.json"):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    paper = json.load(f)
                    self.papers.append(paper)
            except Exception as e:
                print(f"   ⚠️  Failed to load {file.name}: {e}")

        print(f"   Loaded {len(self.papers)} papers\n")
        return self.papers

    def check_missing_fields(self) -> Dict[str, int]:
        """Check for missing critical fields"""
        print(f"🔍 Checking for missing fields...")

        missing = {
            'title': 0,
            'abstract': 0,
            'sections': 0,
            'full_text': 0,
            'authors': 0,
            'publication_date': 0,
            'journal': 0
        }

        for paper in self.papers:
            metadata = paper.get('metadata', {})

            if not metadata.get('title') or metadata.get('title') == 'No title':
                missing['title'] += 1
            if not metadata.get('abstract'):
                missing['abstract'] += 1
            if not paper.get('sections') or len(paper.get('sections', {})) == 0:
                missing['sections'] += 1
            if not paper.get('full_text'):
                missing['full_text'] += 1
            if not metadata.get('authors') or len(metadata.get('authors', [])) == 0:
                missing['authors'] += 1
            if not metadata.get('publication_date'):
                missing['publication_date'] += 1
            if not metadata.get('journal'):
                missing['journal'] += 1

        for field, count in missing.items():
            percentage = (count / len(self.papers)) * 100
            status = "✅" if count == 0 else "⚠️ "
            print(f"   {status} Papers missing {field}: {count} ({percentage:.1f}%)")

        print()
        return missing

    def analyze_sections(self) -> Dict[str, int]:
        """Analyze section distribution across papers"""
        print(f"📑 Analyzing section distribution...")

        all_sections = []
        section_counts_per_paper = []

        for paper in self.papers:
            sections = paper.get('sections', {})
            all_sections.extend(sections.keys())
            section_counts_per_paper.append(len(sections))

        section_distribution = Counter(all_sections)

        print(f"   Total unique section names: {len(section_distribution)}")
        print(f"   Average sections per paper: {sum(section_counts_per_paper)/len(section_counts_per_paper):.1f}")
        print(f"\n   Top 10 most common sections:")

        for section, count in section_distribution.most_common(10):
            percentage = (count / len(self.papers)) * 100
            print(f"      {section}: {count} papers ({percentage:.1f}%)")

        # Check for important sections
        print(f"\n   Standard section coverage:")
        important_sections = ['introduction', 'methods', 'results', 'discussion']
        for section in important_sections:
            count = section_distribution.get(section, 0)
            percentage = (count / len(self.papers)) * 100
            status = "✅" if percentage > 50 else "⚠️ "
            print(f"      {status} {section}: {count} papers ({percentage:.1f}%)")

        print()
        return dict(section_distribution)

    def analyze_text_lengths(self) -> Dict[str, Any]:
        """Analyze text length statistics"""
        print(f"📏 Analyzing text lengths...")

        full_text_lengths = []
        section_text_lengths = []
        abstract_lengths = []

        for paper in self.papers:
            # Full text length
            full_text = paper.get('full_text', '')
            full_text_lengths.append(len(full_text))

            # Total section text length
            sections = paper.get('sections', {})
            total_section_length = sum(len(text) for text in sections.values())
            section_text_lengths.append(total_section_length)

            # Abstract length
            abstract = paper['metadata'].get('abstract', '')
            abstract_lengths.append(len(abstract))

        stats = {
            'full_text': {
                'avg': sum(full_text_lengths) / len(full_text_lengths),
                'min': min(full_text_lengths),
                'max': max(full_text_lengths),
                'median': sorted(full_text_lengths)[len(full_text_lengths)//2]
            },
            'sections': {
                'avg': sum(section_text_lengths) / len(section_text_lengths),
                'min': min(section_text_lengths),
                'max': max(section_text_lengths)
            },
            'abstracts': {
                'avg': sum(abstract_lengths) / len(abstract_lengths),
                'min': min(abstract_lengths),
                'max': max(abstract_lengths)
            }
        }

        print(f"   Full text statistics:")
        print(f"      Average: {stats['full_text']['avg']:,.0f} characters")
        print(f"      Median:  {stats['full_text']['median']:,.0f} characters")
        print(f"      Range:   {stats['full_text']['min']:,} - {stats['full_text']['max']:,}")

        print(f"\n   Sections total statistics:")
        print(f"      Average: {stats['sections']['avg']:,.0f} characters")
        print(f"      Range:   {stats['sections']['min']:,} - {stats['sections']['max']:,}")

        print(f"\n   Abstract statistics:")
        print(f"      Average: {stats['abstracts']['avg']:,.0f} characters")
        print(f"      Range:   {stats['abstracts']['min']:,} - {stats['abstracts']['max']:,}")

        print()
        return stats

    def identify_problematic_papers(self) -> List[Dict[str, Any]]:
        """Identify papers with potential issues"""
        print(f"⚠️  Identifying problematic papers...")

        problematic = []

        for paper in self.papers:
            issues = []

            # Check full text length
            full_text_length = len(paper.get('full_text', ''))
            if full_text_length < 5000:
                issues.append(f"Very short ({full_text_length:,} chars)")

            # Check sections
            sections = paper.get('sections', {})
            if not sections:
                issues.append("No sections")
            elif len(sections) < 3:
                issues.append(f"Only {len(sections)} sections")

            # Check abstract
            abstract = paper['metadata'].get('abstract', '')
            if not abstract:
                issues.append("No abstract")
            elif len(abstract) < 100:
                issues.append("Very short abstract")

            # Check title
            title = paper['metadata'].get('title', '')
            if not title or title == 'No title':
                issues.append("Missing title")

            # Check important sections
            has_methods = any('method' in s.lower() for s in sections.keys())
            has_results = any('result' in s.lower() for s in sections.keys())

            if not has_methods:
                issues.append("No methods section")
            if not has_results:
                issues.append("No results section")

            if issues:
                problematic.append({
                    'pmc_id': paper.get('pmc_id', 'Unknown'),
                    'title': title[:60] + '...' if len(title) > 60 else title,
                    'issues': issues,
                    'full_text_length': full_text_length,
                    'section_count': len(sections)
                })

        if problematic:
            print(f"   Found {len(problematic)} papers with potential issues:")
            print(f"   (Showing top 10 most problematic)\n")

            # Sort by number of issues
            problematic.sort(key=lambda x: len(x['issues']), reverse=True)

            for i, p in enumerate(problematic[:10], 1):
                print(f"   {i}. PMC{p['pmc_id']}")
                print(f"      Title: {p['title']}")
                print(f"      Issues: {', '.join(p['issues'])}")
                print(f"      Length: {p['full_text_length']:,} chars, {p['section_count']} sections")
                print()
        else:
            print(f"   ✅ No problematic papers found!")

        print()
        return problematic

    def analyze_publication_dates(self) -> Dict[str, int]:
        """Analyze publication date distribution"""
        print(f"📅 Analyzing publication dates...")

        years = []
        missing_dates = 0

        for paper in self.papers:
            date = paper['metadata'].get('publication_date', '')
            if date and len(date) >= 4:
                year = date[:4]
                if year.isdigit():
                    years.append(year)
                else:
                    missing_dates += 1
            else:
                missing_dates += 1

        year_distribution = Counter(years)

        print(f"   Papers with valid dates: {len(years)}/{len(self.papers)}")
        print(f"   Papers with missing/invalid dates: {missing_dates}")

        if year_distribution:
            print(f"\n   Publication year distribution:")
            for year, count in sorted(year_distribution.items(), reverse=True):
                percentage = (count / len(self.papers)) * 100
                bar = '█' * int(percentage / 2)
                print(f"      {year}: {count:2d} papers {bar} ({percentage:.1f}%)")

        print()
        return dict(year_distribution)

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        print("="*70)
        print(" "*20 + "📊 DATA QUALITY REPORT")
        print("="*70)
        print()

        # Basic stats
        print(f"📦 Dataset Overview:")
        print(f"   Total papers collected: {len(self.papers)}")
        print(f"   Data directory: {self.data_dir}")
        print()

        # Run all assessments
        missing_fields = self.check_missing_fields()
        section_stats = self.analyze_sections()
        text_stats = self.analyze_text_lengths()
        problematic = self.identify_problematic_papers()
        year_stats = self.analyze_publication_dates()

        # Overall quality score
        print("="*70)
        print("🎯 Overall Quality Assessment:")
        print("="*70)

        quality_score = 100

        # Deduct points for issues
        if missing_fields['title'] > 0:
            quality_score -= 20
        if missing_fields['abstract'] > len(self.papers) * 0.1:
            quality_score -= 15
        if missing_fields['sections'] > 0:
            quality_score -= 20
        if len(problematic) > len(self.papers) * 0.3:
            quality_score -= 15
        if text_stats['full_text']['avg'] < 10000:
            quality_score -= 10

        quality_score = max(0, quality_score)

        if quality_score >= 90:
            grade = "A (Excellent)"
            emoji = "🌟"
        elif quality_score >= 80:
            grade = "B (Good)"
            emoji = "✅"
        elif quality_score >= 70:
            grade = "C (Acceptable)"
            emoji = "👍"
        elif quality_score >= 60:
            grade = "D (Needs Improvement)"
            emoji = "⚠️ "
        else:
            grade = "F (Poor)"
            emoji = "❌"

        print(f"\n   {emoji} Quality Score: {quality_score}/100 - Grade: {grade}")
        print()

        # Recommendations
        print("💡 Recommendations:")
        if len(problematic) > 0:
            print(f"   • Consider filtering out {len(problematic)} problematic papers")
        if missing_fields['abstract'] > 0:
            print(f"   • {missing_fields['abstract']} papers missing abstracts - may need special handling")
        if text_stats['full_text']['avg'] < 15000:
            print(f"   • Average paper length is shorter than typical research articles")
        if quality_score >= 90:
            print(f"   • ✅ Data quality is excellent! Ready for chunking and embedding.")

        print()
        print("="*70)

        return {
            'total_papers': len(self.papers),
            'missing_fields': missing_fields,
            'problematic_count': len(problematic),
            'quality_score': quality_score,
            'grade': grade,
            'text_stats': text_stats,
            'year_stats': year_stats
        }

    def save_report(self, output_file: str = "data/quality_report.json"):
        """Save quality report to JSON file"""
        report = self.generate_summary_report()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"💾 Report saved to: {output_file}")


def main():
    """Run quality assessment"""
    assessor = DataQualityAssessor(data_dir="data/raw")
    assessor.load_papers()
    assessor.save_report()


if __name__ == "__main__":
    main()
