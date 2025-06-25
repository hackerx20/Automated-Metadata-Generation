import json
import uuid
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
import logging
import xml.etree.ElementTree as ET

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataGenerator:
    """
    Metadata generation module for creating structured metadata output
    """
    
    def __init__(self):
        self.metadata_schema_version = "1.0"
        self.generator_version = "1.0.0"
    
    def generate_metadata(self, file_info: Dict, text: str, analysis_results: Dict, pdf_metadata: Dict = None) -> Dict:
        """
        Generate comprehensive metadata from file info and analysis results
        
        Args:
            file_info: Basic file information
            text: Original text content
            analysis_results: Results from content analysis
            pdf_metadata: PDF-specific metadata (if applicable)
            
        Returns:
            Complete metadata dictionary
        """
        try:
            metadata = {
                'metadata_schema': {
                    'version': self.metadata_schema_version,
                    'generator': f"Automated Metadata Generation System v{self.generator_version}",
                    'generated_at': datetime.now().isoformat(),
                    'metadata_id': str(uuid.uuid4())
                },
                'file_info': self._generate_file_metadata(file_info),
                'content_analysis': self._generate_content_metadata(text, analysis_results),
                'semantic_analysis': self._generate_semantic_metadata(text, analysis_results),
                'processing_info': self._generate_processing_metadata(analysis_results),
                'quality_metrics': self._generate_quality_metrics(text, analysis_results)
            }
            
            # Add document classification
            metadata['document_classification'] = self._classify_document(text, analysis_results)
            
            # Add summary if possible
            summary = self._generate_summary(text, analysis_results)
            if summary:
                metadata['semantic_analysis']['summary'] = summary
            
            # Add enhanced metadata features
            metadata['document_relationships'] = self._analyze_document_relationships(text, analysis_results)
            metadata['structural_elements'] = self._catalog_structural_elements(text, analysis_results)
            metadata['integrity_verification'] = self._generate_integrity_hashes(file_info)
            
            # Add PDF-specific metadata if available
            if pdf_metadata:
                metadata['pdf_metadata'] = self._process_pdf_metadata(pdf_metadata)
                metadata['document_security'] = self._analyze_pdf_security(pdf_metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating metadata: {str(e)}")
            return self._generate_error_metadata(str(e), file_info)
    
    def _process_pdf_metadata(self, pdf_metadata: Dict) -> Dict:
        """Process and enhance PDF-specific metadata"""
        processed = {
            'technical_properties': {
                'pdf_version': pdf_metadata.get('pdf_version'),
                'page_count': pdf_metadata.get('page_count', 0),
                'is_encrypted': pdf_metadata.get('is_encrypted', False),
                'has_signatures': pdf_metadata.get('has_signatures', False),
                'embedded_files_count': len(pdf_metadata.get('embedded_files', [])),
                'form_fields_count': len(pdf_metadata.get('form_fields', [])),
                'annotations_count': len(pdf_metadata.get('annotations', []))
            },
            'creation_info': {
                'author': pdf_metadata.get('author', ''),
                'creator_application': pdf_metadata.get('creator', ''),
                'producer': pdf_metadata.get('producer', ''),
                'creation_date': pdf_metadata.get('creation_date', ''),
                'modification_date': pdf_metadata.get('modification_date', '')
            },
            'document_properties': {
                'title': pdf_metadata.get('title', ''),
                'subject': pdf_metadata.get('subject', ''),
                'keywords': pdf_metadata.get('keywords', ''),
            },
            'structure': {
                'bookmarks_count': len(pdf_metadata.get('bookmarks', [])),
                'has_table_of_contents': len(pdf_metadata.get('bookmarks', [])) > 0,
                'page_size_consistency': self._analyze_page_sizes(pdf_metadata.get('page_sizes', [])),
                'bookmarks': pdf_metadata.get('bookmarks', [])[:10]  # First 10 bookmarks
            },
            'interactive_elements': {
                'form_fields': pdf_metadata.get('form_fields', []),
                'annotations': pdf_metadata.get('annotations', []),
                'embedded_files': pdf_metadata.get('embedded_files', [])
            }
        }
        
        return processed
    
    def _analyze_pdf_security(self, pdf_metadata: Dict) -> Dict:
        """Analyze PDF security settings"""
        return {
            'is_encrypted': pdf_metadata.get('is_encrypted', False),
            'has_digital_signatures': pdf_metadata.get('has_signatures', False),
            'security_level': 'encrypted' if pdf_metadata.get('is_encrypted') else 'open',
            'interactive_elements_present': len(pdf_metadata.get('form_fields', [])) > 0,
            'embedded_content': len(pdf_metadata.get('embedded_files', [])) > 0
        }
    
    def _analyze_page_sizes(self, page_sizes: List[Dict]) -> Dict:
        """Analyze page size consistency"""
        if not page_sizes:
            return {'consistent': True, 'variations': 0}
        
        # Group by size
        size_groups = {}
        for page_info in page_sizes:
            size_key = f"{page_info.get('width', 0):.1f}x{page_info.get('height', 0):.1f}"
            if size_key not in size_groups:
                size_groups[size_key] = 0
            size_groups[size_key] += 1
        
        return {
            'consistent': len(size_groups) == 1,
            'variations': len(size_groups),
            'size_distribution': size_groups,
            'most_common_size': max(size_groups.keys(), key=lambda k: size_groups[k]) if size_groups else None
        }
    
    def _analyze_document_relationships(self, text: str, analysis_results: Dict) -> Dict:
        """Analyze document relationships and references"""
        relationships = {
            'references_other_documents': False,
            'is_part_of_series': False,
            'has_appendices': False,
            'citation_count': 0,
            'reference_types': [],
            'series_indicators': [],
            'cross_references': []
        }
        
        # Look for document series indicators
        series_patterns = [
            r'volume\s+\d+', r'part\s+\d+', r'chapter\s+\d+',
            r'section\s+\d+', r'book\s+\d+', r'edition\s+\d+'
        ]
        
        for pattern in series_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                relationships['is_part_of_series'] = True
                relationships['series_indicators'].extend(matches)
        
        # Look for appendices
        appendix_patterns = [r'appendix\s+[a-z]', r'annex\s+\d+', r'attachment\s+\d+']
        for pattern in appendix_patterns:
            if re.search(pattern, text.lower()):
                relationships['has_appendices'] = True
                break
        
        # Count citations and references
        citation_patterns = [
            r'\[[0-9]+\]',  # [1], [2], etc.
            r'\([0-9]{4}\)',  # (2023), etc.
            r'et\s+al\.',  # et al.
            r'ibid\.',  # ibid.
            r'op\.\s*cit\.',  # op. cit.
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            relationships['citation_count'] += len(matches)
            if matches:
                relationships['references_other_documents'] = True
                relationships['reference_types'].append(pattern)
        
        # Look for cross-references
        cross_ref_patterns = [
            r'see\s+(?:page|section|chapter|figure|table)\s+\d+',
            r'as\s+(?:mentioned|discussed|shown)\s+(?:in|on)\s+(?:page|section)\s+\d+',
            r'refer\s+to\s+(?:page|section|chapter|figure|table)\s+\d+'
        ]
        
        for pattern in cross_ref_patterns:
            matches = re.findall(pattern, text.lower())
            relationships['cross_references'].extend(matches)
        
        return relationships
    
    def _catalog_structural_elements(self, text: str, analysis_results: Dict) -> Dict:
        """Catalog structural elements like tables, figures, etc."""
        elements = {
            'tables': {'count': 0, 'indicators': []},
            'figures': {'count': 0, 'indicators': []},
            'equations': {'count': 0, 'indicators': []},
            'code_blocks': {'count': 0, 'indicators': []},
            'lists': {'count': 0, 'types': []},
            'footnotes': {'count': 0, 'indicators': []},
            'headers': {'count': 0, 'levels': []}
        }
        
        # Table indicators
        table_patterns = [
            r'table\s+\d+', r'tab\.\s+\d+', r'\|\s*[^|]+\s*\|',
            r'TABLES?:', r'│.*│'  # Various table formats
        ]
        
        for pattern in table_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                elements['tables']['count'] += len(matches)
                elements['tables']['indicators'].extend(matches[:5])  # First 5 matches
        
        # Figure indicators
        figure_patterns = [
            r'figure\s+\d+', r'fig\.\s+\d+', r'image\s+\d+',
            r'FIGURES?:', r'diagram\s+\d+'
        ]
        
        for pattern in figure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                elements['figures']['count'] += len(matches)
                elements['figures']['indicators'].extend(matches[:5])
        
        # Equation indicators
        equation_patterns = [
            r'equation\s+\d+', r'eq\.\s+\d+', r'\$.*\$',
            r'\\begin\{equation\}', r'\\[.*\\]'
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                elements['equations']['count'] += len(matches)
                elements['equations']['indicators'].extend(matches[:3])
        
        # Code block indicators
        code_patterns = [
            r'```[\s\S]*?```', r'`[^`]+`', r'def\s+\w+\s*\(',
            r'class\s+\w+:', r'#include\s*<', r'import\s+\w+'
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, text)
            if matches:
                elements['code_blocks']['count'] += len(matches)
                elements['code_blocks']['indicators'].extend(matches[:3])
        
        # List indicators
        list_patterns = [
            (r'^\s*[-*+]\s+', 'bullet'),
            (r'^\s*\d+\.\s+', 'numbered'),
            (r'^\s*[a-zA-Z]\.\s+', 'lettered')
        ]
        
        for pattern, list_type in list_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                elements['lists']['count'] += len(matches)
                if list_type not in elements['lists']['types']:
                    elements['lists']['types'].append(list_type)
        
        # Footnote indicators
        footnote_patterns = [r'\[\d+\]', r'^\d+\s+', r'\*+\s*']
        
        for pattern in footnote_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                elements['footnotes']['count'] += len(matches)
                elements['footnotes']['indicators'].extend(matches[:5])
        
        # Header analysis (from structure analysis)
        structure = analysis_results.get('structure', {})
        if structure.get('headings'):
            elements['headers']['count'] = len(structure['headings'])
            elements['headers']['levels'] = list(set(h.get('level', 1) for h in structure['headings']))
        
        return elements
    
    def _generate_integrity_hashes(self, file_info: Dict) -> Dict:
        """Generate multiple hash types for integrity verification"""
        hashes = {
            'sha256': None,
            'md5': None,
            'sha1': None,
            'file_size': file_info.get('size', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if 'content' in file_info and file_info['content']:
                content = file_info['content']
                if isinstance(content, str):
                    content = content.encode('utf-8')
                elif hasattr(content, 'getvalue'):
                    content = content.getvalue()
                
                hashes['sha256'] = hashlib.sha256(content).hexdigest()
                hashes['md5'] = hashlib.md5(content).hexdigest()
                hashes['sha1'] = hashlib.sha1(content).hexdigest()
                hashes['file_size'] = len(content)
        except Exception as e:
            logger.error(f"Error generating hashes: {e}")
        
        return hashes
    
    def _generate_file_metadata(self, file_info: Dict) -> Dict:
        """Generate file-specific metadata"""
        # Handle both 'name' and 'filename' keys for compatibility
        filename = file_info.get('name') or file_info.get('filename', 'unknown')
        file_size = file_info.get('size') or file_info.get('file_size', 0)
        file_type = file_info.get('type') or file_info.get('file_type', 'unknown')
        upload_time = file_info.get('upload_time') or file_info.get('upload_timestamp', datetime.now().isoformat())
        
        return {
            'filename': filename,
            'file_size': file_size,
            'file_size_human': self._format_file_size(file_size),
            'file_type': file_type,
            'mime_type': file_type,
            'upload_timestamp': upload_time,
            'file_hash': file_info.get('file_hash', 'unknown'),
            'file_extension': self._extract_extension(filename)
        }
    
    def _generate_content_metadata(self, text: str, analysis_results: Dict) -> Dict:
        """Generate content analysis metadata"""
        content_meta = {
            'word_count': analysis_results.get('word_count', 0),
            'character_count': analysis_results.get('character_count', 0),
            'character_count_no_spaces': analysis_results.get('character_count_no_spaces', 0),
            'sentence_count': analysis_results.get('sentence_count', 0),
            'paragraph_count': analysis_results.get('paragraph_count', 0),
            'language': analysis_results.get('language', 'Unknown'),
            'average_words_per_sentence': analysis_results.get('average_words_per_sentence', 0),
            'average_sentence_length': analysis_results.get('average_sentence_length', 0)
        }
        
        # Add readability metrics
        if 'readability_score' in analysis_results:
            content_meta.update({
                'readability_score': analysis_results['readability_score'],
                'flesch_kincaid_grade': analysis_results.get('flesch_kincaid_grade', 0),
                'reading_level': analysis_results.get('reading_level', 'Unknown')
            })
        
        # Add sentiment analysis
        if 'sentiment_score' in analysis_results:
            content_meta.update({
                'sentiment_score': analysis_results['sentiment_score'],
                'sentiment_label': analysis_results.get('sentiment_label', 'Neutral'),
                'sentiment_confidence': abs(analysis_results['sentiment_score'])
            })
        
        # Add word frequency (top 20)
        if 'word_frequency' in analysis_results:
            word_freq = analysis_results['word_frequency']
            content_meta['top_words'] = dict(list(word_freq.items())[:20])
            content_meta['vocabulary_diversity'] = len(word_freq) / max(content_meta['word_count'], 1)
        
        return content_meta
    
    def _generate_semantic_metadata(self, text: str, analysis_results: Dict) -> Dict:
        """Generate semantic analysis metadata"""
        semantic_meta = {}
        
        # Keywords
        if 'keywords' in analysis_results:
            semantic_meta['keywords'] = analysis_results['keywords'][:15]  # Top 15 keywords
        
        # Key phrases
        if 'key_phrases' in analysis_results:
            semantic_meta['key_phrases'] = analysis_results['key_phrases'][:10]  # Top 10 phrases
        
        # Named entities
        if 'named_entities' in analysis_results:
            entities = analysis_results['named_entities']
            semantic_meta['named_entities'] = entities
            
            # Group entities by type
            entity_types = {}
            for entity in entities:
                entity_type = entity.get('label', 'UNKNOWN')
                if entity_type not in entity_types:
                    entity_types[entity_type] = []
                entity_types[entity_type].append(entity['text'])
            
            semantic_meta['entity_types'] = entity_types
            semantic_meta['entity_count'] = len(entities)
        
        # Topics
        if 'topics' in analysis_results:
            semantic_meta['topics'] = analysis_results['topics']
            semantic_meta['primary_topic'] = analysis_results['topics'][0] if analysis_results['topics'] else 'General'
        
        # Document structure
        if 'structure' in analysis_results:
            structure = analysis_results['structure']
            semantic_meta['document_structure'] = {
                'has_introduction': structure.get('has_introduction', False),
                'has_conclusion': structure.get('has_conclusion', False),
                'has_headers': structure.get('has_headers', False),
                'structural_complexity': self._calculate_structural_complexity(structure)
            }
        
        return semantic_meta
    
    def _generate_processing_metadata(self, analysis_results: Dict) -> Dict:
        """Generate processing-related metadata"""
        return {
            'processing_timestamp': datetime.now().isoformat(),
            'processing_time': 0,  # Will be updated by the main app
            'nlp_model': 'spaCy en_core_web_sm',
            'analysis_methods': [
                'text_statistics',
                'readability_analysis',
                'sentiment_analysis',
                'keyword_extraction',
                'named_entity_recognition',
                'topic_modeling',
                'document_structure_analysis'
            ],
            'extraction_success': 'error' not in analysis_results,
            'extraction_quality': self._assess_extraction_quality(analysis_results)
        }
    
    def _generate_quality_metrics(self, text: str, analysis_results: Dict) -> Dict:
        """Generate quality assessment metrics"""
        metrics = {
            'text_completeness': min(1.0, len(text) / 1000),  # Normalized completeness score
            'analysis_completeness': 0,
            'metadata_confidence': 0,
            'data_quality_score': 0
        }
        
        # Calculate analysis completeness
        expected_fields = ['word_count', 'keywords', 'named_entities', 'sentiment_score']
        completed_fields = sum(1 for field in expected_fields if field in analysis_results)
        metrics['analysis_completeness'] = completed_fields / len(expected_fields)
        
        # Calculate metadata confidence based on data richness
        confidence_factors = []
        
        # Text length factor
        if len(text) > 500:
            confidence_factors.append(0.3)
        elif len(text) > 100:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(0.1)
        
        # Keywords factor
        if 'keywords' in analysis_results and len(analysis_results['keywords']) >= 5:
            confidence_factors.append(0.2)
        
        # Entities factor
        if 'named_entities' in analysis_results and len(analysis_results['named_entities']) >= 3:
            confidence_factors.append(0.2)
        
        # Structure factor
        if 'structure' in analysis_results:
            structure = analysis_results['structure']
            if structure.get('has_headers') or structure.get('has_introduction'):
                confidence_factors.append(0.1)
        
        # Language detection factor
        if analysis_results.get('language') == 'English':
            confidence_factors.append(0.2)
        
        metrics['metadata_confidence'] = sum(confidence_factors)
        
        # Overall data quality score
        quality_factors = [
            metrics['text_completeness'] * 0.3,
            metrics['analysis_completeness'] * 0.4,
            metrics['metadata_confidence'] * 0.3
        ]
        metrics['data_quality_score'] = sum(quality_factors)
        
        return metrics
    
    def _classify_document(self, text: str, analysis_results: Dict) -> Dict:
        """Classify document type and category"""
        classification = {
            'document_type': 'Unknown',
            'document_category': 'General',
            'confidence': 0.0,
            'classification_method': 'rule_based'
        }
        
        try:
            # Get keywords and topics for classification
            keywords = analysis_results.get('keywords', [])
            topics = analysis_results.get('topics', [])
            text_lower = text.lower()
            
            # Document type classification rules
            type_indicators = {
                'Report': ['report', 'analysis', 'findings', 'results', 'conclusion', 'executive summary'],
                'Article': ['article', 'journal', 'publication', 'author', 'abstract'],
                'Manual': ['manual', 'guide', 'instructions', 'procedure', 'steps', 'how to'],
                'Contract': ['contract', 'agreement', 'terms', 'conditions', 'party', 'legal'],
                'Proposal': ['proposal', 'project', 'budget', 'timeline', 'deliverables'],
                'Email': ['from:', 'to:', 'subject:', 'sent:', 're:', 'fwd:'],
                'Letter': ['dear', 'sincerely', 'regards', 'yours truly', 'signature'],
                'Academic': ['abstract', 'introduction', 'methodology', 'references', 'conclusion', 'university']
            }
            
            best_match = 'General'
            best_score = 0
            
            for doc_type, indicators in type_indicators.items():
                score = 0
                for indicator in indicators:
                    if indicator in text_lower:
                        score += 1
                    # Check in keywords too
                    if any(indicator in kw.lower() for kw in keywords if isinstance(kw, str)):
                        score += 0.5
                
                # Normalize score
                normalized_score = score / len(indicators)
                
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_match = doc_type
            
            classification['document_type'] = best_match
            classification['confidence'] = min(1.0, best_score)
            
            # Document category based on topics
            if topics:
                classification['document_category'] = topics[0]
            else:
                # Fallback category classification
                category_keywords = {
                    'Technical': ['software', 'system', 'technology', 'computer', 'data'],
                    'Business': ['business', 'company', 'market', 'financial', 'strategy'],
                    'Academic': ['research', 'study', 'university', 'academic', 'scientific'],
                    'Legal': ['legal', 'law', 'court', 'regulation', 'compliance'],
                    'Medical': ['health', 'medical', 'patient', 'treatment', 'doctor']
                }
                
                for category, cat_keywords in category_keywords.items():
                    if any(kw in text_lower for kw in cat_keywords):
                        classification['document_category'] = category
                        break
            
        except Exception as e:
            logger.warning(f"Error classifying document: {str(e)}")
        
        return classification
    
    def _generate_summary(self, text: str, analysis_results: Dict, max_sentences: int = 3) -> Optional[str]:
        """Generate document summary"""
        try:
            sentences = text.split('. ')
            
            if len(sentences) < 2:
                return None
            
            # Simple extractive summarization
            # Score sentences based on keyword presence and position
            keywords = analysis_results.get('keywords', [])
            scored_sentences = []
            
            for i, sentence in enumerate(sentences[:20]):  # First 20 sentences
                score = 0
                sentence_lower = sentence.lower()
                
                # Position score (earlier sentences get higher scores)
                position_score = 1.0 - (i / min(len(sentences), 20))
                score += position_score * 0.3
                
                # Keyword score
                keyword_score = sum(1 for kw in keywords if isinstance(kw, str) and kw.lower() in sentence_lower)
                score += keyword_score * 0.7
                
                # Length penalty (very short or very long sentences)
                word_count = len(sentence.split())
                if 10 <= word_count <= 30:
                    score += 0.2
                
                scored_sentences.append((sentence.strip(), score))
            
            # Sort by score and take top sentences
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            top_sentences = [s[0] for s in scored_sentences[:max_sentences]]
            
            # Reorder by original position
            summary_sentences = []
            for sentence in sentences:
                if sentence.strip() in top_sentences:
                    summary_sentences.append(sentence.strip())
                    if len(summary_sentences) >= max_sentences:
                        break
            
            summary = '. '.join(summary_sentences)
            if summary and not summary.endswith('.'):
                summary += '.'
            
            return summary if len(summary) > 50 else None
            
        except Exception as e:
            logger.warning(f"Error generating summary: {str(e)}")
            return None
    
    def _calculate_structural_complexity(self, structure: Dict) -> str:
        """Calculate document structural complexity"""
        complexity_score = 0
        
        if structure.get('has_introduction'):
            complexity_score += 1
        if structure.get('has_conclusion'):
            complexity_score += 1
        if structure.get('has_headers'):
            complexity_score += 2
        
        # Add complexity based on paragraph lengths
        para_lengths = structure.get('paragraph_lengths', [])
        if para_lengths:
            avg_para_length = sum(para_lengths) / len(para_lengths)
            if avg_para_length > 100:
                complexity_score += 1
        
        if complexity_score >= 4:
            return 'High'
        elif complexity_score >= 2:
            return 'Medium'
        else:
            return 'Low'
    
    def _assess_extraction_quality(self, analysis_results: Dict) -> str:
        """Assess the quality of text extraction and analysis"""
        if 'error' in analysis_results:
            return 'Poor'
        
        quality_indicators = 0
        
        # Check if basic analysis succeeded
        if analysis_results.get('word_count', 0) > 0:
            quality_indicators += 1
        
        # Check if advanced analysis succeeded
        if analysis_results.get('keywords'):
            quality_indicators += 1
        if analysis_results.get('named_entities'):
            quality_indicators += 1
        if analysis_results.get('sentiment_score') is not None:
            quality_indicators += 1
        
        if quality_indicators >= 4:
            return 'Excellent'
        elif quality_indicators >= 3:
            return 'Good'
        elif quality_indicators >= 2:
            return 'Fair'
        else:
            return 'Poor'
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def _extract_extension(self, filename: str) -> str:
        """Extract file extension from filename"""
        if '.' in filename:
            return filename.split('.')[-1].lower()
        return ''
    
    def _generate_error_metadata(self, error_message: str, file_info: Dict) -> Dict:
        """Generate error metadata when processing fails"""
        return {
            'metadata_schema': {
                'version': self.metadata_schema_version,
                'generator': f"Automated Metadata Generation System v{self.generator_version}",
                'generated_at': datetime.now().isoformat(),
                'metadata_id': str(uuid.uuid4())
            },
            'file_info': self._generate_file_metadata(file_info),
            'error': {
                'message': error_message,
                'timestamp': datetime.now().isoformat(),
                'processing_failed': True
            },
            'content_analysis': {
                'word_count': 0,
                'character_count': 0,
                'language': 'Unknown'
            },
            'semantic_analysis': {
                'keywords': [],
                'named_entities': [],
                'topics': []
            }
        }
    
    def export_metadata(self, metadata: Dict, format_type: str = 'json') -> str:
        """
        Export metadata in specified format
        
        Args:
            metadata: Metadata dictionary
            format_type: Export format ('json', 'xml', 'yaml')
            
        Returns:
            Formatted metadata string
        """
        if format_type.lower() == 'json':
            return json.dumps(metadata, indent=2, ensure_ascii=False, default=str)
        elif format_type.lower() == 'xml':
            return self._dict_to_xml(metadata)
        else:
            return json.dumps(metadata, indent=2, ensure_ascii=False, default=str)
    
    def _dict_to_xml(self, data: Dict, root_name: str = 'metadata') -> str:
        """Convert dictionary to XML format"""
        def dict_to_xml_recursive(d, root):
            xml_str = f"<{root}>\n"
            for key, value in d.items():
                if isinstance(value, dict):
                    xml_str += dict_to_xml_recursive(value, key)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            xml_str += dict_to_xml_recursive(item, key)
                        else:
                            xml_str += f"  <{key}>{item}</{key}>\n"
                else:
                    xml_str += f"  <{key}>{value}</{key}>\n"
            xml_str += f"</{root}>\n"
            return xml_str
        
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{dict_to_xml_recursive(data, root_name)}'
