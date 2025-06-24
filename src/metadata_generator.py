import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
import hashlib
import logging

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
    
    def generate_metadata(self, file_info: Dict, text: str, analysis_results: Dict) -> Dict:
        """
        Generate comprehensive metadata from file info and analysis results
        
        Args:
            file_info: Basic file information
            text: Original text content
            analysis_results: Results from content analysis
            
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
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating metadata: {str(e)}")
            return self._generate_error_metadata(str(e), file_info)
    
    def _generate_file_metadata(self, file_info: Dict) -> Dict:
        """Generate file-specific metadata"""
        return {
            'filename': file_info.get('filename', 'unknown'),
            'file_size': file_info.get('file_size', 0),
            'file_size_human': self._format_file_size(file_info.get('file_size', 0)),
            'file_type': file_info.get('file_type', 'unknown'),
            'mime_type': file_info.get('file_type', 'unknown'),
            'upload_timestamp': file_info.get('upload_timestamp', datetime.now().isoformat()),
            'file_hash': file_info.get('file_hash', 'unknown'),
            'file_extension': self._extract_extension(file_info.get('filename', ''))
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
