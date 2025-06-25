import re
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except:
    pass

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

class ContentAnalyzer:
    """
    Content analysis module for NLP processing and semantic analysis
    """
    
    def __init__(self):
        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        except:
            logger.warning("VADER sentiment analyzer not available")
            self.sentiment_analyzer = None
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Initialize YAKE keyword extractor
        self.kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,  # n-gram size
            dedupLim=0.7,
            top=20
        )
    
    def analyze_content(self, text: str) -> Dict:
        """
        Perform comprehensive content analysis
        
        Args:
            text: Input text for analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        results = {}
        
        try:
            # Basic text statistics
            results.update(self._calculate_basic_stats(text))
            
            # Language detection
            results['language'] = self._detect_language(text)
            
            # Readability metrics
            results.update(self._calculate_readability_metrics(text))
            
            # Sentiment analysis
            results.update(self._analyze_sentiment(text))
            
            # Word frequency analysis
            results['word_frequency'] = self._analyze_word_frequency(text)
            
            # Extract key phrases and keywords
            results['keywords'] = self._extract_keywords(text)
            results['key_phrases'] = self._extract_key_phrases(text)
            
            # Named entity recognition
            results['named_entities'] = self._extract_named_entities(text)
            
            # Topic extraction
            results['topics'] = self._extract_topics(text)
            
            # Document structure analysis
            results['structure'] = self._analyze_structure(text)
            
        except Exception as e:
            logger.error(f"Error in content analysis: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _calculate_basic_stats(self, text: str) -> Dict:
        """Calculate basic text statistics"""
        words = word_tokenize(text.lower())
        sentences = sent_tokenize(text)
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        return {
            'word_count': len(words),
            'character_count': len(text),
            'character_count_no_spaces': len(text.replace(' ', '')),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'average_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'average_sentence_length': np.mean([len(s) for s in sentences]) if sentences else 0
        }
    
    def _detect_language(self, text: str) -> str:
        """
        Detect document language (simplified version)
        """
        # Simple heuristic-based language detection
        # In a production system, you might want to use langdetect library
        
        # Common English words
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = set(word_tokenize(text.lower())[:100])  # Check first 100 words
        english_word_count = len(words.intersection(english_words))
        
        if english_word_count >= 5:
            return "English"
        else:
            return "Unknown"
    
    def _calculate_readability_metrics(self, text: str) -> Dict:
        """Calculate readability scores"""
        try:
            flesch_score = flesch_reading_ease(text)
            fk_grade = flesch_kincaid_grade(text)
            
            return {
                'readability_score': flesch_score,
                'flesch_kincaid_grade': fk_grade,
                'reading_level': self._interpret_readability(flesch_score)
            }
        except:
            return {
                'readability_score': 0,
                'flesch_kincaid_grade': 0,
                'reading_level': "Unknown"
            }
    
    def _interpret_readability(self, score: float) -> str:
        """Interpret Flesch reading ease score"""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze document sentiment"""
        if not self.sentiment_analyzer:
            return {'sentiment_score': 0, 'sentiment_label': 'Neutral'}
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            compound_score = scores['compound']
            
            if compound_score >= 0.05:
                label = 'Positive'
            elif compound_score <= -0.05:
                label = 'Negative'
            else:
                label = 'Neutral'
            
            return {
                'sentiment_score': compound_score,
                'sentiment_label': label,
                'sentiment_scores': scores
            }
        except:
            return {'sentiment_score': 0, 'sentiment_label': 'Neutral'}
    
    def _analyze_word_frequency(self, text: str) -> Dict:
        """Analyze word frequency distribution"""
        words = word_tokenize(text.lower())
        # Remove stopwords and punctuation
        words = [w for w in words if w.isalnum() and w not in self.stop_words and len(w) > 2]
        
        word_freq = Counter(words)
        return dict(word_freq.most_common(50))  # Top 50 words
    
    def _extract_keywords(self, text: str, num_keywords: int = 15) -> List[str]:
        """Extract keywords using YAKE algorithm"""
        try:
            keywords = self.kw_extractor.extract_keywords(text)
            
            # YAKE returns (keyword, score) tuples - keyword is first, score is second
            extracted = []
            for item in keywords[:num_keywords]:
                if isinstance(item, tuple) and len(item) >= 2:
                    # item[0] is keyword, item[1] is score
                    keyword = str(item[0]).strip()
                    if keyword and len(keyword) > 1:  # Filter out single characters
                        extracted.append(keyword)
                elif isinstance(item, str):
                    keyword = item.strip()
                    if keyword and len(keyword) > 1:
                        extracted.append(keyword)
            
            return extracted if extracted else self._extract_keywords_tfidf(text, num_keywords)
            
        except Exception as e:
            logger.warning(f"YAKE extraction failed: {str(e)}")
            # Fallback to TF-IDF if YAKE fails
            return self._extract_keywords_tfidf(text, num_keywords)
    
    def _extract_keywords_tfidf(self, text: str, num_keywords: int = 15) -> List[str]:
        """Extract keywords using TF-IDF as fallback"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=num_keywords,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Sort by TF-IDF score
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [kw[0] for kw in keyword_scores]
        except:
            return []
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases using NLTK POS tagging"""
        phrases = []
        
        try:
            # Extract phrases using POS tagging patterns
            sentences = sent_tokenize(text)
            
            for sentence in sentences[:15]:  # First 15 sentences
                words = word_tokenize(sentence)
                pos_tags = pos_tag(words)
                
                # Pattern 1: Adjective + Noun
                for i in range(len(pos_tags) - 1):
                    if (pos_tags[i][1] in ['JJ', 'JJR', 'JJS'] and 
                        pos_tags[i+1][1] in ['NN', 'NNS', 'NNP', 'NNPS']):
                        phrase = f"{pos_tags[i][0]} {pos_tags[i+1][0]}"
                        if len(phrase) > 3:
                            phrases.append(phrase)
                
                # Pattern 2: Noun + Noun (compound nouns)
                for i in range(len(pos_tags) - 1):
                    if (pos_tags[i][1] in ['NN', 'NNS', 'NNP', 'NNPS'] and 
                        pos_tags[i+1][1] in ['NN', 'NNS', 'NNP', 'NNPS']):
                        phrase = f"{pos_tags[i][0]} {pos_tags[i+1][0]}"
                        if len(phrase) > 3:
                            phrases.append(phrase)
                
                # Pattern 3: Adjective + Adjective + Noun
                for i in range(len(pos_tags) - 2):
                    if (pos_tags[i][1] in ['JJ', 'JJR', 'JJS'] and 
                        pos_tags[i+1][1] in ['JJ', 'JJR', 'JJS'] and
                        pos_tags[i+2][1] in ['NN', 'NNS', 'NNP', 'NNPS']):
                        phrase = f"{pos_tags[i][0]} {pos_tags[i+1][0]} {pos_tags[i+2][0]}"
                        if len(phrase) > 5:
                            phrases.append(phrase)
            
        except Exception as e:
            logger.warning(f"Error extracting key phrases: {str(e)}")
        
        # Remove duplicates and return top phrases
        unique_phrases = list(dict.fromkeys(phrases))
        return unique_phrases[:20]
    
    def _extract_named_entities(self, text: str) -> List[Dict]:
        """Extract named entities using NLTK's named entity chunker"""
        entities = []
        
        try:
            sentences = sent_tokenize(text)
            
            for sentence in sentences[:10]:  # Process first 10 sentences
                tokens = word_tokenize(sentence)
                pos_tags = pos_tag(tokens)
                
                # Use NLTK's named entity chunker
                try:
                    chunks = ne_chunk(pos_tags, binary=False)
                    
                    for chunk in chunks:
                        if hasattr(chunk, 'label'):
                            entity_text = ' '.join([token for token, pos in chunk.leaves()])
                            entity_label = chunk.label()
                            
                            # Map NLTK labels to more readable descriptions
                            label_descriptions = {
                                'PERSON': 'Person',
                                'ORGANIZATION': 'Organization',
                                'GPE': 'Geopolitical Entity',
                                'LOCATION': 'Location',
                                'FACILITY': 'Facility',
                                'GSP': 'Geopolitical Entity'
                            }
                            
                            entities.append({
                                'text': entity_text,
                                'label': entity_label,
                                'description': label_descriptions.get(entity_label, entity_label),
                                'start': 0,  # NLTK doesn't provide character positions easily
                                'end': 0
                            })
                            
                except Exception as e:
                    # Fallback: simple pattern-based entity extraction
                    entities.extend(self._extract_entities_pattern_based(sentence))
            
            # Remove duplicates
            seen = set()
            unique_entities = []
            for entity in entities:
                key = (entity['text'].lower(), entity['label'])
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)
            
            return unique_entities[:30]  # Top 30 entities
            
        except Exception as e:
            logger.warning(f"Error extracting named entities: {str(e)}")
            return []
    
    def _extract_entities_pattern_based(self, text: str) -> List[Dict]:
        """Simple pattern-based entity extraction as fallback"""
        entities = []
        
        # Simple patterns for common entities
        patterns = {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'URL': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'DATE': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'MONEY': r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        }
        
        for label, pattern in patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'label': label,
                    'description': label.title(),
                    'start': match.start(),
                    'end': match.end()
                })
        
        return entities
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics using keyword clustering and rules"""
        topics = []
        
        try:
            # Get keywords first
            keywords = self._extract_keywords(text, 20)
            if not keywords:
                keywords = []
            
            # Simple topic identification based on keyword patterns
            topic_patterns = {
                'Technology': ['software', 'computer', 'technology', 'digital', 'system', 'data', 'network', 'algorithm', 'programming', 'code'],
                'Business': ['business', 'company', 'market', 'financial', 'revenue', 'profit', 'strategy', 'management', 'corporate', 'sales'],
                'Health': ['health', 'medical', 'patient', 'treatment', 'disease', 'doctor', 'medicine', 'healthcare', 'clinical', 'therapy'],
                'Education': ['education', 'student', 'school', 'learning', 'academic', 'university', 'research', 'study', 'teaching', 'knowledge'],
                'Legal': ['legal', 'law', 'court', 'contract', 'regulation', 'compliance', 'policy', 'attorney', 'justice', 'rights'],
                'Science': ['research', 'study', 'analysis', 'scientific', 'experiment', 'data', 'results', 'theory', 'hypothesis', 'discovery'],
                'Finance': ['finance', 'investment', 'banking', 'money', 'financial', 'economic', 'budget', 'cost', 'price', 'funding'],
                'Marketing': ['marketing', 'advertising', 'brand', 'customer', 'promotion', 'campaign', 'social media', 'digital marketing']
            }
            
            # Check which topics match the keywords
            for topic, pattern_words in topic_patterns.items():
                matches = sum(1 for kw in keywords if isinstance(kw, str) and any(pw in kw.lower() for pw in pattern_words))
                if matches >= 2:  # At least 2 matching keywords
                    topics.append(topic)
            
            # If no topics found, try to infer from content
            if not topics:
                text_lower = text.lower()
                for topic, pattern_words in topic_patterns.items():
                    if sum(1 for pw in pattern_words if pw in text_lower) >= 3:
                        topics.append(topic)
                        
        except Exception as e:
            logger.warning(f"Error extracting topics: {str(e)}")
        
        return topics[:5]  # Top 5 topics
    
    def _analyze_structure(self, text: str) -> Dict:
        """Analyze document structure"""
        sentences = sent_tokenize(text)
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        
        structure = {
            'has_introduction': self._has_introduction(text),
            'has_conclusion': self._has_conclusion(text),
            'has_headers': self._has_headers(text),
            'paragraph_lengths': [len(p.split()) for p in paragraphs[:10]],
            'sentence_lengths': [len(s.split()) for s in sentences[:20]]
        }
        
        return structure
    
    def _has_introduction(self, text: str) -> bool:
        """Check if document has introduction"""
        intro_words = ['introduction', 'overview', 'abstract', 'summary', 'background']
        first_paragraph = text[:500].lower()
        return any(word in first_paragraph for word in intro_words)
    
    def _has_conclusion(self, text: str) -> bool:
        """Check if document has conclusion"""
        conclusion_words = ['conclusion', 'summary', 'recommendations', 'final', 'end', 'results']
        last_paragraph = text[-500:].lower()
        return any(word in last_paragraph for word in conclusion_words)
    
    def _has_headers(self, text: str) -> bool:
        """Check if document has headers/sections"""
        lines = text.split('\n')
        potential_headers = 0
        
        for line in lines:
            line = line.strip()
            if (len(line) < 50 and len(line) > 3 and 
                (line.isupper() or line.istitle() or re.match(r'^\d+\.', line))):
                potential_headers += 1
        
        return potential_headers >= 3