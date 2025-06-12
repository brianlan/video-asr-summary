"""Content classification for determining appropriate analysis templates."""

import re
from typing import Dict, List
from video_asr_summary.analysis import ContentClassifier, ContentType


class KeywordBasedClassifier(ContentClassifier):
    """Simple keyword-based content classifier."""
    
    def __init__(self):
        """Initialize with keyword mappings for each content type."""
        self.keywords: Dict[ContentType, List[str]] = {
            ContentType.POLITICAL_COMMENTARY: [
                "politics", "political", "government", "policy", "election", 
                "candidate", "vote", "democracy", "congress", "senate", 
                "president", "administration", "partisan", "conservative", 
                "liberal", "republican", "democrat", "campaign", "legislation",
                "taxes", "healthcare policy", "immigration", "foreign policy"
            ],
            ContentType.NEWS_REPORT: [
                "breaking news", "report", "reporter", "according to", 
                "sources say", "investigation", "developing story", 
                "update", "confirmed", "officials", "statement", 
                "press conference", "announcement", "incident", "event",
                "authorities", "spokesperson", "witness", "coverage"
            ],
            ContentType.TECHNICAL_REVIEW: [
                "review", "test", "performance", "benchmark", "specification",
                "feature", "technology", "technical", "analysis", "comparison",
                "evaluation", "assessment", "methodology", "results", "data",
                "measurement", "testing", "specifications", "hardware", 
                "software", "application", "system", "device", "product"
            ],
            ContentType.BOOK_SECTION: [
                "chapter", "section", "book", "author", "theory", "concept",
                "research", "study", "literature", "academic", "textbook",
                "publication", "journal", "reference", "citation", "evidence",
                "hypothesis", "framework", "model", "principle", "conclusion",
                "findings", "methodology", "abstract", "introduction", "discussion"
            ],
            ContentType.PERSONAL_CASUAL_TALK: [
                "I think", "I feel", "I believe", "in my opinion", "personally",
                "my experience", "I remember", "I've noticed", "I guess", 
                "you know", "like", "actually", "honestly", "basically",
                "I mean", "sort of", "kind of", "probably", "maybe",
                "my friend", "my family", "my life", "happened to me", "I've seen"
            ]
        }
    
    def classify(self, text: str) -> ContentType:
        """Classify content type from text using keyword matching."""
        text_lower = text.lower()
        
        # Count keyword matches for each content type
        scores: Dict[ContentType, int] = {}
        
        for content_type, keywords in self.keywords.items():
            score = 0
            for keyword in keywords:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                score += matches
            scores[content_type] = score
        
        # Return the content type with the highest score
        if max(scores.values()) == 0:
            return ContentType.GENERAL  # Default if no keywords match
            
        return max(scores, key=lambda x: scores[x])
    
    def add_keywords(self, content_type: ContentType, keywords: List[str]) -> None:
        """Add keywords for a content type."""
        if content_type not in self.keywords:
            self.keywords[content_type] = []
        self.keywords[content_type].extend(keywords)
    
    def get_classification_confidence(self, text: str) -> Dict[ContentType, float]:
        """Get classification confidence scores for all content types."""
        text_lower = text.lower()
        word_count = len(text_lower.split())
        
        if word_count == 0:
            return {content_type: 0.0 for content_type in ContentType}
        
        scores: Dict[ContentType, float] = {}
        
        for content_type, keywords in self.keywords.items():
            match_count = 0
            for keyword in keywords:
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                match_count += matches
            
            # Normalize by text length to get confidence score (0.0 to 1.0)
            confidence = min(match_count / word_count, 1.0)
            scores[content_type] = confidence
        
        # Add general as baseline
        scores[ContentType.GENERAL] = 0.1  # Low baseline confidence
        
        return scores
