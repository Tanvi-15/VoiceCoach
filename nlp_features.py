from __future__ import annotations
import re
import pyphen
from typing import Dict, Any, List, Tuple
import numpy as np

# Optional imports with fallbacks
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_NLP_DEPS = True
except ImportError:
    HAS_NLP_DEPS = False
    print("Warning: sentence-transformers not available. NLP features will be limited.")

class NLPFeatures:
    def __init__(self):
        self.dic = pyphen.Pyphen(lang='en')
        # Load sentence transformer for coherence scoring
        self.sentence_model = None
        if HAS_NLP_DEPS:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.sentence_model = None
        
        # Repair/disfluency patterns
        self.repair_patterns = [
            r'\bI\s+I\b',           # "I I think"
            r'\bthe\s+the\b',       # "the the thing"
            r'\bI\s+mean\b',        # "I mean"
            r'\bno,\s*I\s+mean\b',  # "no, I mean"
            r'\bwell,\s*I\s+mean\b', # "well, I mean"
            r'\byou\s+know\s+what\s+I\s+mean\b', # "you know what I mean"
            r'\bactually,\s*I\s+mean\b', # "actually, I mean"
            r'\bI\s+guess\s+I\s+mean\b', # "I guess I mean"
            r'\bso\s+I\s+mean\b',   # "so I mean"
            r'\bwhat\s+I\s+mean\s+is\b', # "what I mean is"
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.repair_patterns]

    def count_syllables(self, word: str) -> int:
        """Count syllables in a word using pyphen."""
        try:
            hyphenated = self.dic.inserted(word.lower())
            return len([syl for syl in hyphenated.split('-') if syl.strip()])
        except Exception:
            # Fallback: simple vowel counting heuristic
            vowels = 'aeiouy'
            word = word.lower()
            syllable_count = 0
            prev_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = is_vowel
            
            # Handle silent 'e' at the end
            if word.endswith('e') and syllable_count > 1:
                syllable_count -= 1
                
            return max(1, syllable_count)

    def calculate_articulation_rate(self, text: str, duration_sec: float, pause_ratio: float) -> Dict[str, Any]:
        """Calculate articulation rate (syllables per second excluding pauses)."""
        words = [w for w in re.findall(r"[A-Za-z']+", text.lower())]
        
        if not words or duration_sec <= 0:
            return {
                "articulation_rate": 0.0,
                "syllable_count": 0,
                "speech_time_sec": 0.0,
                "target_range_met": False
            }
        
        # Count syllables
        total_syllables = sum(self.count_syllables(word) for word in words)
        
        # Calculate speech time (excluding pauses)
        speech_time = duration_sec * (1.0 - pause_ratio)
        
        # Articulation rate: syllables per second of speech
        articulation_rate = total_syllables / speech_time if speech_time > 0 else 0.0
        
        # Target range: 3.5-5.5 syllables per second
        target_range_met = 3.5 <= articulation_rate <= 5.5
        
        return {
            "articulation_rate": articulation_rate,
            "syllable_count": total_syllables,
            "speech_time_sec": speech_time,
            "target_range_met": target_range_met
        }

    def detect_repairs(self, text: str) -> Dict[str, Any]:
        """Detect repair patterns and disfluencies in text."""
        text_lower = text.lower()
        
        repair_count = 0
        repair_details = []
        
        for i, pattern in enumerate(self.compiled_patterns):
            matches = pattern.findall(text_lower)
            if matches:
                repair_count += len(matches)
                repair_details.append({
                    "pattern": self.repair_patterns[i],
                    "matches": len(matches),
                    "examples": matches[:3]  # First 3 examples
                })
        
        # Calculate repair rate
        words = [w for w in re.findall(r"[A-Za-z']+", text_lower)]
        repair_rate = repair_count / max(len(words), 1)
        
        return {
            "repair_count": repair_count,
            "repair_rate": repair_rate,
            "repair_details": repair_details
        }

    def calculate_coherence(self, text: str) -> Dict[str, Any]:
        """Calculate topical coherence between sentences."""
        if not HAS_NLP_DEPS or not self.sentence_model:
            return {
                "coherence_score": 0.5,
                "sentence_count": 0,
                "avg_similarity": 0.0,
                "coherence_details": "NLP dependencies not available"
            }
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return {
                "coherence_score": 0.5,
                "sentence_count": len(sentences),
                "avg_similarity": 0.0,
                "coherence_details": "Not enough sentences for coherence analysis"
            }
        
        try:
            # Get sentence embeddings
            embeddings = self.sentence_model.encode(sentences)
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(sim)
            
            avg_similarity = float(np.mean(similarities))
            
            # Map similarity to coherence score (0-1)
            # Higher similarity = higher coherence
            coherence_score = min(1.0, max(0.0, avg_similarity))
            
            return {
                "coherence_score": coherence_score,
                "sentence_count": len(sentences),
                "avg_similarity": avg_similarity,
                "coherence_details": f"Average similarity: {avg_similarity:.3f}"
            }
            
        except Exception as e:
            return {
                "coherence_score": 0.5,
                "sentence_count": len(sentences),
                "avg_similarity": 0.0,
                "coherence_details": f"Error in coherence calculation: {str(e)}"
            }

    def extract_all_features(self, text: str, duration_sec: float, pause_ratio: float) -> Dict[str, Any]:
        """Extract all NLP features at once."""
        articulation = self.calculate_articulation_rate(text, duration_sec, pause_ratio)
        repairs = self.detect_repairs(text)
        coherence = self.calculate_coherence(text)
        
        return {
            "articulation_rate": articulation["articulation_rate"],
            "syllable_count": articulation["syllable_count"],
            "speech_time_sec": articulation["speech_time_sec"],
            "target_range_met": articulation["target_range_met"],
            "repair_count": repairs["repair_count"],
            "repair_rate": repairs["repair_rate"],
            "repair_details": repairs["repair_details"],
            "coherence_score": coherence["coherence_score"],
            "sentence_count": coherence["sentence_count"],
            "avg_similarity": coherence["avg_similarity"],
            "coherence_details": coherence["coherence_details"]
        }
