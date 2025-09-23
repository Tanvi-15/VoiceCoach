from __future__ import annotations
from typing import Dict, Any

class RubricScorer:
    def __init__(self):
        pass

    def score(self, m: Dict[str,Any]) -> Dict[str,Any]:
        return {
            "Clarity": self._clarity(m),
            "Confidence": self._confidence(m),
            "Tone": self._tone(m),
            "Pacing": self._pacing(m),
            "Engagement": self._engagement(m),
            "Cadence": self._cadence(m),
            "Flow": self._flow(m),
        }

    # --- helpers ---
    def _clarity(self, m: Dict[str, Any]) -> Dict[str, Any]:
        """Clarity (20% weight): Voice quality + intelligibility + articulation rate"""
        # Voice quality component (40% of clarity)
        hnr = m.get("hnr_mean_db", 0.0)
        cpps = m.get("cpps_smooth_db", 0.0)
        jitter = m.get("jitter_local", 0.0)
        shimmer = m.get("shimmer_local", 0.0)
        
        # Normalize and combine
        hnr_norm = min(1.0, max(0.0, (hnr - 10) / 15))  # 10-25 dB range
        cpps_norm = min(1.0, max(0.0, (cpps - 5) / 10))  # 5-15 dB range
        jitter_norm = min(1.0, max(0.0, 1.0 - jitter * 20))  # 0-0.05 range
        shimmer_norm = min(1.0, max(0.0, 1.0 - shimmer * 15))  # 0-0.067 range
        
        voice_quality = 0.35 * hnr_norm + 0.35 * cpps_norm - 0.15 * jitter_norm - 0.15 * shimmer_norm
        
        # Intelligibility proxy (40% of clarity)
        articulation_rate = m.get("articulation_rate", 0.0)
        if 3.5 <= articulation_rate <= 5.5:
            intelligibility = 1.0
        elif articulation_rate < 3.5:
            intelligibility = max(0.0, articulation_rate / 3.5)
        else:
            intelligibility = max(0.0, 1.0 - (articulation_rate - 5.5) / 2.0)
        
        # Articulation rate component (20% of clarity)
        articulation_score = min(1.0, max(0.0, (articulation_rate - 2.0) / 3.0))
        
        # Combined clarity score
        clarity_score = 0.4 * voice_quality + 0.4 * intelligibility + 0.2 * articulation_score
        clarity_score = max(0.0, min(1.0, clarity_score))
        
        # Map to 1-5 scale
        score = self._map_to_5_scale(clarity_score)
        
        return {
            "score": score,
            "details": {
                "voice_quality": voice_quality,
                "intelligibility": intelligibility,
                "articulation_rate": articulation_score,
                "combined_score": clarity_score
            },
            "why": f"Voice quality: {voice_quality:.2f}, Intelligibility: {intelligibility:.2f}, Articulation: {articulation_score:.2f}"
        }

    def _confidence(self, m: Dict[str, Any]) -> Dict[str, Any]:
        """Confidence (15% weight): Prosodic confidence + stability"""
        # Prosodic confidence (70% of confidence)
        f0_range = m.get("pitch_range_hz", 0.0)
        intensity_mean = m.get("intensity_mean_db", 0.0)
        filler_ratio = m.get("filler_ratio", 0.0)
        rate_stability = 1.0 - min(1.0, m.get("rate_var_win", 0.0) / 50.0)
        
        # Normalize F0 range
        f0_range_norm = min(1.0, max(0.0, f0_range / 200.0))
        
        # Normalize intensity (avoid rewarding extreme loudness)
        intensity_norm = min(1.0, max(0.0, (intensity_mean - 50) / 20))  # 50-70 dB range
        
        prosodic_confidence = 0.4 * f0_range_norm + 0.3 * intensity_norm + 0.2 * (1.0 - filler_ratio) + 0.1 * rate_stability
        
        # Stability (30% of confidence)
        f0_stability = 1.0 - min(1.0, m.get("f0_var_win", 0.0) / 30.0)
        intensity_stability = 1.0 - min(1.0, m.get("intensity_var_win", 0.0) / 10.0)
        stability = (f0_stability + intensity_stability) / 2.0
        
        # Combined confidence score
        confidence_score = 0.7 * prosodic_confidence + 0.3 * stability
        confidence_score = max(0.0, min(1.0, confidence_score))
        
        # Map to 1-5 scale
        score = self._map_to_5_scale(confidence_score)
        
        return {
            "score": score,
            "details": {
                "prosodic_confidence": prosodic_confidence,
                "stability": stability,
                "combined_score": confidence_score
            },
            "why": f"Prosodic confidence: {prosodic_confidence:.2f}, Stability: {stability:.2f}"
        }

    def _tone(self, m: Dict[str, Any]) -> Dict[str, Any]:
        """Tone (15% weight): Pitch dynamics + intensity + contour"""
        # Pitch dynamics (60% of tone)
        f0_std = m.get("pitch_std_hz", 0.0)
        f0_range = m.get("pitch_range_hz", 0.0)
        
        f0_std_norm = min(1.0, max(0.0, f0_std / 50.0))
        f0_range_norm = min(1.0, max(0.0, f0_range / 200.0))
        
        pitch_score = (f0_std_norm + f0_range_norm) / 2.0
        # Avoid monotone and chaotic variation
        if pitch_score < 0.2:
            pitch_score *= 0.5
        elif pitch_score > 0.8:
            pitch_score = 0.8 + (pitch_score - 0.8) * 0.2
        
        # Intensity dynamics (25% of tone)
        intensity_std = m.get("intensity_std_db", 0.0)
        intensity_score = min(1.0, max(0.0, intensity_std / 8.0))
        
        # Contour shape (15% of tone)
        final_fall_ratio = m.get("final_fall_ratio", 0.0)
        final_rise_ratio = m.get("final_rise_ratio", 0.0)
        contour_score = 0.5 * final_fall_ratio + 0.5 * final_rise_ratio
        
        # Combined tone score
        tone_score = 0.6 * pitch_score + 0.25 * intensity_score + 0.15 * contour_score
        tone_score = max(0.0, min(1.0, tone_score))
        
        # Map to 1-5 scale
        score = self._map_to_5_scale(tone_score)
        
        return {
            "score": score,
            "details": {
                "pitch_dynamics": pitch_score,
                "intensity_dynamics": intensity_score,
                "contour_shape": contour_score,
                "combined_score": tone_score
            },
            "why": f"Pitch dynamics: {pitch_score:.2f}, Intensity: {intensity_score:.2f}, Contour: {contour_score:.2f}"
        }

    def _pacing(self, m: Dict[str, Any]) -> Dict[str, Any]:
        """Pacing (15% weight): WPM + pause quality"""
        # WPM component (60% of pacing)
        wpm = m.get("wpm", 0.0)
        if 140 <= wpm <= 160:
            wpm_score = 1.0
        elif 130 <= wpm <= 170:
            wpm_score = 0.8
        elif 120 <= wpm <= 180:
            wpm_score = 0.6
        else:
            wpm_score = 0.3
        
        # Pause quality component (40% of pacing)
        good_pause_ratio = m.get("good_pause_ratio", 0.0)
        bad_pause_ratio = m.get("bad_pause_ratio", 0.0)
        pause_quality = good_pause_ratio - 0.3 * bad_pause_ratio
        pause_quality = max(0.0, min(1.0, pause_quality))
        
        # Combined pacing score
        pacing_score = 0.6 * wpm_score + 0.4 * pause_quality
        pacing_score = max(0.0, min(1.0, pacing_score))
        
        # Map to 1-5 scale
        score = self._map_to_5_scale(pacing_score)
        
        return {
            "score": score,
            "details": {
                "wpm_score": wpm_score,
                "pause_quality": pause_quality,
                "combined_score": pacing_score
            },
            "why": f"WPM: {wpm:.0f} (score: {wpm_score:.2f}), Pause quality: {pause_quality:.2f}"
        }

    def _engagement(self, m: Dict[str, Any]) -> Dict[str, Any]:
        """Engagement (15% weight): Variability + strategic pausing"""
        # Variability that isn't chaotic (60% of engagement)
        f0_range = m.get("pitch_range_hz", 0.0)
        intensity_std = m.get("intensity_std_db", 0.0)
        
        f0_range_norm = min(1.0, max(0.0, f0_range / 200.0))
        intensity_variation = min(1.0, max(0.0, intensity_std / 8.0))
        
        variability = (f0_range_norm + intensity_variation) / 2.0
        
        # Penalize if too erratic
        short_term_variance = (m.get("f0_var_win", 0.0) + m.get("intensity_var_win", 0.0)) / 2.0
        if short_term_variance > 0.3:
            variability *= 0.7
        
        # Strategic pausing & disfluency (40% of engagement)
        good_pause_ratio = m.get("good_pause_ratio", 0.0)
        bad_pause_ratio = m.get("bad_pause_ratio", 0.0)
        pause_strategy = good_pause_ratio - 0.3 * bad_pause_ratio
        
        # U-curve penalty for fillers (some are good, too many are bad)
        filler_ratio = m.get("filler_ratio", 0.0)
        filler_penalty = self._u_curve_penalty(filler_ratio)
        
        # Combined engagement score
        engagement_score = 0.6 * variability + 0.4 * (pause_strategy + filler_penalty)
        engagement_score = max(0.0, min(1.0, engagement_score))
        
        # Map to 1-5 scale
        score = self._map_to_5_scale(engagement_score)
        
        return {
            "score": score,
            "details": {
                "variability": variability,
                "pause_strategy": pause_strategy,
                "filler_penalty": filler_penalty,
                "combined_score": engagement_score
            },
            "why": f"Variability: {variability:.2f}, Pause strategy: {pause_strategy:.2f}, Fillers: {filler_penalty:.2f}"
        }

    def _cadence(self, m: Dict[str, Any]) -> Dict[str, Any]:
        """Cadence (10% weight): Timing regularity + phrase rhythm"""
        # Timing regularity (60% of cadence)
        # Use pause distribution as proxy for timing regularity
        pause_ratio = m.get("pause_ratio", 0.0)
        regularity = 1.0 - min(1.0, pause_ratio * 2.0)  # Lower pause ratio = more regular
        
        # Phrase rhythm (40% of cadence)
        good_pause_ratio = m.get("good_pause_ratio", 0.0)
        short_pause_ratio = m.get("pause_bins", {}).get("short_rate_per_min", 0.0) / 60.0  # Convert to ratio
        long_pause_ratio = m.get("pause_bins", {}).get("long_rate_per_min", 0.0) / 60.0    # Convert to ratio
        
        rhythm = good_pause_ratio - 0.2 * short_pause_ratio - 0.3 * long_pause_ratio
        rhythm = max(0.0, min(1.0, rhythm))
        
        # Combined cadence score
        cadence_score = 0.6 * regularity + 0.4 * rhythm
        cadence_score = max(0.0, min(1.0, cadence_score))
        
        # Map to 1-5 scale
        score = self._map_to_5_scale(cadence_score)
        
        return {
            "score": score,
            "details": {
                "timing_regularity": regularity,
                "phrase_rhythm": rhythm,
                "combined_score": cadence_score
            },
            "why": f"Timing regularity: {regularity:.2f}, Phrase rhythm: {rhythm:.2f}"
        }

    def _flow(self, m: Dict[str, Any]) -> Dict[str, Any]:
        """Flow (10% weight): Disfluency + coherence"""
        # Disfluency rate (50% of flow)
        repair_rate = m.get("repair_rate", 0.0)
        disfluency_penalty = min(1.0, repair_rate * 3.0)  # Penalize repairs/repeats
        disfluency_score = 1.0 - disfluency_penalty
        
        # Coherence (50% of flow)
        coherence_score = m.get("coherence_score", 0.5)
        
        # Combined flow score
        flow_score = 0.5 * disfluency_score + 0.5 * coherence_score
        flow_score = max(0.0, min(1.0, flow_score))
        
        # Map to 1-5 scale
        score = self._map_to_5_scale(flow_score)
        
        return {
            "score": score,
            "details": {
                "disfluency_score": disfluency_score,
                "coherence_score": coherence_score,
                "combined_score": flow_score
            },
            "why": f"Disfluency: {disfluency_score:.2f}, Coherence: {coherence_score:.2f}"
        }

    # --- Helper methods ---
    
    def _map_to_5_scale(self, score: float) -> int:
        """Map 0-1 score to 1-5 scale"""
        if score >= 0.85: return 5
        elif score >= 0.75: return 4
        elif score >= 0.65: return 3
        elif score >= 0.55: return 2
        else: return 1
    
    def _u_curve_penalty(self, filler_ratio: float) -> float:
        """U-curve penalty for fillers: some are good, too many are bad"""
        # Optimal filler ratio is around 0.02-0.05 (2-5%)
        if 0.02 <= filler_ratio <= 0.05:
            return 1.0  # Optimal
        elif filler_ratio < 0.02:
            return filler_ratio / 0.02  # Too few fillers
        else:
            return max(0.0, 1.0 - (filler_ratio - 0.05) * 2.0)  # Too many fillers
