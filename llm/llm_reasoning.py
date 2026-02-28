"""
llm_reasoning.py - Advanced reasoning module for VideoHighlighter.

Connects facts across the analysis data to infer relationships.
Now optimized for action-only cache files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime
import numpy as np

# ---------------------------------------------------------------------------
# Data structures for reasoning
# ---------------------------------------------------------------------------

@dataclass
class InferredFact:
    """A fact inferred by connecting multiple detections."""
    description: str
    entities: List[str]
    timestamp: float
    confidence: float
    reasoning_chain: List[str]  # How we derived this fact
    evidence: Dict[str, Any]  # Raw data that supports it
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "entities": self.entities,
            "timestamp": self.timestamp,
            "timestamp_str": f"{int(self.timestamp)//60}:{int(self.timestamp)%60:02d}",
            "confidence": self.confidence,
            "reasoning_chain": self.reasoning_chain,
            "evidence": self.evidence
        }


@dataclass
class ActionCluster:
    """A cluster of related actions occurring close in time."""
    start_time: float
    end_time: float
    actions: List[str]
    action_count: int
    density: float  # actions per second
    description: str


# ---------------------------------------------------------------------------
# Reasoning Engine
# ---------------------------------------------------------------------------

class VideoReasoningEngine:
    """
    Connects facts across video analysis to infer higher-level relationships.
    Optimized for action-only cache files.
    """
    
    # Action categories for grouping
    ACTION_CATEGORIES = {
        # Movement actions
        "walking": "movement",
        "running": "movement",
        "jumping": "movement",
        "entering": "movement",
        "exiting": "movement",
        "approaching": "movement",
        "leaving": "movement",
        
        # Interaction actions
        "talking": "interaction",
        "speaking": "interaction",
        "pointing": "interaction",
        "waving": "interaction",
        "handshaking": "interaction",
        "hugging": "interaction",
        "kissing": "interaction",
        
        # Combat actions
        "punching": "combat",
        "kicking": "combat",
        "fighting": "combat",
        "blocking": "combat",
        "shooting": "combat",
        
        # Object manipulation
        "picking up": "manipulation",
        "putting down": "manipulation",
        "holding": "manipulation",
        "carrying": "manipulation",
        "throwing": "manipulation",
        "catching": "manipulation",
        "opening": "manipulation",
        "closing": "manipulation",
        
        # Sports/activities
        "dribbling": "sports",
        "kicking ball": "sports",
        "throwing ball": "sports",
        "catching ball": "sports",
        "hitting": "sports",
        "swinging": "sports",
        
        # Stationary
        "standing": "stationary",
        "sitting": "stationary",
        "lying": "stationary",
        "waiting": "stationary",
        "looking": "stationary",
    }
    
    # Action sequences that form meaningful patterns
    ACTION_PATTERNS = [
        {
            "name": "arrival",
            "pattern": ["entering", "walking", "standing"],
            "description": "Someone arrives and stops"
        },
        {
            "name": "departure", 
            "pattern": ["standing", "walking", "exiting"],
            "description": "Someone leaves after being present"
        },
        {
            "name": "interaction",
            "pattern": ["approaching", "talking", "leaving"],
            "description": "Social interaction sequence"
        },
        {
            "name": "combat",
            "pattern": ["approaching", "fighting", "exiting"],
            "description": "Fight sequence"
        },
        {
            "name": "sports_play",
            "pattern": ["running", "jumping", "throwing"],
            "description": "Athletic activity"
        }
    ]
    
    def __init__(self, analysis_data: Dict[str, Any], video_path: str = ""):
        self.data = analysis_data
        self.video_path = video_path
        self._build_indexes()
        self.fact_cache: Dict[float, List[InferredFact]] = {}
        self.action_clusters: List[ActionCluster] = []
        
    def _build_indexes(self):
        """Build fast lookup indexes for reasoning."""
        # Index actions by timestamp
        self.actions_by_time = defaultdict(list)
        actions_raw = self.data.get("actions", [])
        
        for act in actions_raw:
            ts = act.get("timestamp", 0)
            self.actions_by_time[ts].append({
                "name": act.get("action_name", "unknown").lower(),
                "confidence": act.get("confidence", 0.5),
                "timestamp": ts
            })
        
        # Sort timestamps
        self.all_timestamps = sorted(self.actions_by_time.keys())
        
        # Build action timeline
        self._build_action_timeline()
        
        # Find action clusters
        self._find_action_clusters()
    
    def _build_action_timeline(self):
        """Build a continuous timeline of actions."""
        self.action_timeline = []
        for ts in sorted(self.actions_by_time.keys()):
            actions = [a["name"] for a in self.actions_by_time[ts]]
            self.action_timeline.append({
                "timestamp": ts,
                "actions": actions,
                "count": len(actions)
            })
    
    def _find_action_clusters(self, min_density: float = 0.5):
        """Find clusters of high action density."""
        if len(self.all_timestamps) < 2:
            return
        
        clusters = []
        current_cluster = {
            "start": self.all_timestamps[0],
            "end": self.all_timestamps[0],
            "actions": set(),
            "count": 0
        }
        
        for i in range(1, len(self.all_timestamps)):
            time_gap = self.all_timestamps[i] - self.all_timestamps[i-1]
            
            # If gap is small, extend current cluster
            if time_gap < 2.0:  # Less than 2 seconds between actions
                current_cluster["end"] = self.all_timestamps[i]
                for a in self.actions_by_time[self.all_timestamps[i]]:
                    current_cluster["actions"].add(a["name"])
                current_cluster["count"] += 1
            else:
                # End current cluster if it has enough density
                duration = current_cluster["end"] - current_cluster["start"]
                if duration > 0:
                    density = current_cluster["count"] / duration
                    if density >= min_density:
                        clusters.append(ActionCluster(
                            start_time=current_cluster["start"],
                            end_time=current_cluster["end"],
                            actions=sorted(list(current_cluster["actions"])),
                            action_count=current_cluster["count"],
                            density=density,
                            description=self._describe_cluster(
                                list(current_cluster["actions"]), density
                            )
                        ))
                
                # Start new cluster
                current_cluster = {
                    "start": self.all_timestamps[i],
                    "end": self.all_timestamps[i],
                    "actions": set([a["name"] for a in self.actions_by_time[self.all_timestamps[i]]]),
                    "count": 1
                }
        
        # Add last cluster
        duration = current_cluster["end"] - current_cluster["start"]
        if duration > 0:
            density = current_cluster["count"] / duration
            if density >= min_density:
                clusters.append(ActionCluster(
                    start_time=current_cluster["start"],
                    end_time=current_cluster["end"],
                    actions=sorted(list(current_cluster["actions"])),
                    action_count=current_cluster["count"],
                    density=density,
                    description=self._describe_cluster(
                        list(current_cluster["actions"]), density
                    )
                ))
        
        self.action_clusters = sorted(clusters, key=lambda c: c.density, reverse=True)
    
    def _describe_cluster(self, actions: List[str], density: float) -> str:
        """Generate a human-readable description of an action cluster."""
        # Get unique action categories in this cluster
        categories = set()
        for action in actions:
            cat = self.ACTION_CATEGORIES.get(action, "other")
            categories.add(cat)
        
        if len(actions) == 1:
            return f"Prolonged {actions[0]} activity"
        elif len(categories) == 1:
            cat = list(categories)[0]
            if density > 2.0:
                return f"Intense {cat} sequence"
            else:
                return f"Sustained {cat} activity"
        else:
            if density > 2.0:
                return "Rapid action sequence with multiple activities"
            else:
                return "Mixed activity period"
    
    # -------------------------------------------------------------------
    # Core reasoning methods
    # -------------------------------------------------------------------
    
    def reason_about_timestamp(self, timestamp: float, 
                               window_seconds: float = 3.0,
                               use_cache: bool = True) -> List[InferredFact]:
        """
        Infer facts around a given timestamp.
        
        Args:
            timestamp: Center time to analyze
            window_seconds: Window around timestamp to consider
            use_cache: Use cached facts if available
            
        Returns:
            List of inferred facts, sorted by confidence
        """
        # Check cache
        if use_cache and timestamp in self.fact_cache:
            return self.fact_cache[timestamp]
        
        facts = []
        
        # Find actions in window
        start_ts = timestamp - window_seconds
        end_ts = timestamp + window_seconds
        
        nearby_actions = []
        for ts in self.all_timestamps:
            if start_ts <= ts <= end_ts:
                nearby_actions.extend(self.actions_by_time.get(ts, []))
        
        if not nearby_actions:
            return []
        
        # Run reasoning rules
        facts.extend(self._reason_action_patterns(nearby_actions, timestamp))
        facts.extend(self._reason_action_transitions(nearby_actions, timestamp))
        facts.extend(self._reason_action_intensity(nearby_actions, timestamp))
        
        # Sort by confidence
        facts.sort(key=lambda f: f.confidence, reverse=True)
        
        # Cache results
        self.fact_cache[timestamp] = facts
        
        return facts
    
    def reason_over_time_range(self, start_time: float, end_time: float,
                               interval: float = 5.0) -> Dict[float, List[InferredFact]]:
        """
        Reason over a range of timestamps.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            interval: Interval between analyses
            
        Returns:
            Dictionary mapping timestamps to lists of facts
        """
        results = {}
        current = start_time
        while current <= end_time:
            results[current] = self.reason_about_timestamp(current)
            current += interval
        return results
    
    # -------------------------------------------------------------------
    # Action-only reasoning rules
    # -------------------------------------------------------------------
    
    def _reason_action_patterns(self, actions: List[Dict],
                                 timestamp: float) -> List[InferredFact]:
        """
        Detect known action patterns.
        """
        facts = []
        
        # Get unique action names in window
        action_names = list(set([a["name"] for a in actions]))
        
        # Check against known patterns
        for pattern in self.ACTION_PATTERNS:
            # Check if pattern actions are present
            matches = sum(1 for pa in pattern["pattern"] if pa in action_names)
            match_ratio = matches / len(pattern["pattern"])
            
            if match_ratio >= 0.6:  # At least 60% match
                confidence = match_ratio * 0.8
                
                fact = InferredFact(
                    description=f"Detected {pattern['name']} pattern: {pattern['description']}",
                    entities=["action_sequence"],
                    timestamp=timestamp,
                    confidence=confidence,
                    reasoning_chain=[
                        f"Found {matches}/{len(pattern['pattern'])} actions in pattern",
                        f"Actions detected: {', '.join(action_names[:5])}"
                    ],
                    evidence={
                        "pattern": pattern["name"],
                        "detected_actions": action_names,
                        "timestamp": timestamp
                    }
                )
                facts.append(fact)
        
        return facts
    
    def _reason_action_transitions(self, actions: List[Dict],
                                    timestamp: float) -> List[InferredFact]:
        """
        Detect meaningful action transitions.
        """
        facts = []
        
        # Sort actions by confidence
        sorted_actions = sorted(actions, key=lambda a: a["confidence"], reverse=True)
        
        if len(sorted_actions) >= 2:
            # Look for action pairs that indicate transitions
            action_pairs = []
            for i in range(min(3, len(sorted_actions))):
                for j in range(i+1, min(4, len(sorted_actions))):
                    a1 = sorted_actions[i]["name"]
                    a2 = sorted_actions[j]["name"]
                    
                    # Check for meaningful transitions
                    if self._is_meaningful_transition(a1, a2):
                        action_pairs.append((a1, a2))
            
            if action_pairs:
                pair = action_pairs[0]
                fact = InferredFact(
                    description=f"Transition from {pair[0]} to {pair[1]}",
                    entities=[pair[0], pair[1]],
                    timestamp=timestamp,
                    confidence=0.7,
                    reasoning_chain=[
                        f"Detected {pair[0]} and {pair[1]} in close succession",
                        "This indicates a change in activity"
                    ],
                    evidence={
                        "actions": [{"name": a["name"], "conf": a["confidence"]} 
                                  for a in sorted_actions[:4]],
                        "timestamp": timestamp
                    }
                )
                facts.append(fact)
        
        return facts
    
    def _reason_action_intensity(self, actions: List[Dict],
                                  timestamp: float) -> List[InferredFact]:
        """
        Reason about action intensity and density.
        """
        facts = []
        
        action_count = len(actions)
        
        if action_count >= 5:
            fact = InferredFact(
                description=f"High activity moment with {action_count} simultaneous actions",
                entities=["high_intensity"],
                timestamp=timestamp,
                confidence=min(0.9, 0.5 + (action_count * 0.05)),
                reasoning_chain=[
                    f"{action_count} actions detected simultaneously",
                    "This is a peak of activity"
                ],
                evidence={
                    "action_count": action_count,
                    "actions": [a["name"] for a in actions[:10]],
                    "timestamp": timestamp
                }
            )
            facts.append(fact)
        elif 3 <= action_count < 5:
            fact = InferredFact(
                description=f"Moderate activity with {action_count} simultaneous actions",
                entities=["moderate_intensity"],
                timestamp=timestamp,
                confidence=0.6,
                reasoning_chain=[
                    f"{action_count} actions occurring together"
                ],
                evidence={
                    "action_count": action_count,
                    "actions": [a["name"] for a in actions],
                    "timestamp": timestamp
                }
            )
            facts.append(fact)
        
        return facts
    
    def _is_meaningful_transition(self, action1: str, action2: str) -> bool:
        """Check if a transition between two actions is meaningful."""
        # Define meaningful transitions
        meaningful = [
            ("standing", "walking"),
            ("walking", "running"),
            ("sitting", "standing"),
            ("lying", "sitting"),
            ("entering", "walking"),
            ("walking", "exiting"),
            ("approaching", "talking"),
            ("talking", "leaving"),
            ("walking", "jumping"),
            ("running", "jumping"),
        ]
        
        return (action1, action2) in meaningful or (action2, action1) in meaningful
    
    # -------------------------------------------------------------------
    # Public API methods
    # -------------------------------------------------------------------
    
    def get_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about actions in the video."""
        all_actions = []
        for actions in self.actions_by_time.values():
            all_actions.extend([a["name"] for a in actions])
        
        from collections import Counter
        action_counts = Counter(all_actions)
        
        return {
            "total_actions": len(all_actions),
            "unique_actions": len(action_counts),
            "timestamps_with_actions": len(self.actions_by_time),
            "most_common": action_counts.most_common(10),
            "action_clusters": len(self.action_clusters)
        }
    
    def get_action_timeline_summary(self) -> str:
        """Get a human-readable summary of the action timeline."""
        if not self.actions_by_time:
            return "No actions detected in video."
        
        stats = self.get_action_statistics()
        
        lines = [
            f"📊 Action Analysis Summary",
            f"• Total action detections: {stats['total_actions']}",
            f"• Unique action types: {stats['unique_actions']}",
            f"• Timestamps with actions: {stats['timestamps_with_actions']}",
            f"• Action clusters found: {stats['action_clusters']}",
            "\nMost common actions:"
        ]
        
        for action, count in stats['most_common'][:5]:
            lines.append(f"  • {action}: {count} times")
        
        if self.action_clusters:
            lines.append("\nKey activity periods:")
            for cluster in self.action_clusters[:3]:
                start_min = int(cluster.start_time) // 60
                start_sec = int(cluster.start_time) % 60
                end_min = int(cluster.end_time) // 60
                end_sec = int(cluster.end_time) % 60
                lines.append(
                    f"  • {cluster.description} "
                    f"({start_min}:{start_sec:02d} - {end_min}:{end_sec:02d})"
                )
        
        return "\n".join(lines)
    
    # -------------------------------------------------------------------
    # Cache and serialization
    # -------------------------------------------------------------------
    
    def save_facts_to_cache(self, cache_dir: str = "./cache") -> str:
        """Save inferred facts to cache file."""
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        video_name = Path(self.video_path).stem if self.video_path else "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = Path(cache_dir) / f"{video_name}_facts_{timestamp}.json"
        
        # Collect all facts
        all_facts = []
        for ts, facts in self.fact_cache.items():
            for fact in facts:
                fact_dict = fact.to_dict()
                all_facts.append(fact_dict)
        
        # Prepare output
        output = {
            "video_path": self.video_path,
            "generated": timestamp,
            "total_facts": len(all_facts),
            "action_clusters": [
                {
                    "start": c.start_time,
                    "end": c.end_time,
                    "description": c.description,
                    "density": c.density
                }
                for c in self.action_clusters
            ],
            "facts": all_facts,
            "statistics": self.get_action_statistics()
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return str(cache_file)


# ---------------------------------------------------------------------------
# LLM Integration
# ---------------------------------------------------------------------------

class ReasoningLLMIntegration:
    """
    Integrates reasoning with the LLM chat system.
    Optimized for action-only cache files.
    """
    
    def __init__(self, llm_module, analysis_data: Dict[str, Any], video_path: str = ""):
        self.llm = llm_module
        self.reasoning_engine = VideoReasoningEngine(analysis_data, video_path)
        self.reasoning_enabled = True
        
        # Pre-analyze key timestamps
        self._pre_analyze_key_timestamps()
    
    def _pre_analyze_key_timestamps(self):
        """Pre-analyze important timestamps."""
        # Get all timestamps with actions
        timestamps = list(self.reasoning_engine.actions_by_time.keys())
        
        # Sample every 10th timestamp to avoid too much processing
        for ts in timestamps[::10]:
            self.reasoning_engine.reason_about_timestamp(ts)
    
    def enhance_llm_context(self, timestamp: Optional[float] = None) -> str:
        """
        Add reasoning facts to the LLM context.
        
        Args:
            timestamp: Current timestamp to get facts for
            
        Returns:
            Formatted context string with inferred facts
        """
        if not self.reasoning_enabled or timestamp is None:
            return ""
        
        # Get facts for this timestamp
        facts = self.reasoning_engine.reason_about_timestamp(timestamp)
        
        if not facts:
            return ""
        
        lines = ["\n## INFERRED RELATIONSHIPS (from action patterns)"]
        
        for fact in facts[:3]:
            lines.append(f"  - {fact.description} (confidence: {fact.confidence:.2f})")
            
            if fact.confidence > 0.7:
                for step in fact.reasoning_chain[:1]:
                    lines.append(f"      • {step}")
        
        # Add cluster info if relevant
        current_cluster = None
        for cluster in self.reasoning_engine.action_clusters:
            if cluster.start_time <= timestamp <= cluster.end_time:
                current_cluster = cluster
                break
        
        if current_cluster:
            lines.append(f"\n  Current activity: {current_cluster.description}")
        
        lines.append("")
        return "\n".join(lines)
    
    def answer_why_question(self, question: str, current_time: float) -> str:
        """
        Answer "why" questions by showing the reasoning chain.
        """
        facts = self.reasoning_engine.reason_about_timestamp(current_time)
        
        # Extract key terms from question
        question_lower = question.lower()
        key_terms = [word for word in question_lower.split() 
                    if len(word) > 3 and word not in 
                    ['why', 'how', 'what', 'where', 'when', 'does', 'this', 'that']]
        
        # Find relevant facts
        relevant = []
        for fact in facts:
            if any(term in fact.description.lower() for term in key_terms):
                relevant.append(fact)
            elif any(term in ' '.join(fact.entities).lower() for term in key_terms):
                relevant.append(fact)
        
        if not relevant and facts:
            relevant = facts[:2]
        
        if not relevant:
            # Show action summary instead
            stats = self.reasoning_engine.get_action_statistics()
            return (
                f"At {int(current_time)//60}:{int(current_time)%60:02d}, "
                f"I detect {stats['timestamps_with_actions']} action timestamps. "
                f"The most common actions in the video are: "
                f"{', '.join(a for a, _ in stats['most_common'][:3])}."
            )
        
        explanation = ["🔍 **Based on the action patterns:**\n"]
        for fact in relevant:
            explanation.append(f"• {fact.description}")
            if fact.reasoning_chain:
                explanation.append(f"  → {fact.reasoning_chain[0]}")
            explanation.append("")
        
        return "\n".join(explanation)
    
    def get_action_summary(self) -> str:
        """Get a summary of all actions in the video."""
        return self.reasoning_engine.get_action_timeline_summary()
    
    def save_analysis(self, cache_dir: str = "./cache") -> str:
        """Save reasoning analysis to cache."""
        return self.reasoning_engine.save_facts_to_cache(cache_dir)