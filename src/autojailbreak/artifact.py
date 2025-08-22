"""
Artifact handling for jailbreak results.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union


@dataclass
class JailbreakInfo:
    """Information about a specific jailbreak instance."""
    
    index: int
    goal: str
    behavior: str
    category: str
    prompt: str
    response: str
    number_of_queries: int = 1
    queries_to_jailbreak: int = 1
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    jailbroken: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "index": self.index,
            "goal": self.goal,
            "behavior": self.behavior,
            "category": self.category,
            "prompt": self.prompt,
            "response": self.response,
            "number_of_queries": self.number_of_queries,
            "queries_to_jailbreak": self.queries_to_jailbreak,
            "prompt_tokens": self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "jailbroken": self.jailbroken
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JailbreakInfo":
        """Create a JailbreakInfo instance from a dictionary."""
        return cls(
            index=data["index"],
            goal=data["goal"],
            behavior=data["behavior"],
            category=data["category"],
            prompt=data["prompt"],
            response=data["response"],
            number_of_queries=data.get("number_of_queries", 1),
            queries_to_jailbreak=data.get("queries_to_jailbreak", 1),
            prompt_tokens=data.get("prompt_tokens"),
            response_tokens=data.get("response_tokens"),
            jailbroken=data.get("jailbroken", False)
        )


@dataclass
class ArtifactParameters:
    """Parameters of a jailbreak artifact."""
    
    method: str
    model_name: str
    attack_success_rate: Optional[float] = None
    median_queries: Optional[float] = None
    max_queries: Optional[int] = None
    date: Optional[str] = None
    authors: Optional[List[str]] = None
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        result = {
            "method": self.method,
            "model_name": self.model_name,
        }
        
        if self.attack_success_rate is not None:
            result["attack_success_rate"] = self.attack_success_rate
        
        if self.median_queries is not None:
            result["median_queries"] = self.median_queries
        
        if self.max_queries is not None:
            result["max_queries"] = self.max_queries
        
        if self.date is not None:
            result["date"] = self.date
        
        if self.authors is not None:
            result["authors"] = self.authors
        
        if self.additional_params:
            result.update(self.additional_params)
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactParameters":
        """Create an ArtifactParameters instance from a dictionary."""
        # Extract known fields
        known_fields = {
            "method", "model_name", "attack_success_rate", 
            "median_queries", "max_queries", "date", "authors"
        }
        
        params = {k: data[k] for k in known_fields if k in data}
        
        # Everything else goes to additional_params
        additional_params = {k: v for k, v in data.items() if k not in known_fields}
        
        return cls(**params, additional_params=additional_params)


@dataclass
class JailbreakArtifact:
    """A collection of jailbreaks and their parameters."""
    
    parameters: ArtifactParameters
    jailbreaks: List[JailbreakInfo]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "parameters": self.parameters.to_dict(),
            "jailbreaks": [jb.to_dict() for jb in self.jailbreaks]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JailbreakArtifact":
        """Create a JailbreakArtifact instance from a dictionary."""
        parameters = ArtifactParameters.from_dict(data["parameters"])
        jailbreaks = [JailbreakInfo.from_dict(jb) for jb in data["jailbreaks"]]
        return cls(parameters=parameters, jailbreaks=jailbreaks)
    
    def save(self, file_path: str) -> None:
        """
        Save the artifact to a JSON file.
        
        Args:
            file_path: Path to save the artifact
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> "JailbreakArtifact":
        """
        Load an artifact from a JSON file.
        
        Args:
            file_path: Path to the artifact file
            
        Returns:
            JailbreakArtifact instance
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def get_jailbreak_by_behavior(self, behavior: str) -> Optional[JailbreakInfo]:
        """
        Get the jailbreak for a specific behavior.
        
        Args:
            behavior: The behavior to look for
            
        Returns:
            JailbreakInfo for the behavior, or None if not found
        """
        for jailbreak in self.jailbreaks:
            if jailbreak.behavior.lower() == behavior.lower():
                return jailbreak
        return None
    
    def get_attack_success_rate(self) -> float:
        """
        Calculate the attack success rate.
        
        Returns:
            The percentage of successful jailbreaks
        """
        success_count = sum(1 for jb in self.jailbreaks if jb.jailbroken)
        return success_count / len(self.jailbreaks) if self.jailbreaks else 0.0
    
    def get_median_queries(self) -> float:
        """
        Calculate the median number of queries to jailbreak.
        
        Returns:
            The median number of queries for successful jailbreaks
        """
        queries = [jb.queries_to_jailbreak for jb in self.jailbreaks if jb.jailbroken]
        if not queries:
            return 0.0
        
        queries.sort()
        n = len(queries)
        if n % 2 == 0:
            return (queries[n // 2 - 1] + queries[n // 2]) / 2.0
        else:
            return float(queries[n // 2])


def read_artifact(method: str, model_name: str, artifact_dir: Optional[str] = None) -> JailbreakArtifact:
    """
    Read a jailbreak artifact from a file.
    
    Args:
        method: The jailbreak method
        model_name: The target model
        artifact_dir: Directory containing artifact files. If None, uses default location.
        
    Returns:
        JailbreakArtifact instance
    """
    if artifact_dir is None:
        # Use default path
        artifact_dir = os.path.expanduser("~/.cache/autojailbreak/artifacts")
    
    file_path = os.path.join(artifact_dir, f"{method}_{model_name}.json")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artifact file not found: {file_path}")
    
    return JailbreakArtifact.load(file_path) 