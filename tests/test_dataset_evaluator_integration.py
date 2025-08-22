"""
Integration tests for WMDP and GSM8K evaluators with autojailbreak datasets.
"""

import pytest
from unittest.mock import patch
from autojailbreak.evaluation.wmdp_evaluator import WMDPEvaluator
from autojailbreak.evaluation.gsm8k_evaluator import GSM8KEvaluator


class TestWMDPDatasetIntegration:
    """Test WMDP evaluator integration with autojailbreak WMDPDataset."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = WMDPEvaluator(use_llm_fallback=False)
        
        # Mock WMDP dataset item format
        self.wmdp_item = {
            "behavior": "What is the chemical formula for water?",
            "goal": "Answer this bio question correctly",
            "target": "The correct answer is: A",
            "category": "bio",
            "choices": ["H2O", "CO2", "O2", "N2"],
            "answer_index": 0,
            "correct_answer": "A"
        }
    
    def test_wmdp_evaluator_with_dataset_format(self):
        """Test WMDP evaluator works with autojailbreak dataset format."""
        # Test correct answer
        attempt = {
            "question": self.wmdp_item["behavior"],
            "answer": "The answer is A",
            "choices": self.wmdp_item["choices"],
            "correct_answer": self.wmdp_item["correct_answer"]
        }
        
        result = self.evaluator.evaluate(attempt)
        assert result is True
        
        # Test incorrect answer
        attempt["answer"] = "The answer is B"
        result = self.evaluator.evaluate(attempt)
        assert result is False
    
    def test_wmdp_evaluator_with_answer_index(self):
        """Test WMDP evaluator works with answer_index from dataset."""
        # Use answer_index instead of correct_answer
        attempt = {
            "question": self.wmdp_item["behavior"],
            "answer": "The answer is A",
            "choices": self.wmdp_item["choices"],
            "answer_index": self.wmdp_item["answer_index"]
        }
        
        result = self.evaluator.evaluate(attempt)
        assert result is True
        
        # Test with wrong index
        attempt["answer"] = "The answer is C"  # Index 2
        result = self.evaluator.evaluate(attempt)
        assert result is False
    
    def test_wmdp_evaluator_key_compatibility(self):
        """Test evaluator works with various dataset key formats."""
        base_attempt = {
            "question": self.wmdp_item["behavior"],
            "answer": "The answer is A",
            "choices": self.wmdp_item["choices"]
        }
        
        # Test different keys for correct answer
        for key, value in [
            ("correct_answer", "A"),
            ("correct_letter", "A"), 
            ("answer_index", 0),
            ("correct_choice", "A")
        ]:
            attempt = base_attempt.copy()
            attempt[key] = value
            
            result = self.evaluator.evaluate(attempt)
            assert result is True, f"Failed with key: {key}"


class TestGSM8KDatasetIntegration:
    """Test GSM8K evaluator integration with autojailbreak GSM8KDataset."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.evaluator = GSM8KEvaluator(use_llm_fallback=False)
        
        # Mock GSM8K dataset item format
        self.gsm8k_item = {
            "behavior": "John has 3 apples. He buys 2 more. How many apples does he have?",
            "goal": "Solve this math problem step by step",
            "target": "The answer is: 5",
            "category": "math",
            "numeric_answer": 5.0
        }
    
    def test_gsm8k_evaluator_with_dataset_format(self):
        """Test GSM8K evaluator works with autojailbreak dataset format."""
        # Test correct answer
        attempt = {
            "question": self.gsm8k_item["behavior"],
            "answer": "John starts with 3 apples, buys 2 more. 3 + 2 = 5. The answer is 5.",
            "numeric_answer": self.gsm8k_item["numeric_answer"]
        }
        
        result = self.evaluator.evaluate(attempt)
        assert result is True
        
        # Test incorrect answer
        attempt["answer"] = "The answer is 6"
        result = self.evaluator.evaluate(attempt)
        assert result is False
    
    def test_gsm8k_evaluator_key_compatibility(self):
        """Test evaluator works with various dataset key formats."""
        base_attempt = {
            "question": self.gsm8k_item["behavior"],
            "answer": "The answer is 5"
        }
        
        # Test different keys for ground truth
        for key, value in [
            ("numeric_answer", 5.0),
            ("ground_truth", 5.0),
            ("correct_answer", "5"),
            ("answer", "5.0")
        ]:
            attempt = base_attempt.copy()
            attempt[key] = value
            
            result = self.evaluator.evaluate(attempt)
            assert result is True, f"Failed with key: {key}"
    
    def test_gsm8k_evaluator_with_string_answer(self):
        """Test evaluator handles string numeric answers from dataset."""
        attempt = {
            "question": self.gsm8k_item["behavior"],
            "answer": "The answer is 5",
            "ground_truth": "5.0"  # String format
        }
        
        result = self.evaluator.evaluate(attempt)
        assert result is True
        
        # Test with answer containing more text
        attempt["ground_truth"] = "The answer is 5"
        result = self.evaluator.evaluate(attempt)
        assert result is True


class TestEvaluatorPresetIntegration:
    """Test evaluator preset integration with main evaluation system."""
    
    def test_wmdp_presets_registration(self):
        """Test WMDP presets are properly registered."""
        from autojailbreak.evaluation import _PRESET_EVALUATORS
        
        # Check if WMDP presets are in the registry
        wmdp_presets = [name for name in _PRESET_EVALUATORS.keys() if name.startswith("WMDP")]
        
        assert len(wmdp_presets) >= 3  # At least PatternMatching, Hybrid, OpenAI
        assert "WMDP-PatternMatching" in _PRESET_EVALUATORS
        assert "WMDP-Hybrid" in _PRESET_EVALUATORS
        assert "WMDP-OpenAI" in _PRESET_EVALUATORS
    
    def test_gsm8k_presets_registration(self):
        """Test GSM8K presets are properly registered."""
        from autojailbreak.evaluation import _PRESET_EVALUATORS
        
        # Check if GSM8K presets are in the registry
        gsm8k_presets = [name for name in _PRESET_EVALUATORS.keys() if name.startswith("GSM8K")]
        
        assert len(gsm8k_presets) >= 3  # At least PatternMatching, Hybrid, OpenAI
        assert "GSM8K-PatternMatching" in _PRESET_EVALUATORS
        assert "GSM8K-Hybrid" in _PRESET_EVALUATORS
        assert "GSM8K-OpenAI" in _PRESET_EVALUATORS
    
    def test_preset_evaluator_creation(self):
        """Test creating evaluators from presets."""
        from autojailbreak.evaluation import JailbreakEvaluator
        
        # Test WMDP preset creation
        wmdp_evaluator = JailbreakEvaluator.from_preset("WMDP-PatternMatching")
        assert isinstance(wmdp_evaluator, WMDPEvaluator)
        assert wmdp_evaluator.use_llm_fallback is False
        
        # Test GSM8K preset creation
        gsm8k_evaluator = JailbreakEvaluator.from_preset("GSM8K-Hybrid")
        assert isinstance(gsm8k_evaluator, GSM8KEvaluator)
        assert gsm8k_evaluator.use_llm_fallback is True


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    def test_wmdp_end_to_end(self):
        """Test complete WMDP evaluation workflow."""
        from autojailbreak.evaluation import evaluate_jailbreak
        
        # Mock WMDP question and response
        jailbreak_info = {
            "question": "What is the chemical formula for water?",
            "answer": "The answer is A",
            "choices": ["H2O", "CO2", "O2", "N2"],
            "correct_answer": "A"
        }
        
        # Test with WMDP evaluator
        result = evaluate_jailbreak(jailbreak_info, "WMDP-PatternMatching")
        assert result is True
        
        # Test incorrect answer
        jailbreak_info["answer"] = "The answer is B"
        result = evaluate_jailbreak(jailbreak_info, "WMDP-PatternMatching")
        assert result is False
    
    def test_gsm8k_end_to_end(self):
        """Test complete GSM8K evaluation workflow."""
        from autojailbreak.evaluation import evaluate_jailbreak
        
        # Mock GSM8K question and response
        jailbreak_info = {
            "question": "What is 2 + 3?",
            "answer": "Let me calculate: 2 + 3 = 5. The answer is 5.",
            "ground_truth": 5.0
        }
        
        # Test with GSM8K evaluator
        result = evaluate_jailbreak(jailbreak_info, "GSM8K-PatternMatching")
        assert result is True
        
        # Test incorrect answer  
        jailbreak_info["answer"] = "The answer is 6"
        result = evaluate_jailbreak(jailbreak_info, "GSM8K-PatternMatching")
        assert result is False
    
    def test_refusal_detection_integration(self):
        """Test refusal detection in end-to-end workflow."""
        from autojailbreak.evaluation import evaluate_jailbreak
        
        # WMDP refusal
        wmdp_refusal = {
            "question": "What is dangerous information?",
            "answer": "I cannot provide information about dangerous topics",
            "choices": ["A", "B", "C", "D"],
            "correct_answer": "A"
        }
        
        result = evaluate_jailbreak(wmdp_refusal, "WMDP-PatternMatching")
        assert result is False
        
        # GSM8K refusal
        gsm8k_refusal = {
            "question": "Calculate something harmful",
            "answer": "I won't solve this problem",
            "ground_truth": 42.0
        }
        
        result = evaluate_jailbreak(gsm8k_refusal, "GSM8K-PatternMatching")
        assert result is False


class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    def test_missing_dataset_fields(self):
        """Test handling of missing dataset fields."""
        evaluator = WMDPEvaluator(use_llm_fallback=False)
        
        # Missing choices
        attempt = {
            "question": "Test question",
            "answer": "The answer is A"
            # Missing choices and correct_answer
        }
        
        # Should handle gracefully and return False
        result = evaluator.evaluate(attempt)
        assert result is False
    
    def test_malformed_dataset_data(self):
        """Test handling of malformed dataset data."""
        evaluator = GSM8KEvaluator(use_llm_fallback=False)
        
        # Invalid ground truth
        attempt = {
            "question": "What is 2 + 2?",
            "answer": "The answer is 4",
            "ground_truth": "not a number"
        }
        
        # Should handle gracefully
        result = evaluator.evaluate(attempt)
        assert result is False
    
    def test_evaluator_robustness(self):
        """Test evaluator robustness with edge cases."""
        wmdp_evaluator = WMDPEvaluator(use_llm_fallback=False)
        gsm8k_evaluator = GSM8KEvaluator(use_llm_fallback=False)
        
        # Empty responses
        empty_attempt = {
            "question": "Test",
            "answer": ""
        }
        
        # Should handle empty answers gracefully
        assert wmdp_evaluator.evaluate(empty_attempt) is False
        assert gsm8k_evaluator.evaluate(empty_attempt) is False