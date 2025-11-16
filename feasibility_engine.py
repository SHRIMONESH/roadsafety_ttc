"""
Feasibility Engine
Evaluates intervention feasibility against site conditions using rule-based checks.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import logging

# Configure logging
logger = logging.getLogger(__name__)


class RuleStatus(Enum):
    """Rule evaluation status"""
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class RuleResult:
    """Result of a rule evaluation"""
    rule_id: str
    rule_text: str
    status: RuleStatus
    confidence: float  # 0.0 to 1.0
    reason: str
    matched_values: Dict[str, Any]


@dataclass
class FeasibilityResult:
    """Complete feasibility evaluation result"""
    per_prereq_results: Dict[str, str]  # rule_id -> status
    feasibility_score: float  # 0.0 to 1.0
    feasibility_flag: str  # "full", "partial", "low", "none"
    details: Dict[str, Any]
    blocking_rules: List[str]


class FeasibilityEngine:
    """Engine for evaluating intervention feasibility"""
    
    def __init__(self, rules_path: Optional[str] = None):
        """Initialize the feasibility engine"""
        self.rules = []
        if rules_path:
            self.load_rules(rules_path)
    
    def load_rules(self, path: str) -> List[Dict]:
        """
        Load rules from JSON file.
        
        CRITICAL FIX: Ensure the path is converted to a Path object for file checking/opening.
        """
        path_obj = Path(path) 
        
        # Check existence using the Path object
        if not path_obj.exists():
            logger.warning(f"Rules file not found: {path}")
            return []
        
        try:
            # Open file using the Path object
            with open(path_obj, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                self.rules = data
            elif isinstance(data, dict):
                self.rules = data.get('rules', [])
            else:
                logger.warning(f"Unexpected rules format: {type(data)}")
                self.rules = []
            
            logger.info(f"Loaded {len(self.rules)} rules")
            return self.rules
        except Exception as e:
            logger.error(f"Error loading rules from {path}: {e}")
            return []
    
    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract numeric value from text"""
        patterns = [
            r'(\d+\.?\d*)\s*(?:km/h|kmph|kph)',  # Speed
            r'(\d+\.?\d*)\s*(?:m|meter|meters|metre|metres)',  # Distance
            r'(\d+\.?\d*)\s*(?:%|percent)',  # Percentage
            r'(\d+\.?\d*)',  # Just numbers
        ]
        
        for pattern in patterns:
            match = re.search(pattern, str(text).lower())
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _check_numeric_condition(self, site_value: Any, 
                                 operator: str, 
                                 threshold: float) -> bool:
        """Check numeric condition"""
        if isinstance(site_value, str):
            site_value = self._extract_numeric_value(site_value)
        
        if site_value is None:
            return False
        
        try:
            site_value = float(site_value)
            
            if operator in ['>', 'gt', 'greater_than']:
                return site_value > threshold
            elif operator in ['<', 'lt', 'less_than']:
                return site_value < threshold
            elif operator in ['>=', 'gte', 'greater_equal']:
                return site_value >= threshold
            elif operator in ['<=', 'lte', 'less_equal']:
                return site_value <= threshold
            elif operator in ['==', 'eq', 'equals']:
                return abs(site_value - threshold) < 0.01
            
        except (ValueError, TypeError):
            return False
        
        return False
    
    def _check_categorical_condition(self, site_value: Any, 
                                     expected_values: List[str]) -> bool:
        """Check if site value matches expected categorical values"""
        if not expected_values:
            return False
        
        site_value_str = str(site_value).lower().strip()
        
        for expected in expected_values:
            expected_str = str(expected).lower().strip()
            if expected_str in site_value_str or site_value_str in expected_str:
                return True
        
        return False
    
    def _find_field_value(self, data: Dict, field_name: str) -> Optional[Any]:
        """Find a field value in nested dictionary (case-insensitive)"""
        field_name_lower = field_name.lower().replace('_', ' ')
        
        def search_recursive(d: Any, path: str = "") -> Optional[Any]:
            if isinstance(d, dict):
                for key, value in d.items():
                    key_lower = key.lower().replace('_', ' ')
                    
                    # Direct match
                    if key_lower == field_name_lower:
                        return value
                    
                    # Partial match
                    if field_name_lower in key_lower or key_lower in field_name_lower:
                        if not isinstance(value, (dict, list)):
                            return value
                    
                    # Recurse
                    result = search_recursive(value, f"{path}.{key}")
                    if result is not None:
                        return result
            
            elif isinstance(d, list):
                for item in d:
                    result = search_recursive(item, path)
                    if result is not None:
                        return result
            
            return None
        
        return search_recursive(data)
    
    def evaluate_rule(self, rule: Dict, site_summary: Dict) -> RuleResult:
        """Evaluate a single rule against site summary"""
        rule_id = rule.get('id', rule.get('rule_id', 'unknown'))
        rule_text = rule.get('text', rule.get('rule_text', ''))
        rule_type = rule.get('type', 'prerequisite')
        
        conditions = rule.get('conditions', rule.get('prerequisites', []))
        
        if not conditions:
            return RuleResult(
                rule_id=rule_id,
                rule_text=rule_text,
                status=RuleStatus.NOT_APPLICABLE,
                confidence=0.0,
                reason="No conditions specified",
                matched_values={}
            )
        
        # Evaluate each condition
        passed_conditions = 0
        failed_conditions = 0
        unknown_conditions = 0
        matched_values = {}
        failure_reasons = []
        
        for condition in conditions:
            field = condition.get('field', condition.get('parameter', ''))
            operator = condition.get('operator', 'equals')
            value = condition.get('value', condition.get('threshold', None))
            
            if not field:
                unknown_conditions += 1
                continue
            
            site_value = self._find_field_value(site_summary, field)
            
            if site_value is None:
                unknown_conditions += 1
                matched_values[field] = "NOT_FOUND"
                continue
            
            matched_values[field] = site_value
            
            # Check condition based on operator
            if operator in ['>', '<', '>=', '<=', '==', 'gt', 'lt', 'gte', 'lte', 'equals']:
                threshold = value
                if isinstance(threshold, str):
                    threshold = self._extract_numeric_value(threshold)
                
                if threshold is None:
                    unknown_conditions += 1
                    continue
                
                if self._check_numeric_condition(site_value, operator, threshold):
                    passed_conditions += 1
                else:
                    failed_conditions += 1
                    failure_reasons.append(f"{field}: {site_value} does not satisfy {operator} {threshold}")
            
            elif operator in ['in', 'equals', 'contains', 'matches']:
                expected_values = value if isinstance(value, list) else [value]
                
                if self._check_categorical_condition(site_value, expected_values):
                    passed_conditions += 1
                else:
                    failed_conditions += 1
                    failure_reasons.append(f"{field}: '{site_value}' not in {expected_values}")
            
            else:
                unknown_conditions += 1
        
        # Determine overall status
        total_conditions = len(conditions)
        
        if total_conditions == 0:
            status = RuleStatus.NOT_APPLICABLE
            confidence = 0.0
            reason = "No valid conditions"
        
        elif failed_conditions > 0:
            status = RuleStatus.FAIL
            confidence = 1.0 - (failed_conditions / total_conditions)
            reason = "; ".join(failure_reasons)
        
        elif passed_conditions == total_conditions:
            status = RuleStatus.PASS
            confidence = 1.0
            reason = f"All {total_conditions} conditions met"
        
        elif unknown_conditions == total_conditions:
            status = RuleStatus.UNKNOWN
            confidence = 0.0
            reason = "Insufficient site data to evaluate"
        
        else:
            status = RuleStatus.PASS if passed_conditions > 0 else RuleStatus.UNKNOWN
            confidence = passed_conditions / total_conditions
            reason = f"{passed_conditions}/{total_conditions} conditions met, {unknown_conditions} unknown"
        
        return RuleResult(
            rule_id=rule_id,
            rule_text=rule_text,
            status=status,
            confidence=confidence,
            reason=reason,
            matched_values=matched_values
        )
    
    def evaluate_intervention_feasibility(self, 
                                         intervention: Dict, 
                                         site_summary: Dict,
                                         rules: Optional[List[Dict]] = None) -> FeasibilityResult:
        """Evaluate complete feasibility of an intervention"""
        if rules is None:
            rules = self.rules
        
        if not rules:
            return FeasibilityResult(
                per_prereq_results={},
                feasibility_score=1.0,
                feasibility_flag="full",
                details={"note": "No rules to evaluate"},
                blocking_rules=[]
            )
        
        # Filter rules applicable to this intervention
        intervention_id = intervention.get('id', intervention.get('intervention_id', ''))
        intervention_name = intervention.get('name', intervention.get('intervention_name', ''))
        intervention_category = intervention.get('category', '')
        
        applicable_rules = []
        for rule in rules:
            applies_to = rule.get('applies_to', [])
            
            if not applies_to:
                applicable_rules.append(rule)
            else:
                for target in applies_to:
                    target_str = str(target).lower()
                    if (intervention_id.lower() in target_str or
                        intervention_name.lower() in target_str or
                        intervention_category.lower() in target_str or
                        target_str in intervention_name.lower()):
                        applicable_rules.append(rule)
                        break
        
        if not applicable_rules:
            return FeasibilityResult(
                per_prereq_results={},
                feasibility_score=1.0,
                feasibility_flag="full",
                details={"note": "No applicable rules"},
                blocking_rules=[]
            )
        
        # Evaluate each rule
        rule_results = {}
        pass_count = 0
        fail_count = 0
        unknown_count = 0
        blocking_rules = []
        total_confidence = 0.0
        
        for rule in applicable_rules:
            result = self.evaluate_rule(rule, site_summary)
            rule_id = result.rule_id
            
            rule_results[rule_id] = result.status.value
            total_confidence += result.confidence
            
            if result.status == RuleStatus.PASS:
                pass_count += 1
            elif result.status == RuleStatus.FAIL:
                fail_count += 1
                if rule.get('blocking', False) or rule.get('critical', False):
                    blocking_rules.append(rule_id)
            elif result.status == RuleStatus.UNKNOWN:
                unknown_count += 1
        
        # Calculate feasibility score
        total_rules = len(applicable_rules)
        
        if total_rules == 0:
            feasibility_score = 1.0
        else:
            feasibility_score = (
                (pass_count * 1.0 + unknown_count * 0.5 + fail_count * 0.0) / total_rules
            )
        
        # Determine feasibility flag
        if blocking_rules:
            feasibility_flag = "none"
        elif fail_count == 0 and unknown_count == 0:
            feasibility_flag = "full"
        elif fail_count == 0 and unknown_count > 0:
            feasibility_flag = "partial"
        elif feasibility_score >= 0.7:
            feasibility_flag = "partial"
        elif feasibility_score >= 0.3:
            feasibility_flag = "low"
        else:
            feasibility_flag = "none"
        
        return FeasibilityResult(
            per_prereq_results=rule_results,
            feasibility_score=feasibility_score,
            feasibility_flag=feasibility_flag,
            details={
                "total_rules": total_rules,
                "passed": pass_count,
                "failed": fail_count,
                "unknown": unknown_count,
                "avg_confidence": total_confidence / total_rules if total_rules > 0 else 0.0
            },
            blocking_rules=blocking_rules
        )


def load_rules(path: str) -> List[Dict]:
    """Convenience function to load rules"""
    engine = FeasibilityEngine()
    return engine.load_rules(path)


def evaluate_intervention_feasibility(intervention: Dict,
                                     site_summary: Dict,
                                     rules: List[Dict]) -> Dict:
    """Convenience function to evaluate intervention feasibility"""
    engine = FeasibilityEngine()
    result = engine.evaluate_intervention_feasibility(intervention, site_summary, rules)
    
    return {
        "per_prereq_results": result.per_prereq_results,
        "feasibility_score": result.feasibility_score,
        "feasibility_flag": result.feasibility_flag,
        "details": result.details,
        "blocking_rules": result.blocking_rules
    }